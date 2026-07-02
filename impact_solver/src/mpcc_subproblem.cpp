#include "impact/mpcc_subproblem.h"

#include <Eigen/Core>
#include <cmath>
#include <string>

namespace impact {

using casadi::SX;

namespace {

// Closure for a CasADi function (z, p_full) -> v, reading p_full from the buffer.
std::function<Eigen::VectorXd(const Eigen::VectorXd&)> evalClosureP(casadi::Function f,
                                                                    AulaSubproblem* sp) {
    return [f, sp](const Eigen::VectorXd& z) -> Eigen::VectorXd {
        casadi::DM zdm(std::vector<double>(z.data(), z.data() + z.size()));
        casadi::DM out = f(std::vector<casadi::DM>{zdm, sp->params()})[0];
        return Eigen::Map<const Eigen::VectorXd>(out.ptr(), out.numel());
    };
}

}  // namespace

MPCCSubproblem buildMPCC(const MPCCDescription& desc) {
    const SX& z = desc.z;
    const int n_opt = static_cast<int>(z.size1());
    const int np = desc.p.is_empty() ? 0 : static_cast<int>(desc.p.size1());
    const SX& cost = desc.cost;
    const int n_cost_rows = static_cast<int>(cost.size1());
    const bool linear = desc.cost_is_linear;

    // Symbolic AuLa parameters and offsets into the flat parameter buffer.
    std::vector<SX> psyms;
    int off = 0;
    auto addParam = [&](const SX& s) {
        psyms.push_back(s);
        const int o = off;
        off += static_cast<int>(s.size1());
        return o;
    };
    const int off_rho_one = addParam(SX::sym("rho_one", 1));

    // Per-constraint metadata, kept in residual-row order.
    struct DualInfo {
        bool inequality;
        std::string name;
        int dim;
        double scale, rho, tol;
        int off_kappa, off_rho;
        SX scaled;  // scale * c
    };
    std::vector<DualInfo> duals;
    struct CompInfo {
        std::string name;
        int dim = 0;
        double scale = 1.0, rho = 1.0, tol = 1e-5;
        int off_sG = 0, off_sH = 0, off_kappaG = 0, off_kappaH = 0, off_rho = 0;
        SX G, H;
    };
    std::vector<CompInfo> comps;
    // Saddle blocks (count, rho_offset), again in residual-row order.
    std::vector<std::pair<int, int>> saddle_blocks;

    std::vector<SX> res{cost};
    for (const MPCCConstraint& cc : desc.constraints) {
        if (cc.kind == MPCCConstraint::Complementarity) {
            const int n = static_cast<int>(cc.G.size1());
            const std::string suffix = std::to_string(comps.size());
            const int off_sG = addParam(SX::sym("sG_" + suffix, n));
            const int off_sH = addParam(SX::sym("sH_" + suffix, n));
            const int off_kappaG = addParam(SX::sym("kappaG_" + suffix, n));
            const int off_kappaH = addParam(SX::sym("kappaH_" + suffix, n));
            const int off_rho = addParam(SX::sym("rho_comp_" + suffix, 1));
            const SX sG = psyms[psyms.size() - 5];
            const SX sH = psyms[psyms.size() - 4];
            const SX kappaG = psyms[psyms.size() - 3];
            const SX kappaH = psyms[psyms.size() - 2];
            const SX rho = psyms[psyms.size() - 1];
            res.push_back(SX::sqrt(rho) * (cc.scale * (cc.G - sG) + kappaG / rho));
            res.push_back(SX::sqrt(rho) * (cc.scale * (cc.H - sH) + kappaH / rho));
            comps.push_back({cc.name.empty() ? ("comp_" + suffix) : cc.name,
                             n,
                             cc.scale,
                             cc.rho_init,
                             cc.tol,
                             off_sG,
                             off_sH,
                             off_kappaG,
                             off_kappaH,
                             off_rho,
                             cc.G,
                             cc.H});
            saddle_blocks.push_back({n, off_rho});
            saddle_blocks.push_back({n, off_rho});
        } else {
            const int n = static_cast<int>(cc.c.size1());
            const int off_kappa = addParam(SX::sym("kappa_" + cc.name, n));
            const int off_rho = addParam(SX::sym("rho_" + cc.name, 1));
            const SX kappa = psyms[psyms.size() - 2];
            const SX rho = psyms[psyms.size() - 1];
            const bool ineq = (cc.kind == MPCCConstraint::Inequality);
            if (ineq)
                res.push_back(SX::sqrt(rho) * SX::fmax(cc.scale * cc.c + kappa / rho, 0.0));
            else
                res.push_back(SX::sqrt(rho) * (cc.scale * cc.c + kappa / rho));
            duals.push_back({ineq, cc.name, n, cc.scale, cc.rho_init, cc.tol, off_kappa, off_rho,
                             cc.scale * cc.c});
            saddle_blocks.push_back({n, off_rho});
        }
    }

    const int off_p = off;
    if (np > 0) {
        psyms.push_back(desc.p);
        off += np;
    }
    const int n_params = off;
    const SX p_full = SX::vertcat(psyms);

    // CasADi functions. With desc.jit they are compiled through the shell C
    // compiler instead of interpreted by the VM.
    casadi::Dict fopts;
    if (desc.jit) {
        fopts["jit"] = true;
        fopts["compiler"] = "shell";
        fopts["jit_cleanup"] = true;
        // -O0: SX codegen emits a huge flat C file; gcc -O2/-O3 on it is pathologically
        // slow (allegro: ~37s for -O1, minutes for -O3). -O0 compiles in ~4s and still
        // gives ~+15% MPC frequency over the VM; -O1 gives ~+38% if the longer build is
        // acceptable. For repeated runs, cache the generated .so instead of recompiling.
        fopts["jit_options"] =
            casadi::Dict{{"compiler", std::string("gcc")},
                         {"flags", std::vector<std::string>{"-O0"}}};
    }
    casadi::Dict jopt = fopts;
    jopt["enable_fd"] = false;
    const SX r = SX::vertcat(res);
    casadi::Function residual_func("mpcc_residual", {z, p_full}, {r}, fopts);
    casadi::Function jacobian_func("mpcc_jac", {z, p_full}, {SX::jacobian(r, z)}, jopt);
    std::vector<SX> all_G, all_H;
    for (const CompInfo& c : comps) {
        all_G.push_back(c.G);
        all_H.push_back(c.H);
    }
    casadi::Function gh_func =
        !comps.empty()
            ? casadi::Function("GH", {z, p_full}, {SX::vertcat(all_G), SX::vertcat(all_H)},
                               fopts)
            : casadi::Function("GH", {z, p_full}, {SX::zeros(0, 1), SX::zeros(0, 1)}, fopts);
    casadi::Function obj_func("obj", {z, p_full}, {SX::sumsqr(cost)}, fopts);

    auto sub = std::make_unique<AulaSubproblem>();
    sub->setFunctions(residual_func, jacobian_func, gh_func, n_opt, n_params);
    sub->setParamValue(off_rho_one, Eigen::VectorXd::Constant(1, 1.0));
    AulaSubproblem* sp = sub.get();

    // Saddle layout: linear cost rows are constant; nonlinear cost rows use rho=1.
    SaddleLayout L;
    L.n_z = n_opt;
    L.n_cost = linear ? n_cost_rows : 0;
    int row = L.n_cost;
    auto addSaddle = [&](int count, int rho_off) {
        if (count > 0) {
            L.blocks.push_back({row, count, rho_off});
            row += count;
        }
    };
    if (!linear) addSaddle(n_cost_rows, off_rho_one);
    for (const auto& b : saddle_blocks) addSaddle(b.first, b.second);
    L.n_dual = row - L.n_cost;
    sub->setSaddleLayout(L);

    // Standard equality / inequality dual blocks.
    std::vector<DualBlock>& blocks = sub->dualBlocks();
    for (const DualInfo& d : duals) {
        DualBlock b;
        b.name = d.name;
        b.kind = d.inequality ? DualKind::Inequality : DualKind::Equality;
        b.dim = d.dim;
        b.scale = d.scale;
        b.tol = d.tol;
        b.rho = d.rho;
        b.rho_init = d.rho;
        b.kappa = Eigen::VectorXd::Zero(d.dim);
        b.kappa_offset = d.off_kappa;
        b.rho_offset = d.off_rho;
        casadi::Function f("eval_" + d.name, {z, p_full}, {d.scaled});
        b.eval_scaled = evalClosureP(f, sp);
        blocks.push_back(std::move(b));
    }

    // Complementarity blocks.
    std::vector<CompBlock>& comp_blocks = sub->compBlocks();
    for (const CompInfo& info : comps) {
        CompBlock c;
        c.name = info.name;
        c.dim = info.dim;
        c.scale = info.scale;
        c.tol = info.tol;
        c.rho = info.rho;
        c.rho_init = info.rho;
        c.kappaG = Eigen::VectorXd::Zero(info.dim);
        c.kappaH = Eigen::VectorXd::Zero(info.dim);
        c.sG = Eigen::VectorXd::Zero(info.dim);
        c.sH = Eigen::VectorXd::Zero(info.dim);
        c.kappaG_offset = info.off_kappaG;
        c.kappaH_offset = info.off_kappaH;
        c.sG_offset = info.off_sG;
        c.sH_offset = info.off_sH;
        c.rho_offset = info.off_rho;
        comp_blocks.push_back(std::move(c));
    }

    // Task objective reported to callers: ||cost||^2.
    sub->setObjective([obj_func, sp](const Eigen::VectorXd& zv) -> double {
        casadi::DM zdm(std::vector<double>(zv.data(), zv.data() + zv.size()));
        return static_cast<double>(obj_func(std::vector<casadi::DM>{zdm, sp->params()})[0]);
    });

    MPCCSubproblem out;
    out.sub = std::move(sub);
    out.off_p = off_p;
    return out;
}

}  // namespace impact
