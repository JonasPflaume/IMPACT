#include "impact/subproblem_spec.h"

#include <casadi/casadi.hpp>
#include <stdexcept>

namespace impact {

namespace {

casadi::Function deserialize(const std::string& blob, const char* what) {
    if (blob.empty())
        throw std::invalid_argument(std::string("buildFromSpec: missing required function '") +
                                    what + "'");
    return casadi::Function::deserialize(blob);
}

// Closure for a CasADi function (z, p_full) -> v, reading p_full from the buffer.
// Mirrors evalClosureP in mpcc_subproblem.cpp so both build paths report the same
// numbers from the same evaluation.
std::function<Eigen::VectorXd(const Eigen::VectorXd&)> evalClosureP(casadi::Function f,
                                                                    AulaSubproblem* sp) {
    return [f, sp](const Eigen::VectorXd& z) -> Eigen::VectorXd {
        casadi::DM zdm(std::vector<double>(z.data(), z.data() + z.size()));
        casadi::DM out = f(std::vector<casadi::DM>{zdm, sp->params()})[0];
        return Eigen::Map<const Eigen::VectorXd>(out.ptr(), out.numel());
    };
}

void checkRange(int offset, int count, int n_params, const std::string& what) {
    if (offset < 0 || count < 0 || offset + count > n_params)
        throw std::invalid_argument("buildFromSpec: parameter offset for '" + what +
                                    "' falls outside the parameter buffer");
}

}  // namespace

std::unique_ptr<AulaSubproblem> buildFromSpec(const SubproblemSpec& spec) {
    if (spec.n_opt <= 0)
        throw std::invalid_argument("buildFromSpec: n_opt must be positive");
    if (spec.n_params < 0)
        throw std::invalid_argument("buildFromSpec: n_params must be nonnegative");

    auto sub = std::make_unique<AulaSubproblem>();
    sub->setFunctions(deserialize(spec.residual, "residual"),
                      deserialize(spec.jacobian, "jacobian"), deserialize(spec.gh, "gh"),
                      spec.n_opt, spec.n_params);
    sub->setTerminationOptions(spec.check_stationarity, spec.conditioned_complementarity,
                               spec.stationarity_tol, spec.max_stagnation_restarts);
    sub->setSaddleLayout(spec.layout);

    for (const auto& kv : spec.param_values) {
        checkRange(kv.first, static_cast<int>(kv.second.size()), spec.n_params, "param_values");
        Eigen::VectorXd v =
            Eigen::Map<const Eigen::VectorXd>(kv.second.data(), kv.second.size());
        sub->setParamValue(kv.first, v);
    }

    AulaSubproblem* sp = sub.get();

    std::vector<DualBlock>& blocks = sub->dualBlocks();
    for (const DualBlockSpec& d : spec.dual_blocks) {
        checkRange(d.kappa_offset, d.dim, spec.n_params, d.name + ".kappa");
        checkRange(d.rho_offset, 1, spec.n_params, d.name + ".rho");
        DualBlock b;
        b.name = d.name;
        b.kind = d.inequality ? DualKind::Inequality : DualKind::Equality;
        b.dim = d.dim;
        b.scale = d.scale;
        b.tol = d.tol;
        b.rho = d.rho_init;
        b.rho_init = d.rho_init;
        b.kappa = Eigen::VectorXd::Zero(d.dim);
        b.kappa_offset = d.kappa_offset;
        b.rho_offset = d.rho_offset;
        b.eval_scaled = evalClosureP(deserialize(d.eval_scaled, "eval_scaled"), sp);
        blocks.push_back(std::move(b));
    }

    std::vector<CompBlock>& comps = sub->compBlocks();
    for (const CompBlockSpec& c : spec.comp_blocks) {
        checkRange(c.kappaG_offset, c.dim, spec.n_params, c.name + ".kappaG");
        checkRange(c.kappaH_offset, c.dim, spec.n_params, c.name + ".kappaH");
        checkRange(c.sG_offset, c.dim, spec.n_params, c.name + ".sG");
        checkRange(c.sH_offset, c.dim, spec.n_params, c.name + ".sH");
        checkRange(c.rho_offset, 1, spec.n_params, c.name + ".rho");
        CompBlock b;
        b.name = c.name;
        b.dim = c.dim;
        b.scale = c.scale;
        b.tol = c.tol;
        b.rho = c.rho_init;
        b.rho_init = c.rho_init;
        b.kappaG = Eigen::VectorXd::Zero(c.dim);
        b.kappaH = Eigen::VectorXd::Zero(c.dim);
        b.sG = Eigen::VectorXd::Zero(c.dim);
        b.sH = Eigen::VectorXd::Zero(c.dim);
        b.kappaG_offset = c.kappaG_offset;
        b.kappaH_offset = c.kappaH_offset;
        b.sG_offset = c.sG_offset;
        b.sH_offset = c.sH_offset;
        b.rho_offset = c.rho_offset;
        comps.push_back(std::move(b));
    }

    if (!spec.obj.empty()) {
        casadi::Function obj_func = casadi::Function::deserialize(spec.obj);
        sub->setObjective([obj_func, sp](const Eigen::VectorXd& zv) -> double {
            casadi::DM zdm(std::vector<double>(zv.data(), zv.data() + zv.size()));
            return static_cast<double>(obj_func(std::vector<casadi::DM>{zdm, sp->params()})[0]);
        });
    }
    if (!spec.obj_grad.empty()) {
        casadi::Function g = casadi::Function::deserialize(spec.obj_grad);
        sub->setObjectiveGradient([g, sp](const Eigen::VectorXd& zv) -> Eigen::VectorXd {
            casadi::DM zdm(std::vector<double>(zv.data(), zv.data() + zv.size()));
            casadi::DM out = g(std::vector<casadi::DM>{zdm, sp->params()})[0];
            return Eigen::Map<const Eigen::VectorXd>(out.ptr(), out.numel());
        });
    }
    if (!spec.stationarity.empty()) {
        casadi::Function s = casadi::Function::deserialize(spec.stationarity);
        sub->setStationarityEvaluator([s, sp](const Eigen::VectorXd& zv) -> double {
            casadi::DM zdm(std::vector<double>(zv.data(), zv.data() + zv.size()));
            casadi::DM out = s(std::vector<casadi::DM>{zdm, sp->params()})[0];
            const Eigen::Map<const Eigen::VectorXd> gv(out.ptr(), out.numel());
            return gv.size() ? gv.lpNorm<Eigen::Infinity>() : 0.0;
        });
    }

    return sub;
}

}  // namespace impact
