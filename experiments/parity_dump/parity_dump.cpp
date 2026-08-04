/**
 * @file parity_dump.cpp
 * @brief Dump a C++-assembled subproblem so the Python build can be checked against it.
 *
 * The Python modelling layer re-derives, in Python, what the C++ builders derive
 * in C++: the same augmented-Lagrangian rows, the same parameter-buffer offsets,
 * the same saddle layout. "The trajectories look similar" is not evidence that it
 * did -- these are nonconvex solves over hundreds of outer iterations, where a
 * genuine formulation difference and a last-bit rounding difference both show up
 * as a slightly different iteration count.
 *
 * So the comparison is made one level down, on the artefacts themselves. This
 * tool writes out the CasADi functions the C++ builder emitted plus the metadata
 * it chose; `python/tests/test_parity.py` rebuilds the same task through the
 * Python path and checks that the metadata matches exactly and that the residual
 * and Jacobian agree to the last bit at random points. That is a statement about
 * the formulation, independent of how either side was compiled.
 *
 * Usage: parity_dump <task> <multiple|single|direct> <horizon> <outdir>
 *
 * `direct` is for tasks that assemble an MPCCDescription themselves instead of
 * going through a shooting builder.
 */

#include <casadi/casadi.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "impact/mpcc_stage.h"
#include "impact/multiple_shooting.h"
#include "impact/single_shooting.h"
#include "push_circle.h"
#include "box_pushing.h"
#include "push_t.h"
#include "cart_transporter.h"

namespace {

void writeJson(const std::string& path, const impact::AulaSubproblem& sub, int n_opt,
               int n_params, int off_p, int off_x0) {
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "{\n";
    f << "  \"n_opt\": " << n_opt << ",\n";
    f << "  \"n_params\": " << n_params << ",\n";
    f << "  \"off_p\": " << off_p << ",\n";
    f << "  \"off_x0\": " << off_x0 << ",\n";

    const impact::SaddleLayout& L = sub.saddleLayout();
    f << "  \"layout\": {\"n_z\": " << L.n_z << ", \"n_cost\": " << L.n_cost
      << ", \"n_dual\": " << L.n_dual << ", \"blocks\": [";
    for (size_t i = 0; i < L.blocks.size(); ++i) {
        if (i) f << ", ";
        f << "[" << L.blocks[i].row_start << ", " << L.blocks[i].count << ", "
          << L.blocks[i].rho_param_offset << "]";
    }
    f << "]},\n";

    f << "  \"dual_blocks\": [";
    const auto& duals = sub.dualBlocks();
    for (size_t i = 0; i < duals.size(); ++i) {
        const impact::DualBlock& b = duals[i];
        if (i) f << ",";
        f << "\n    {\"name\": \"" << b.name << "\", \"inequality\": "
          << (b.kind == impact::DualKind::Inequality ? "true" : "false") << ", \"dim\": " << b.dim
          << ", \"scale\": " << b.scale << ", \"rho_init\": " << b.rho_init
          << ", \"tol\": " << b.tol << ", \"kappa_offset\": " << b.kappa_offset
          << ", \"rho_offset\": " << b.rho_offset << "}";
    }
    f << "\n  ],\n";

    f << "  \"comp_blocks\": [";
    const auto& comps = sub.compBlocks();
    for (size_t i = 0; i < comps.size(); ++i) {
        const impact::CompBlock& c = comps[i];
        if (i) f << ",";
        f << "\n    {\"name\": \"" << c.name << "\", \"dim\": " << c.dim << ", \"scale\": "
          << c.scale << ", \"rho_init\": " << c.rho_init << ", \"tol\": " << c.tol
          << ", \"sG_offset\": " << c.sG_offset << ", \"sH_offset\": " << c.sH_offset
          << ", \"kappaG_offset\": " << c.kappaG_offset
          << ", \"kappaH_offset\": " << c.kappaH_offset << ", \"rho_offset\": "
          << c.rho_offset << "}";
    }
    f << "\n  ]\n}\n";
}

/// Each task's tuned config, copied verbatim from its driver. The point of the
/// comparison is the formulation the drivers actually run, so the numbers here
/// have to be the drivers' numbers, not library defaults.
impact::AulaConfig configFor(const std::string& task, int horizon) {
    impact::AulaConfig config;
    config.horizon = horizon;

    if (task == "push_circle") {
        const double ang = 225.0 * M_PI / 180.0, D = 1.5;
        const double tx = D * std::cos(ang), ty = D * std::sin(ang);
        config.x_0 = Eigen::Vector4d(0.0, 0.0, tx, ty);
        config.x_goal = Eigen::Vector4d(tx, ty, tx, ty);
        config.use_constant_state_init = true;
        config.stage_cost_weight = 1e-2;
        config.final_cost_weight = 100.0;
        config.rho_max = 200.0;
        config.rho_scale = 1.05;
        config.fix_point_scale = config.dynamics_scale = config.eq_scale = config.ineq_scale = 10.0;
        config.comp_scale = 1.0;
        config.max_outer_iters = 800;
        config.max_inner_iters = 50;
        config.newton_max_iter = 100;
        config.newton_tol = 1e-5;
    } else if (task == "box") {
        config.stage_cost_weight = 0.001;
        config.final_cost_weight = 100.0;
        config.rho_max = 200.0;
        config.rho_scale = 1.05;
        config.fix_point_scale = config.dynamics_scale = config.eq_scale = config.ineq_scale = 25.0;
        config.comp_scale = 0.1;
        config.max_outer_iters = 500;
        config.outer_tol_h = 1e-5;
        config.outer_tol_comp = 1e-5;
        config.max_inner_iters = 50;
        config.newton_max_iter = 50;
        config.newton_tol = 1e-6;
        config.newton_regularization = 2e-5;
    } else if (task == "push_t") {
        config.stage_cost_weight = 0.01;
        config.stage_state_cost_weight = 0.0;
        config.control_rate_weight = 0.0;
        config.final_cost_weight = 100.0;
        config.rho_max = 1000.0;
        config.rho_scale = 1.05;
        config.fix_point_scale = config.dynamics_scale = config.eq_scale = config.ineq_scale = 25.0;
        config.comp_scale = 0.1;
        config.max_outer_iters = 1000;
        config.outer_tol_h = config.outer_tol_comp = config.outer_tol_g = 1e-5;
        config.max_inner_iters = 50;
        config.newton_max_iter = 200;
        config.newton_tol = 1e-6;
        config.newton_regularization = 5e-5;
    } else if (task == "cart_transporter") {
        config.stage_cost_weight = 1e-6;
        config.stage_state_cost_weight = 0.0;
        config.final_cost_weight = 5000.0;
        config.rho_max = 100000.0;
        config.rho_scale = 1.5;
        config.fix_point_scale = config.dynamics_scale = config.eq_scale = config.ineq_scale = 1.0;
        config.comp_scale = 0.002;
        config.max_outer_iters = 1000;
        config.outer_tol_h = config.outer_tol_comp = config.outer_tol_g = 1e-5;
        config.max_inner_iters = 10;
        config.newton_max_iter = 100;
        config.newton_tol = 1e-6;
        config.newton_regularization = 1e-5;
        config.use_constant_state_init = true;
    }
    return config;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: parity_dump <task> <multiple|single> <horizon> <outdir>\n";
        return 1;
    }
    const std::string task = argv[1];
    const std::string mode = argv[2];
    const int horizon = std::atoi(argv[3]);
    const std::string outdir = argv[4];

    impact::AulaConfig config = configFor(task, horizon);
    for (int i = 5; i < argc; ++i) {
        std::cerr << "unknown flag " << argv[i] << "\n";
        return 1;
    }

    std::error_code ec_early;
    std::filesystem::create_directories(outdir, ec_early);

    std::shared_ptr<impact::MPCCProblem> problem;
    std::unique_ptr<impact::StageProblem> stage;
    if (task == "push_circle") {
        problem = std::make_shared<push_circle::PushCircle>();
        stage = std::make_unique<impact::MPCCStage>(problem, config);
    } else if (task == "box") {
        problem = std::make_shared<box_pushing::BoxPushing>();
        stage = std::make_unique<impact::MPCCStage>(problem, config);
    } else if (task == "push_t") {
        problem = std::make_shared<push_t::PushT>();
        stage = std::make_unique<impact::MPCCStage>(problem, config);
    } else if (task == "cart_transporter") {
        problem = std::make_shared<cart_transporter::CartTransporter>();
        stage = std::make_unique<impact::MPCCStage>(problem, config);
    } else {
        std::cerr << "unknown task " << task << "\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);

    const impact::AulaSubproblem* sub = nullptr;
    int off_p = 0, off_x0 = 0;
    impact::MultipleShootingLayout ms;
    impact::SingleShootingLayout ss;
    if (mode == "multiple") {
        ms = impact::buildMultipleShooting(*stage, config);
        sub = ms.sub.get();
        off_p = ms.off_p;
        off_x0 = ms.off_x0;
    } else if (mode == "single") {
        ss = impact::buildSingleShooting(*stage, config);
        sub = ss.sub.get();
        off_p = ss.off_p;
        off_x0 = ss.off_x0;
    } else {
        std::cerr << "unknown mode " << mode << "\n";
        return 1;
    }

    sub->residualFunction().save(outdir + "/residual.casadi");
    sub->jacobianFunction().save(outdir + "/jacobian.casadi");
    // n_params is not exposed directly; the parameter buffer's length is it.
    const int n_params = static_cast<int>(sub->params().numel());
    writeJson(outdir + "/meta.json", *sub, sub->numOpt(), n_params, off_p, off_x0);

    std::cout << "wrote " << outdir << " (n_opt=" << sub->numOpt() << ", n_params=" << n_params
              << ")" << std::endl;
    return 0;
}
