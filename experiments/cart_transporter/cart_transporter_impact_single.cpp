/**
 * @file cart_transporter_impact_single.cpp
 * @brief Cart-transporter single-shooting driver.
 *
 * Uses the same task and solver settings as cart_transporter_impact_multiple.cpp.
 *
 * NOTE: horizon = 300 with a 4-dim state that includes velocity. This is the
 * long-horizon / sensitive-dynamics regime where single shooting is expected to
 * struggle: the rolled-out Jacobian is dense and ∂x_T/∂u_0 grows quickly.
 *
 * Usage: cart_transporter_impact_single x1 x2 x1d x2d  gx1 gx2 gx1d gx2d
 */

#include <casadi/casadi.hpp>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "impact/bcd_aula_solver.h"
#include "impact/mpcc_stage.h"
#include "impact/single_shooting.h"
#include "cart_transporter.h"

int main(int argc, char* argv[]) {
    std::cout << "=== Cart Transporter MPCC with BCD-AuLa (IMPACT) — SINGLE SHOOTING ===" << std::endl;
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] << " x1 x2 x1d x2d  gx1 gx2 gx1d gx2d\n";
        return 1;
    }
    Eigen::VectorXd x0(4), x_goal(4);
    for (int i = 0; i < 4; ++i) {
        x0(i) = std::atof(argv[1 + i]);
        x_goal(i) = std::atof(argv[5 + i]);
    }

    auto problem = std::make_shared<cart_transporter::CartTransporter>();

    // Same settings as the multiple-shooting driver.
    impact::BCDAULAConfig config;
    config.horizon = 300;
    config.x_0 = x0;
    config.x_goal = x_goal;
    config.stage_cost_weight = 1e-6;
    config.stage_state_cost_weight = 0.0;
    config.final_cost_weight = 5000.0;
    config.rho_max = 100000.0;
    config.rho_scale = 1.5;
    config.fix_point_scale = 1.0;
    config.dynamics_scale = 1.0;
    config.eq_scale = 1.0;
    config.ineq_scale = 1.0;
    config.comp_scale = 0.002;
    config.max_outer_iters = 1000;
    config.outer_tol_h = 1e-5;
    config.outer_tol_comp = 1e-5;
    config.outer_tol_g = 1e-5;
    config.max_inner_iters = 10;
    config.inner_tol_init = 1e-2;
    config.inner_tol_final = 1e-3;
    config.newton_max_iter = 100;
    config.newton_tol = 1e-6;
    config.newton_regularization = 1e-5;
    config.use_saddle = true;
    config.print_level = 1;

    impact::MPCCStage stage(problem, config);
    impact::SingleShootingLayout layout = impact::buildSingleShooting(stage, config);
    impact::AulaSubproblem& sub = *layout.sub;
    sub.setParamValue(layout.off_p, x_goal);
    sub.setParamValue(layout.off_x0, x0);

    Eigen::VectorXd z = Eigen::VectorXd::Zero(sub.numOpt());

    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(sub, config, z);

    const int nx = problem->getStateDim();
    const int nu = problem->getControlDim();
    const double dt = problem->getTimeStep();
    casadi::Function dyn = problem->getDynamicsFunction();
    Eigen::VectorXd x = x0;
    for (int k = 0; k < config.horizon; ++k) {
        Eigen::VectorXd u = r.z.segment(k * nu, nu);
        casadi::DM xdm(std::vector<double>(x.data(), x.data() + nx));
        casadi::DM udm(std::vector<double>(u.data(), u.data() + nu));
        casadi::DM f = dyn(std::vector<casadi::DM>{xdm, udm})[0];
        x += dt * Eigen::Map<const Eigen::VectorXd>(f.ptr(), nx);
    }

    std::cout << "\n=== Single-Shooting Solution Summary ===" << std::endl;
    std::cout << "Converged: " << (r.converged ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << r.objective_value << std::endl;
    std::cout << "Outer iterations: " << r.outer_iterations << std::endl;
    std::cout << "Total inner iterations: " << r.total_inner_iterations << std::endl;
    std::cout << "Solve time: " << r.solve_time << " seconds" << std::endl;
    std::cout << "Variables (single): " << sub.numOpt() << "  (multiple has "
              << nx * (config.horizon + 1) + nu * config.horizon << ")" << std::endl;
    std::cout << "Final state: [" << x.transpose() << "]" << std::endl;
    std::cout << "Goal state:  [" << x_goal.transpose() << "]" << std::endl;
    std::cout << "Complementarity violation: " << r.complementarity_violation << std::endl;
    return 0;
}
