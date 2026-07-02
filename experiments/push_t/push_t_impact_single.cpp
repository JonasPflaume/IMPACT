/**
 * @file push_t_impact_single.cpp
 * @brief Push-T single-shooting driver.
 *
 * Same problem and BCD-AuLa settings as the multiple-shooting driver. The only
 * difference is the transcription:
 *   - multiple shooting: z = [vec(X); vec(U)], state free, dynamics as a defect
 *     equality block  -> SPARSE BANDED Jacobian, linear (constant) cost block.
 *   - single shooting (here): z = [vec(U)], state rolled out from x_0 via step()
 *     -> no dynamics/fix-point blocks, but x_k depends on u_0..u_{k-1}, so the
 *     Jacobian is DENSE lower-triangular and the cost is nonlinear (rho=1 block).
 *
 * Usage: push_t_single_shooting x0_px x0_py x0_theta goal_px goal_py goal_theta
 */

#include <casadi/casadi.hpp>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "impact/bcd_aula_solver.h"
#include "impact/mpcc_stage.h"
#include "impact/single_shooting.h"
#include "push_t.h"

int main(int argc, char* argv[]) {
    std::cout << "=== Push-T MPCC with BCD-AuLa (IMPACT) — SINGLE SHOOTING ===" << std::endl;
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " x0_px x0_py x0_theta goal_px goal_py goal_theta\n";
        return 1;
    }
    Eigen::VectorXd x0(3), x_goal(3);
    for (int i = 0; i < 3; ++i) {
        x0(i) = std::atof(argv[1 + i]);
        x_goal(i) = std::atof(argv[4 + i]);
    }

    auto problem = std::make_shared<push_t::PushT>();

    // Same settings as the multiple-shooting driver; only the transcription differs.
    impact::BCDAULAConfig config;
    config.horizon = 50;
    config.x_0 = x0;
    config.x_goal = x_goal;
    config.stage_cost_weight = 0.01;
    config.stage_state_cost_weight = 0.0;
    config.control_rate_weight = 0.0;
    config.final_cost_weight = 100.0;
    config.rho_max = 500.0;
    config.rho_scale = 1.05;
    const double all_scale = 25.0;
    config.fix_point_scale = all_scale;
    config.dynamics_scale = all_scale;
    config.eq_scale = all_scale;
    config.ineq_scale = all_scale;
    config.comp_scale = 1.0;
    config.max_outer_iters = 1000;
    config.outer_tol_h = 1e-5;
    config.outer_tol_comp = 1e-5;
    config.outer_tol_g = 1e-5;
    config.max_inner_iters = 50;
    config.inner_tol_init = 5e-3;
    config.inner_tol_final = 1e-3;
    config.newton_max_iter = 200;
    config.newton_tol = 1e-6;
    config.newton_regularization = 5e-5;
    config.print_level = 1;
    config.use_saddle = true;

    // Build the single-shooting subproblem from the same MPCCStage.
    impact::MPCCStage stage(problem, config);
    impact::SingleShootingLayout layout = impact::buildSingleShooting(stage, config);
    impact::AulaSubproblem& sub = *layout.sub;
    sub.setParamValue(layout.off_p, x_goal);  // runtime param = x_goal
    sub.setParamValue(layout.off_x0, x0);     // rollout start

    Eigen::VectorXd z = Eigen::VectorXd::Zero(sub.numOpt());  // U = 0 warm start

    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(sub, config, z);

    // Roll out the final state from the optimized controls.
    const int nu = problem->getControlDim();
    const double dt = problem->getTimeStep();
    casadi::Function dyn = problem->getDynamicsFunction();
    Eigen::VectorXd x = x0;
    for (int k = 0; k < config.horizon; ++k) {
        Eigen::VectorXd u = r.z.segment(k * nu, nu);
        casadi::DM xdm(std::vector<double>(x.data(), x.data() + 3));
        casadi::DM udm(std::vector<double>(u.data(), u.data() + nu));
        casadi::DM f = dyn(std::vector<casadi::DM>{xdm, udm})[0];
        x += dt * Eigen::Map<const Eigen::VectorXd>(f.ptr(), 3);
    }

    std::cout << "\n=== Single-Shooting Solution Summary ===" << std::endl;
    std::cout << "Converged: " << (r.converged ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << r.objective_value << std::endl;
    std::cout << "Outer iterations: " << r.outer_iterations << std::endl;
    std::cout << "Total inner iterations: " << r.total_inner_iterations << std::endl;
    std::cout << "Solve time: " << r.solve_time << " seconds" << std::endl;
    std::cout << "Variables (single): " << sub.numOpt() << "  (multiple has "
              << 3 * (config.horizon + 1) + nu * config.horizon << ")" << std::endl;
    std::cout << "Final state: [" << x.transpose() << "]" << std::endl;
    std::cout << "Goal state:  [" << x_goal.transpose() << "]" << std::endl;
    std::cout << "Complementarity violation: " << r.complementarity_violation << std::endl;
    return 0;
}
