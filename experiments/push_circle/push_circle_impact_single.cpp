/**
 * @file push_circle_impact_single.cpp
 * @brief Disk-pushing single-shooting driver.
 *
 * Uses the same scenario, solver settings, and output format as
 * push_circle_impact_multiple.cpp; only the shooting transcription changes.
 */

#include <casadi/casadi.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

#include "impact/bcd_aula_solver.h"
#include "impact/mpcc_stage.h"
#include "impact/single_shooting.h"
#include "push_circle.h"

struct Scenario {
    double D = 1.5;
    double angle_deg = 225;
    int horizon = 50; // too long horizon for mul-shoot is quite slow
    std::string output_file;
};

int main(int argc, char* argv[]) {
    std::cout << "=== Disk Pushing MPCC with BCD-AuLa (IMPACT) — SINGLE SHOOTING ==="
              << std::endl;

    Scenario sc;
    if (argc >= 2) sc.D = std::atof(argv[1]);
    if (argc >= 3) sc.angle_deg = std::atof(argv[2]);
    if (argc >= 4) sc.horizon = std::atoi(argv[3]);
    if (argc >= 5) {
        sc.output_file = argv[4];
    } else {
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        std::ostringstream oss;
        oss << "results/push_circle/bcd_aula/trajectory_single_" << ts << ".txt";
        sc.output_file = oss.str();
    }

    auto problem = std::make_shared<push_circle::PushCircle>();
    const double R = problem->getRadius();
    const double ang = sc.angle_deg * M_PI / 180.0;
    const double tx = sc.D * std::cos(ang);
    const double ty = sc.D * std::sin(ang);

    double goal_x = tx;
    double goal_y = ty;
    if (argc >= 7) {
        goal_x = std::atof(argv[5]);
        goal_y = std::atof(argv[6]);
    }

    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.0, tx, ty;

    const double goal_distance = std::hypot(goal_x, goal_y);
    const double goal_dir_x = goal_distance > std::numeric_limits<double>::epsilon()
                                  ? goal_x / goal_distance
                                  : std::cos(ang);
    const double goal_dir_y = goal_distance > std::numeric_limits<double>::epsilon()
                                  ? goal_y / goal_distance
                                  : std::sin(ang);
    Eigen::VectorXd x_goal(4);
    x_goal << goal_x, goal_y, goal_x - R * goal_dir_x, goal_y - R * goal_dir_y;

    std::cout << "Disk radius R = " << R << ", push distance D = " << sc.D
              << ", direction = " << sc.angle_deg << " deg, horizon = " << sc.horizon << "\n";
    std::cout << "x0    = [" << x0.transpose() << "]\n";
    std::cout << "goal  = [" << x_goal.transpose() << "]\n";

    impact::BCDAULAConfig config;
    config.horizon = sc.horizon;
    config.x_0 = x0;
    config.x_goal = x_goal;
    config.use_constant_state_init = true;
    config.stage_cost_weight = 1e-2;
    config.final_cost_weight = 100.0;
    config.rho_max = 200.0;
    config.rho_scale = 1.05;
    const double all_scale = 10.0;
    config.fix_point_scale = all_scale;
    config.dynamics_scale = all_scale;
    config.eq_scale = all_scale;
    config.ineq_scale = all_scale;
    config.comp_scale = 1.0;
    config.max_outer_iters = 800;
    config.outer_tol_h = 1e-5;
    config.outer_tol_g = 1e-5;
    config.outer_tol_comp = 1e-5;
    config.max_inner_iters = 50;
    config.inner_tol_init = 1e-2;
    config.inner_tol_final = 1e-3;
    config.newton_max_iter = 100;
    config.newton_tol = 1e-5;
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
    impact::BCDAULAResult solution = solver.solve(sub, config, z);

    const int nx = problem->getStateDim();
    const int nu = problem->getControlDim();
    const double dt = problem->getTimeStep();
    const casadi::Function dyn = problem->getDynamicsFunction();
    Eigen::MatrixXd state_trajectory(nx, sc.horizon + 1);
    Eigen::MatrixXd control_trajectory(nu, sc.horizon);
    state_trajectory.col(0) = x0;
    for (int k = 0; k < sc.horizon; ++k) {
        const Eigen::VectorXd u = solution.z.segment(k * nu, nu);
        control_trajectory.col(k) = u;
        const Eigen::VectorXd x = state_trajectory.col(k);
        const casadi::DM xdm(std::vector<double>(x.data(), x.data() + nx));
        const casadi::DM udm(std::vector<double>(u.data(), u.data() + nu));
        const casadi::DM f = dyn(std::vector<casadi::DM>{xdm, udm})[0];
        state_trajectory.col(k + 1) = x + dt * Eigen::Map<const Eigen::VectorXd>(f.ptr(), nx);
    }

    std::cout << "\n=== Solution Summary ===" << std::endl;
    std::cout << "Converged: " << (solution.converged ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << solution.objective_value << std::endl;
    std::cout << "Outer iterations: " << solution.outer_iterations << std::endl;
    std::cout << "Total inner iterations: " << solution.total_inner_iterations << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " seconds" << std::endl;
    std::cout << "Variables (single): " << sub.numOpt() << "  (multiple has "
              << nx * (sc.horizon + 1) + nu * sc.horizon << ")" << std::endl;
    std::cout << "Final disk:  [" << state_trajectory.col(sc.horizon).head(2).transpose()
              << "]" << std::endl;
    std::cout << "Goal disk:   [" << x_goal.head(2).transpose() << "]" << std::endl;
    std::cout << "Complementarity violation: " << solution.complementarity_violation << std::endl;

    const std::filesystem::path out_path(sc.output_file);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }
    std::ofstream file(sc.output_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << sc.output_file << std::endl;
        return 1;
    }
    file << std::fixed << std::setprecision(10);
    file << "# Disk Pushing BCD-AULA Trajectory\n# Planner: bcd_aula_single\n# Task: push_circle\n\n";
    file << "# Disk Radius\n" << R << "\n\n";
    file << "# Start State (qx, qy, sx, sy)\n";
    for (int i = 0; i < x0.rows(); ++i) file << x0(i) << (i < x0.rows() - 1 ? " " : "");
    file << "\n\n# Goal State (qx, qy, sx, sy)\n";
    for (int i = 0; i < x_goal.rows(); ++i)
        file << x_goal(i) << (i < x_goal.rows() - 1 ? " " : "");
    file << "\n\n# Iterations\n" << solution.total_inner_iterations
         << "\n\n# Solve Time (seconds)\n" << solution.solve_time << "\n\n# Success\n"
         << (solution.converged ? 1 : 0) << "\n\n";
    file << "# State Trajectory (rows: timesteps, cols: qx, qy, sx, sy)\n";
    for (int k = 0; k < state_trajectory.cols(); ++k) {
        for (int i = 0; i < state_trajectory.rows(); ++i)
            file << state_trajectory(i, k) << (i < state_trajectory.rows() - 1 ? " " : "");
        file << "\n";
    }
    file << "\n# Control Trajectory (rows: timesteps, cols: fn, ft, vx, vy)\n";
    for (int k = 0; k < control_trajectory.cols(); ++k) {
        for (int i = 0; i < control_trajectory.rows(); ++i)
            file << control_trajectory(i, k)
                 << (i < control_trajectory.rows() - 1 ? " " : "");
        file << "\n";
    }
    file.close();
    std::cout << "Solution saved to: " << sc.output_file << std::endl;
    std::cout << "\n=== Optimization Complete ===" << std::endl;
    return 0;
}
