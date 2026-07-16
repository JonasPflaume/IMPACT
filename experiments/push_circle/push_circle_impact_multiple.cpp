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

#include "impact/multiple_shooting.h"
#include "push_circle.h"

// Scenario: a disk starts at the origin; the pusher starts far to the lower-left
// (at distance D in direction `angle_deg`); by default the disk's goal is the
// pusher's own initial position, but optional command-line coordinates can place
// it elsewhere.
// For the default goal the pusher must travel around to the upper-right and push
// the disk back down. With the trajectory initialized by replicating the start
// (use_constant_state_init), this is a hard local-minimum-escape test: pushing
// from where the pusher starts moves the disk the wrong way.
struct Scenario {
    double D = 1.5;          // pusher/goal distance from the disk center
    double angle_deg = 225;  // lower-left
    int horizon = 100;
    std::string output_file;
};

int main(int argc, char* argv[]) {
    std::cout << "=== Disk Pushing MPCC with BCD-AuLa (IMPACT) — MULTIPLE SHOOTING ===" << std::endl;

    Scenario sc;
    if (argc >= 2) sc.D = std::atof(argv[1]);
    if (argc >= 3) sc.angle_deg = std::atof(argv[2]);
    if (argc >= 4) sc.horizon = std::atoi(argv[3]);
    if (argc >= 5) {
        sc.output_file = argv[4];
    } else {
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        std::ostringstream oss;
        oss << "results/push_circle/bcd_aula/trajectory_" << ts << ".txt";
        sc.output_file = oss.str();
    }

    auto problem = std::make_shared<push_circle::PushCircle>();
    const double R = problem->getRadius();

    // Disk at the origin; the pusher starts at distance D in direction angle_deg.
    // The goal circle defaults to the same position. Supplying both goal_x and
    // goal_y after the output-file argument overrides that default.
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
    x0 << 0.0, 0.0, tx, ty;  // [qx, qy, sx, sy]

    // The pusher's terminal target is the contact point behind the disk along the
    // direction of travel. This keeps the final knot non-penetrating (the terminal
    // state carries no per-stage contact constraint). If the disk goal is its start,
    // retain the scenario direction to define a sensible contact point.
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

    // The key knob for the local-minimum test: initialize every knot at the start
    // instead of interpolating toward the goal.
    config.use_constant_state_init = false;

    config.stage_cost_weight = 1e-2;    // ||[f_n, f_t, v]||^2
    config.final_cost_weight = 100.0;

    config.rho_max = 1000.0;
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

    impact::MultipleShootingSolver solver(problem);
    impact::MultipleShootingSolution solution = solver.solve(config);

    std::cout << "\n=== Solution Summary ===" << std::endl;
    std::cout << "Converged: " << (solution.converged ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << solution.objective_value << std::endl;
    std::cout << "Outer iterations: " << solution.outer_iterations << std::endl;
    std::cout << "Total inner iterations: " << solution.total_inner_iterations << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " seconds" << std::endl;
    std::cout << "Final disk:  [" << solution.state_trajectory.col(sc.horizon).head(2).transpose()
              << "]" << std::endl;
    std::cout << "Goal disk:   [" << x_goal.head(2).transpose() << "]" << std::endl;
    std::cout << "Complementarity violation: " << solution.complementarity_violation << std::endl;

    // Write the trajectory.
    std::filesystem::path out_path(sc.output_file);
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
    file << "# Disk Pushing BCD-AULA Trajectory\n# Planner: bcd_aula\n# Task: push_circle\n\n";
    file << "# Disk Radius\n" << R << "\n\n";
    file << "# Start State (qx, qy, sx, sy)\n";
    for (int i = 0; i < x0.rows(); ++i) file << x0(i) << (i < x0.rows() - 1 ? " " : "");
    file << "\n\n# Goal State (qx, qy, sx, sy)\n";
    for (int i = 0; i < x_goal.rows(); ++i) file << x_goal(i) << (i < x_goal.rows() - 1 ? " " : "");
    file << "\n\n# Iterations\n" << solution.total_inner_iterations << "\n\n# Solve Time (seconds)\n"
         << solution.solve_time << "\n\n# Success\n" << (solution.converged ? 1 : 0) << "\n\n";
    file << "# State Trajectory (rows: timesteps, cols: qx, qy, sx, sy)\n";
    for (int k = 0; k < solution.state_trajectory.cols(); ++k) {
        for (int i = 0; i < solution.state_trajectory.rows(); ++i)
            file << solution.state_trajectory(i, k)
                 << (i < solution.state_trajectory.rows() - 1 ? " " : "");
        file << "\n";
    }
    file << "\n# Control Trajectory (rows: timesteps, cols: fn, ft, vx, vy)\n";
    for (int k = 0; k < solution.control_trajectory.cols(); ++k) {
        for (int i = 0; i < solution.control_trajectory.rows(); ++i)
            file << solution.control_trajectory(i, k)
                 << (i < solution.control_trajectory.rows() - 1 ? " " : "");
        file << "\n";
    }
    file.close();
    std::cout << "Solution saved to: " << sc.output_file << std::endl;
    std::cout << "\n=== Optimization Complete ===" << std::endl;
    return 0;
}
