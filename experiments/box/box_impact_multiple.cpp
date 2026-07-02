#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "impact/multiple_shooting.h"
#include "box_pushing.h"

bool parseArgs(int argc, char* argv[], Eigen::VectorXd& x0, Eigen::VectorXd& x_goal,
               std::string& output_file) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " x0_px x0_py x0_theta goal_px goal_py goal_theta [output_file]\n";
        return false;
    }
    x0 = Eigen::VectorXd::Zero(3);
    x_goal = Eigen::VectorXd::Zero(3);
    for (int i = 0; i < 3; ++i) x0(i) = std::atof(argv[1 + i]);
    for (int i = 0; i < 3; ++i) x_goal(i) = std::atof(argv[4 + i]);

    if (argc >= 8) {
        output_file = argv[7];
    } else {
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        std::ostringstream oss;
        oss << "results/box/bcd_aula/trajectory_" << ts << ".txt";
        output_file = oss.str();
    }
    return true;
}

void saveSolutionToFile(const impact::MultipleShootingSolution& solution, const Eigen::VectorXd& x0,
                        const Eigen::VectorXd& x_goal, int iterations, double solve_time,
                        const std::string& filename) {
    std::filesystem::path out_path(filename);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << std::fixed << std::setprecision(10);
    file << "# Box Pushing BCD-AULA Trajectory\n# Planner: bcd_aula\n# Task: box\n\n";
    file << "# Start State (px, py, theta)\n";
    for (int i = 0; i < x0.rows(); ++i) file << x0(i) << (i < x0.rows() - 1 ? " " : "");
    file << "\n\n# Goal State (px, py, theta)\n";
    for (int i = 0; i < x_goal.rows(); ++i) file << x_goal(i) << (i < x_goal.rows() - 1 ? " " : "");
    file << "\n\n# Iterations\n" << iterations << "\n\n# Solve Time (seconds)\n" << solve_time
         << "\n\n# Success\n" << (solution.converged ? 1 : 0) << "\n\n";
    file << "# State Trajectory (rows: timesteps, cols: px, py, theta)\n";
    for (int k = 0; k < solution.state_trajectory.cols(); ++k) {
        for (int i = 0; i < solution.state_trajectory.rows(); ++i)
            file << solution.state_trajectory(i, k)
                 << (i < solution.state_trajectory.rows() - 1 ? " " : "");
        file << "\n";
    }
    file << "\n# Control Trajectory (rows: timesteps, cols: cx, cy, ld1y, ld2x, ld3y, ld4x)\n";
    for (int k = 0; k < solution.control_trajectory.cols(); ++k) {
        for (int i = 0; i < solution.control_trajectory.rows(); ++i)
            file << solution.control_trajectory(i, k)
                 << (i < solution.control_trajectory.rows() - 1 ? " " : "");
        file << "\n";
    }
    file.close();
    std::cout << "Solution saved to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Box Pushing MPCC with BCD-AuLa (IMPACT) Solver ===" << std::endl;

    Eigen::VectorXd x0, x_goal;
    std::string output_file;
    if (!parseArgs(argc, argv, x0, x_goal, output_file)) return 1;

    auto box_problem = std::make_shared<box_pushing::BoxPushing>();

    impact::BCDAULAConfig config;
    config.horizon = 50;
    config.x_0 = x0;
    config.x_goal = x_goal;

    config.stage_cost_weight = 0.001;
    config.final_cost_weight = 100.0;

    config.rho_max = 100.0;
    config.rho_scale = 1.1;

    const double all_scale = 25.0;
    config.fix_point_scale = all_scale;
    config.dynamics_scale = all_scale;
    config.eq_scale = all_scale;
    config.ineq_scale = all_scale;
    config.comp_scale = 0.1;

    config.max_outer_iters = 500;
    config.outer_tol_h = 1e-5;
    config.outer_tol_comp = 1e-5;

    config.max_inner_iters = 50;
    config.inner_tol_init = 1e-2;
    config.inner_tol_final = 1e-3;

    config.newton_max_iter = 50;
    config.newton_tol = 1e-6;
    config.newton_regularization = 2e-5;
    config.use_saddle = true;

    config.print_level = 1;

    impact::MultipleShootingSolver solver(box_problem);
    impact::MultipleShootingSolution solution = solver.solve(config);

    std::cout << "\n=== Solution Summary ===" << std::endl;
    std::cout << "Converged: " << (solution.converged ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << solution.objective_value << std::endl;
    std::cout << "Outer iterations: " << solution.outer_iterations << std::endl;
    std::cout << "Total inner iterations: " << solution.total_inner_iterations << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " seconds" << std::endl;
    std::cout << "Final state: [" << solution.state_trajectory.col(config.horizon).transpose()
              << "]" << std::endl;
    std::cout << "Goal state:  [" << config.x_goal.transpose() << "]" << std::endl;
    std::cout << "Complementarity violation: " << solution.complementarity_violation << std::endl;

    saveSolutionToFile(solution, config.x_0, config.x_goal, solution.total_inner_iterations,
                       solution.solve_time, output_file);
    std::cout << "\n=== Optimization Complete ===" << std::endl;
    return 0;
}
