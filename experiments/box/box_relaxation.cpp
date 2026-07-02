#include <sys/stat.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "box_pushing.h"
#include "relaxation_solver/mpcc_relaxation_solver.h"

/**
 * @brief Parse command line arguments for initial state, goal state, and output file
 */
bool parseArgs(int argc, char* argv[], casadi::DM& x0, casadi::DM& x_goal,
               std::string& output_file) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " x0_px x0_py x0_theta goal_px goal_py goal_theta [output_file]\n";
        std::cerr << "  x0: initial state (px, py, theta)\n";
        std::cerr << "  goal: goal state (px, py, theta)\n";
        std::cerr << "  output_file: optional output file path\n";
        return false;
    }

    x0 = casadi::DM::zeros(3, 1);
    x_goal = casadi::DM::zeros(3, 1);

    x0(0) = std::atof(argv[1]);
    x0(1) = std::atof(argv[2]);
    x0(2) = std::atof(argv[3]);
    x_goal(0) = std::atof(argv[4]);
    x_goal(1) = std::atof(argv[5]);
    x_goal(2) = std::atof(argv[6]);

    if (argc >= 8) {
        output_file = argv[7];
    } else {
        auto now = std::chrono::system_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << "results/box/relaxation/trajectory_" << timestamp << ".txt";
        output_file = oss.str();
    }

    return true;
}

/**
 * @brief Save Scholtes relaxation solution to file for visualization
 */
void saveSolutionToFile(const aula::ScholtesRelaxationSolution& solution, const casadi::DM& x0,
                        const casadi::DM& x_goal, int iterations, double solve_time,
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

    // Write header with experiment metadata
    file << "# Box Pushing Scholtes Relaxation Trajectory\n";
    file << "# Planner: relaxation\n";
    file << "# Task: box\n\n";

    // Write start state
    file << "# Start State (px, py, theta)\n";
    for (casadi_int i = 0; i < x0.rows(); ++i) {
        file << static_cast<double>(x0(i));
        if (i < x0.rows() - 1) file << " ";
    }
    file << "\n\n";

    // Write goal state
    file << "# Goal State (px, py, theta)\n";
    for (casadi_int i = 0; i < x_goal.rows(); ++i) {
        file << static_cast<double>(x_goal(i));
        if (i < x_goal.rows() - 1) file << " ";
    }
    file << "\n\n";

    // Write iterations (total IPOPT iterations for relaxation)
    file << "# Iterations\n" << iterations << "\n\n";
    file << "# Solve Time (seconds)\n" << solve_time << "\n\n";

    // Write success status
    file << "# Success\n" << (solution.converged ? 1 : 0) << "\n\n";

    // Write state trajectory
    file << "# State Trajectory (rows: timesteps, cols: px, py, theta)\n";
    for (casadi_int k = 0; k < solution.state_trajectory.size2(); ++k) {
        for (casadi_int i = 0; i < solution.state_trajectory.size1(); ++i) {
            file << static_cast<double>(solution.state_trajectory(i, k));
            if (i < solution.state_trajectory.size1() - 1) file << " ";
        }
        file << "\n";
    }

    file << "\n# Control Trajectory (rows: timesteps, cols: cx, cy, ld1y, ld2x, ld3y, ld4x)\n";
    for (casadi_int k = 0; k < solution.control_trajectory.size2(); ++k) {
        for (casadi_int i = 0; i < solution.control_trajectory.size1(); ++i) {
            file << static_cast<double>(solution.control_trajectory(i, k));
            if (i < solution.control_trajectory.size1() - 1) file << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Solution saved to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Box Pushing MPCC with Scholtes Relaxation Solver ===" << std::endl;

    // Parse command line arguments
    casadi::DM x0, x_goal;
    std::string output_file;
    if (!parseArgs(argc, argv, x0, x_goal, output_file)) {
        return 1;
    }

    // Create box pushing problem
    auto box_problem = std::make_shared<box_pushing::BoxPushing>();

    std::cout << "\nProblem dimensions:" << std::endl;
    std::cout << "  State dimension: " << box_problem->getStateDim() << std::endl;
    std::cout << "  Control dimension: " << box_problem->getControlDim() << std::endl;
    std::cout << "  Complementarity constraints: " << box_problem->getComplementarityDim()
              << std::endl;
    std::cout << "  Time step: " << box_problem->getTimeStep() << " s" << std::endl;

    // Configure Scholtes relaxation solver
    aula::ScholtesRelaxationConfig config;
    config.horizon = 50;

    // Use parsed initial and goal states
    config.x_0 = x0;
    config.x_goal = x_goal;

    // Cost weights
    config.stage_cost_weight = 0.001;
    config.final_cost_weight = 100.0;

    // Scholtes relaxation parameters
    config.t_init = 1.0;    // Initial relaxation (G*H <= t)
    config.t_final = 1e-8;  // Final relaxation parameter
    config.gamma = 0.1;     // Reduction factor per iteration
    config.max_outer_iters = 30;
    config.comp_tol = 1e-5;  // Stop when max|G*H| < comp_tol

    // IPOPT options
    config.ipopt_print_level = 0;  // Quiet IPOPT, we print our own summary
    config.ipopt_max_iter = 500;
    config.ipopt_tol = 1e-5;
    config.warm_start = true;

    // Verbosity
    config.print_level = 1;  // Summary output
    config.print_time = false;

    std::cout << "\nOptimization configuration:" << std::endl;
    std::cout << "  Horizon: " << config.horizon << " steps" << std::endl;
    std::cout << "  Initial state: [" << config.x_0(0) << ", " << config.x_0(1) << ", "
              << config.x_0(2) << "]" << std::endl;
    std::cout << "  Goal state: [" << config.x_goal(0) << ", " << config.x_goal(1) << ", "
              << config.x_goal(2) << "]" << std::endl;
    std::cout << "  Stage cost weight: " << config.stage_cost_weight << std::endl;
    std::cout << "  Final cost weight: " << config.final_cost_weight << std::endl;
    std::cout << "  t_init: " << config.t_init << std::endl;
    std::cout << "  t_final: " << config.t_final << std::endl;
    std::cout << "  gamma: " << config.gamma << std::endl;
    std::cout << "  comp_tol: " << config.comp_tol << std::endl;

    // Create solver and solve
    std::cout << "\n=== Solving MPCC with Scholtes Relaxation ===" << std::endl;
    aula::MPCCScholtesRelaxationSolver solver(box_problem);

    aula::ScholtesRelaxationSolution solution = solver.solve(config);

    // Print results
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Solution Summary ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Converged: " << (solution.converged ? "YES" : "NO") << std::endl;
    std::cout << "Status: " << solution.status_message << std::endl;
    std::cout << "Objective value: " << solution.objective_value << std::endl;
    std::cout << "Outer iterations: " << solution.outer_iterations << std::endl;
    std::cout << "Total IPOPT iterations: " << solution.total_ipopt_iterations << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " seconds" << std::endl;
    std::cout << "Max complementarity: " << solution.max_complementarity << std::endl;
    std::cout << "Final relaxation t: " << solution.final_relaxation_t << std::endl;

    std::cout << "\nFinal state: ["
              << static_cast<double>(solution.state_trajectory(0, config.horizon)) << ", "
              << static_cast<double>(solution.state_trajectory(1, config.horizon)) << ", "
              << static_cast<double>(solution.state_trajectory(2, config.horizon)) << "]"
              << std::endl;
    std::cout << "Goal state:  [" << config.x_goal(0) << ", " << config.x_goal(1) << ", "
              << config.x_goal(2) << "]" << std::endl;

    // Save solution to file with all metrics
    // For relaxation, iterations = total_ipopt_iterations (cumulative inner loop iterations)
    saveSolutionToFile(solution, config.x_0, config.x_goal, solution.total_ipopt_iterations,
                       solution.solve_time, output_file);

    std::cout << "\n=== Optimization Complete ===" << std::endl;

    return 0;
}
