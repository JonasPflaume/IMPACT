#include <sys/stat.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "cart_transporter.h"
#include "relaxation_solver/mpcc_relaxation_solver.h"

/**
 * @brief Parse command line arguments for initial state, goal state, and output file
 * Cart transporter has 4D state: x1, x2, x1_dot, x2_dot
 */
bool parseArgs(int argc, char* argv[], casadi::DM& x0, casadi::DM& x_goal,
               std::string& output_file) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " x0_x1 x0_x2 x0_x1dot x0_x2dot goal_x1 goal_x2 goal_x1dot goal_x2dot "
                     "[output_file]\n";
        std::cerr << "  x0: initial state (x1, x2, x1_dot, x2_dot)\n";
        std::cerr << "  goal: goal state (x1, x2, x1_dot, x2_dot)\n";
        std::cerr << "  output_file: optional output file path\n";
        return false;
    }

    x0 = casadi::DM::zeros(4, 1);
    x_goal = casadi::DM::zeros(4, 1);

    x0(0) = std::atof(argv[1]);
    x0(1) = std::atof(argv[2]);
    x0(2) = std::atof(argv[3]);
    x0(3) = std::atof(argv[4]);
    x_goal(0) = std::atof(argv[5]);
    x_goal(1) = std::atof(argv[6]);
    x_goal(2) = std::atof(argv[7]);
    x_goal(3) = std::atof(argv[8]);

    if (argc >= 10) {
        output_file = argv[9];
    } else {
        auto now = std::chrono::system_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << "results/cart_transporter/relaxation/trajectory_" << timestamp << ".txt";
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
    file << "# Cart Transporter Scholtes Relaxation Trajectory\n";
    file << "# Planner: relaxation\n";
    file << "# Task: cart_transporter\n\n";

    // Write start state
    file << "# Start State (x1, x2, x1_dot, x2_dot)\n";
    for (casadi_int i = 0; i < x0.rows(); ++i) {
        file << static_cast<double>(x0(i));
        if (i < x0.rows() - 1) file << " ";
    }
    file << "\n\n";

    // Write goal state
    file << "# Goal State (x1, x2, x1_dot, x2_dot)\n";
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
    file << "# State Trajectory (rows: timesteps, cols: x1, x2, x1_dot, x2_dot)\n";
    for (casadi_int k = 0; k < solution.state_trajectory.size2(); ++k) {
        for (casadi_int i = 0; i < solution.state_trajectory.size1(); ++i) {
            file << static_cast<double>(solution.state_trajectory(i, k));
            if (i < solution.state_trajectory.size1() - 1) file << " ";
        }
        file << "\n";
    }

    file << "\n# Control Trajectory (rows: timesteps, cols: f, u, v, w)\n";
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
    std::cout << "=== Cart Transporter MPCC with Scholtes Relaxation Solver ===" << std::endl;

    // Parse command line arguments
    casadi::DM x0, x_goal;
    std::string output_file;
    if (!parseArgs(argc, argv, x0, x_goal, output_file)) {
        return 1;
    }

    // Create cart transporter problem
    auto problem = std::make_shared<cart_transporter::CartTransporter>();

    std::cout << "\nProblem dimensions:" << std::endl;
    std::cout << "  State dimension: " << problem->getStateDim() << std::endl;
    std::cout << "  Control dimension: " << problem->getControlDim() << std::endl;
    std::cout << "  Complementarity constraints: " << problem->getComplementarityDim() << std::endl;
    std::cout << "  Equality constraints: " << problem->getEqualityConstraintDim() << std::endl;
    std::cout << "  Inequality constraints: " << problem->getInequalityConstraintDim() << std::endl;
    std::cout << "  Time step: " << problem->getTimeStep() << " s" << std::endl;

    // Configure Scholtes relaxation solver
    aula::ScholtesRelaxationConfig config;
    config.horizon = 300;  // N = 300 time steps as in original

    // Use parsed initial and goal states
    config.x_0 = x0;
    config.x_goal = x_goal;

    // Cost weights
    config.stage_cost_weight = 1e-6;
    config.stage_state_cost_weight = 0.0;
    config.final_cost_weight = 5000.0;

    // Scholtes relaxation parameters
    config.t_init = 1.0;
    config.t_final = 1e-8;
    config.gamma = 0.1;  // Faster relaxation (was 0.5, now t reduces by 4x each iteration)
    config.max_outer_iters = 20;
    config.comp_tol = 1e-5;

    // IPOPT options
    config.ipopt_print_level = 0;
    config.ipopt_max_iter = 1000;
    config.ipopt_tol = 1e-6;
    config.warm_start = true;

    // Verbosity
    config.print_level = 1;
    config.print_time = false;

    std::cout << "\nOptimization configuration:" << std::endl;
    std::cout << "  Horizon: " << config.horizon << " steps" << std::endl;
    std::cout << "  Initial state: [" << config.x_0(0) << ", " << config.x_0(1) << ", "
              << config.x_0(2) << ", " << config.x_0(3) << "]" << std::endl;
    std::cout << "  Goal state: [" << config.x_goal(0) << ", " << config.x_goal(1) << ", "
              << config.x_goal(2) << ", " << config.x_goal(3) << "]" << std::endl;

    // Create solver and solve
    std::cout << "\n=== Solving MPCC with Scholtes Relaxation ===" << std::endl;
    aula::MPCCScholtesRelaxationSolver solver(problem);

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

    std::cout << "\nFinal state: ["
              << static_cast<double>(solution.state_trajectory(0, config.horizon)) << ", "
              << static_cast<double>(solution.state_trajectory(1, config.horizon)) << ", "
              << static_cast<double>(solution.state_trajectory(2, config.horizon)) << ", "
              << static_cast<double>(solution.state_trajectory(3, config.horizon)) << "]"
              << std::endl;
    std::cout << "Goal state:  [" << config.x_goal(0) << ", " << config.x_goal(1) << ", "
              << config.x_goal(2) << ", " << config.x_goal(3) << "]" << std::endl;

    // Save solution to file with all metrics
    // For relaxation, iterations = total_ipopt_iterations (cumulative inner loop iterations)
    saveSolutionToFile(solution, config.x_0, config.x_goal, solution.total_ipopt_iterations,
                       solution.solve_time, output_file);

    // Check if goal is reached (position error threshold)
    Eigen::VectorXd final_state(4);
    for (int i = 0; i < 4; ++i) {
        final_state(i) = static_cast<double>(solution.state_trajectory(i, config.horizon));
    }
    Eigen::VectorXd goal_state(4);
    for (int i = 0; i < 4; ++i) {
        goal_state(i) = static_cast<double>(config.x_goal(i));
    }
    double position_error = (final_state.head(2) - goal_state.head(2)).norm();
    double velocity_error = (final_state.tail(2) - goal_state.tail(2)).norm();
    constexpr double POSITION_THRESHOLD = 0.01;  // 1cm
    constexpr double VELOCITY_THRESHOLD = 0.01;  // 0.1 m/s
    bool goal_reached =
        (position_error < POSITION_THRESHOLD) && (velocity_error < VELOCITY_THRESHOLD);

    std::cout << "\nGoal reaching check:" << std::endl;
    std::cout << "  Position error: " << position_error << " m (threshold: " << POSITION_THRESHOLD
              << ")" << std::endl;
    std::cout << "  Velocity error: " << velocity_error << " m/s (threshold: " << VELOCITY_THRESHOLD
              << ")" << std::endl;
    std::cout << "  Goal reached: " << (goal_reached ? "YES" : "NO") << std::endl;

    std::cout << "\n=== Optimization Complete ===" << std::endl;

    // Return non-zero if goal not reached or solver did not converge
    if (!solution.converged || !goal_reached) {
        return 1;
    }
    return 0;
}
