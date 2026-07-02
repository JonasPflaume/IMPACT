#include <sys/stat.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "penalty_solver/mpcc_penalty_solver.h"
#include "push_t.h"

/**
 * @brief Parse command line arguments for initial state, goal state, and output file
 * @param argc Argument count
 * @param argv Argument values
 * @param x0 Output initial state (3D: px, py, theta)
 * @param x_goal Output goal state (3D: px, py, theta)
 * @param output_file Output file path
 * @return true if parsing succeeded, false otherwise
 */
bool parseArgs(int argc, char* argv[], casadi::DM& x0, casadi::DM& x_goal,
               std::string& output_file) {
    // Usage: ./push_t_penalty x0_0 x0_1 x0_2 goal_0 goal_1 goal_2 [output_file]
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " x0_px x0_py x0_theta goal_px goal_py goal_theta [output_file]\n";
        std::cerr << "  x0: initial state (px, py, theta)\n";
        std::cerr << "  goal: goal state (px, py, theta)\n";
        std::cerr << "  output_file: optional output file path (default: "
                     "results/push_t/penalty/trajectory_<timestamp>.txt)\n";
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
        // Generate default output file with timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << "results/push_t/penalty/trajectory_" << timestamp << ".txt";
        output_file = oss.str();
    }

    return true;
}

/**
 * @brief Save solution with all required metrics for paper evaluation
 */
void saveSolutionToFile(const aula::MPCCSolution& solution, const casadi::DM& x0,
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
    file << "# PushT Penalty-IPOPT Trajectory\n";
    file << "# Planner: penalty\n";
    file << "# Task: push_t\n\n";

    // Write start state
    file << "# Start State (px, py, theta)\n";
    for (int i = 0; i < x0.rows(); ++i) {
        file << static_cast<double>(x0(i));
        if (i < x0.rows() - 1) file << " ";
    }
    file << "\n\n";

    // Write goal state
    file << "# Goal State (px, py, theta)\n";
    for (int i = 0; i < x_goal.rows(); ++i) {
        file << static_cast<double>(x_goal(i));
        if (i < x_goal.rows() - 1) file << " ";
    }
    file << "\n\n";

    // Write iterations and solve time
    file << "# Iterations\n" << iterations << "\n\n";
    file << "# Solve Time (seconds)\n" << solve_time << "\n\n";

    // Write success status
    file << "# Success\n" << (solution.success ? 1 : 0) << "\n\n";

    // Write state trajectory
    file << "# State Trajectory (rows: timesteps, cols: px, py, theta)\n";
    int num_states = solution.state_trajectory.columns();
    int state_dim = solution.state_trajectory.rows();
    for (int k = 0; k < num_states; ++k) {
        for (int i = 0; i < state_dim; ++i) {
            file << static_cast<double>(solution.state_trajectory(i, k));
            if (i < state_dim - 1) file << " ";
        }
        file << "\n";
    }

    // Write control trajectory
    file << "\n# Control Trajectory\n";
    int num_controls = solution.control_trajectory.columns();
    int control_dim = solution.control_trajectory.rows();
    for (int k = 0; k < num_controls; ++k) {
        for (int i = 0; i < control_dim; ++i) {
            file << static_cast<double>(solution.control_trajectory(i, k));
            if (i < control_dim - 1) file << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Solution saved to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== PushT MPCC Trajectory Optimization ===" << std::endl;

    // Parse command line arguments
    casadi::DM x0, x_goal;
    std::string output_file;
    if (!parseArgs(argc, argv, x0, x_goal, output_file)) {
        return 1;
    }

    // Create PushT problem
    auto push_t_problem = std::make_shared<push_t::PushT>();

    std::cout << "Problem dimensions:" << std::endl;
    std::cout << "  State dimension: " << push_t_problem->getStateDim() << std::endl;
    std::cout << "  Control dimension: " << push_t_problem->getControlDim() << std::endl;
    std::cout << "  Complementarity constraints: " << push_t_problem->getComplementarityDim()
              << std::endl;
    std::cout << "  Equality constraints: " << push_t_problem->getEqualityConstraintDim()
              << std::endl;
    std::cout << "  Inequality constraints: " << push_t_problem->getInequalityConstraintDim()
              << std::endl;
    std::cout << "  Time step: " << push_t_problem->getTimeStep() << " s" << std::endl;

    // Configure MPCC solver
    aula::MPCCPenaltyConfig config;
    config.horizon = 50;

    // Use parsed initial and goal states
    config.x_0 = x0;
    config.x_goal = x_goal;

    // Cost weights
    config.stage_cost_weight = 0.01;
    config.stage_state_cost_weight = 0.0;
    config.final_cost_weight = 100.0;
    config.complementarity_penalty_weight = 10000.0;

    // Solver options
    config.ipopt_print_level = 5;
    config.ipopt_max_iter = 1000;
    config.ipopt_tol = 1e-5;
    config.ipopt_acceptable_tol = 1e-5;
    config.print_time = true;

    std::cout << "\nOptimization configuration:" << std::endl;
    std::cout << "  Horizon: " << config.horizon << " steps" << std::endl;
    std::cout << "  Initial state: [" << config.x_0(0) << ", " << config.x_0(1) << ", "
              << config.x_0(2) << "]" << std::endl;
    std::cout << "  Goal state: [" << config.x_goal(0) << ", " << config.x_goal(1) << ", "
              << config.x_goal(2) << "]" << std::endl;
    std::cout << "  Stage cost weight: " << config.stage_cost_weight << std::endl;
    std::cout << "  Final cost weight: " << config.final_cost_weight << std::endl;
    std::cout << "  Complementarity penalty: " << config.complementarity_penalty_weight
              << std::endl;

    // Create solver and solve
    std::cout << "\n=== Solving MPCC problem with IPOPT ===" << std::endl;
    aula::MPCCPenaltySolver solver(push_t_problem);

    aula::MPCCSolution solution = solver.solve(config);

    // Print results
    std::cout << "\n=== Solution Summary ===" << std::endl;
    std::cout << "Success: " << (solution.success ? "YES" : "NO") << std::endl;
    std::cout << "Objective value: " << solution.objective_value << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " seconds" << std::endl;

    std::cout << "\nFinal state: [" << solution.state_trajectory(0, config.horizon).scalar() << ", "
              << solution.state_trajectory(1, config.horizon).scalar() << ", "
              << solution.state_trajectory(2, config.horizon).scalar() << "]" << std::endl;

    // Save solution to file with all metrics
    saveSolutionToFile(solution, config.x_0, config.x_goal, solution.iterations,
                       solution.solve_time, output_file);

    std::cout << "\n=== Optimization Complete ===" << std::endl;

    return 0;
}
