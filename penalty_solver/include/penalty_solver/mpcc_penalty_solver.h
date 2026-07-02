#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>
#include <string>

#include "impact/problem.h"

namespace aula {

// IPOPT penalty baseline using the MPCCProblem interface from impact_solver.
using MPCCProblem = impact::MPCCProblem;

/**
 * @brief Solution structure for MPCC penalty method optimization
 */
struct MPCCSolution {
    casadi::DM state_trajectory;    // State trajectory: (state_dim x horizon+1)
    casadi::DM control_trajectory;  // Control trajectory: (control_dim x horizon)
    double objective_value;         // Final objective value
    bool success;                   // Whether optimization succeeded
    std::string solver_stats;       // Solver statistics
    int iterations;                 // Number of iterations
    double solve_time;              // Solution time in seconds
};

/**
 * @brief Configuration parameters for MPCC penalty-based solver
 */
struct MPCCPenaltyConfig {
    int horizon;                            // Planning horizon
    casadi::DM x_goal;                      // Goal state
    casadi::DM x_0;                         // Initial state
    double stage_cost_weight;               // Weight for control stage cost
    double stage_state_cost_weight;         // Weight for state tracking at each stage
    double final_cost_weight;               // Weight for terminal cost
    double complementarity_penalty_weight;  // Penalty weight for complementarity violations

    // IPOPT solver options
    int ipopt_print_level;        // IPOPT verbosity (0-12)
    int ipopt_max_iter;           // Maximum iterations
    double ipopt_tol;             // Convergence tolerance
    double ipopt_acceptable_tol;  // Acceptable tolerance
    bool print_time;              // Print timing information

    // Default constructor with reasonable defaults
    MPCCPenaltyConfig()
        : horizon(20),
          stage_cost_weight(0.5),
          stage_state_cost_weight(0.0),  // Default: no stage state tracking
          final_cost_weight(2000.0),
          complementarity_penalty_weight(2000.0),
          ipopt_print_level(5),
          ipopt_max_iter(3000),
          ipopt_tol(1e-6),
          ipopt_acceptable_tol(1e-4),
          print_time(false) {
        x_goal = casadi::DM::zeros(3, 1);
        x_0 = casadi::DM::zeros(3, 1);
    }
};

/**
 * @brief MPCC penalty-based solver using IPOPT
 *
 * This solver converts MPCC problems into standard NLP problems by:
 * 1. Adding penalty terms for complementarity constraint violations: G(x,u) * H(x,u)
 * 2. Adding inequality constraints: G(x,u) >= 0, H(x,u) >= 0
 * 3. Minimizing stage cost, terminal cost, and penalty terms
 */
class MPCCPenaltySolver {
   public:
    /**
     * @brief Constructor
     * @param problem Shared pointer to MPCC problem instance
     */
    explicit MPCCPenaltySolver(std::shared_ptr<MPCCProblem> problem);

    /**
     * @brief Solve the MPCC problem using penalty method with IPOPT
     * @param config Configuration parameters
     * @return Solution structure containing trajectories and solver info
     */
    MPCCSolution solve(const MPCCPenaltyConfig& config);

    /**
     * @brief Solve with initial guess for warm-starting
     * @param config Configuration parameters
     * @param x_init Initial guess for state trajectory (state_dim x horizon+1)
     * @param u_init Initial guess for control trajectory (control_dim x horizon)
     * @return Solution structure
     */
    MPCCSolution solveWithInitialGuess(const MPCCPenaltyConfig& config, const casadi::DM& x_init,
                                       const casadi::DM& u_init);

   private:
    std::shared_ptr<MPCCProblem> problem_;

    /**
     * @brief Build the NLP bounds (helper function)
     */
    void buildNLP(const MPCCPenaltyConfig& config, casadi::SX& opt_vars, std::vector<double>& lbx,
                  std::vector<double>& ubx, std::vector<double>& lbg, std::vector<double>& ubg,
                  int& nx_total, int& nu_total);

    /**
     * @brief Extract solution from optimization result
     */
    MPCCSolution extractSolution(const std::map<std::string, casadi::DM>& sol_map,
                                 const MPCCPenaltyConfig& config, int nx_total, int nu_total);

    /**
     * @brief Create IPOPT solver options dictionary
     */
    casadi::Dict createIPOPTOptions(const MPCCPenaltyConfig& config);
};

}  // namespace aula
