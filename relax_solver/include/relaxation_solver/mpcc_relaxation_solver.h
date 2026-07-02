#pragma once

#include <casadi/casadi.hpp>
#include <memory>
#include <string>

#include "impact/problem.h"

namespace aula {

// Scholtes relaxation baseline using the MPCCProblem interface from impact_solver.
using MPCCProblem = impact::MPCCProblem;

/**
 * @brief Solution structure for MPCC Scholtes relaxation optimization
 */
struct ScholtesRelaxationSolution {
    casadi::DM state_trajectory;    // State trajectory: (state_dim x horizon+1)
    casadi::DM control_trajectory;  // Control trajectory: (control_dim x horizon)
    double objective_value;         // Final objective value
    bool converged;                 // Whether optimization converged
    std::string status_message;     // Solver status message
    int outer_iterations;           // Number of outer iterations
    int total_ipopt_iterations;     // Total IPOPT iterations across all outer loops
    double solve_time;              // Total solution time in seconds
    double max_complementarity;     // Maximum complementarity violation at convergence
    double final_relaxation_t;      // Final relaxation parameter t
};

/**
 * @brief Configuration parameters for Scholtes relaxation solver
 */
struct ScholtesRelaxationConfig {
    // Problem parameters
    int horizon;        // Planning horizon
    casadi::DM x_goal;  // Goal state
    casadi::DM x_0;     // Initial state

    // Cost weights
    double stage_cost_weight;        // Weight for control stage cost
    double stage_state_cost_weight;  // Weight for state tracking at each stage
    double final_cost_weight;        // Weight for terminal cost

    // Scholtes relaxation parameters
    double t_init;        // Initial relaxation parameter (G*H <= t)
    double t_final;       // Final/minimum relaxation parameter
    double gamma;         // Relaxation parameter reduction factor (t_new = t * gamma)
    int max_outer_iters;  // Maximum outer iterations (relaxation steps)
    double comp_tol;      // Complementarity tolerance for stopping

    // IPOPT solver options
    int ipopt_print_level;        // IPOPT verbosity (0-12)
    int ipopt_max_iter;           // Maximum iterations per solve
    double ipopt_tol;             // Convergence tolerance
    double ipopt_acceptable_tol;  // Acceptable tolerance
    bool warm_start;              // Enable warm starting between iterations
    bool print_time;              // Print timing information

    // Verbosity
    int print_level;  // Solver verbosity (0=silent, 1=summary, 2=detailed)

    // Default constructor with reasonable defaults
    ScholtesRelaxationConfig()
        : horizon(50),
          stage_cost_weight(0.5),
          stage_state_cost_weight(0.0),
          final_cost_weight(2000.0),
          t_init(1.0),
          t_final(1e-10),
          gamma(0.5),
          max_outer_iters(50),
          comp_tol(1e-5),
          ipopt_print_level(3),
          ipopt_max_iter(500),
          ipopt_tol(1e-6),
          ipopt_acceptable_tol(1e-4),
          warm_start(true),
          print_time(false),
          print_level(1) {
        x_goal = casadi::DM::zeros(3, 1);
        x_0 = casadi::DM::zeros(3, 1);
    }
};

/**
 * @brief MPCC Scholtes relaxation solver using IPOPT
 *
 * This solver handles MPCC problems using the Scholtes relaxation approach:
 * 1. Relaxes complementarity constraints: G >= 0, H >= 0, G*H <= t
 * 2. Progressively decreases t from t_init to t_final
 * 3. Uses warm-starting between iterations for efficiency
 * 4. Stops when max|G*H| < comp_tol
 *
 * Reference: Scholtes, S. (2001). Convergence properties of a regularization
 *            scheme for mathematical programs with complementarity constraints.
 */
class MPCCScholtesRelaxationSolver {
   public:
    /**
     * @brief Constructor
     * @param problem Shared pointer to MPCC problem instance
     */
    explicit MPCCScholtesRelaxationSolver(std::shared_ptr<MPCCProblem> problem);

    /**
     * @brief Solve the MPCC problem using Scholtes relaxation
     * @param config Configuration parameters
     * @return Solution structure containing trajectories and solver info
     */
    ScholtesRelaxationSolution solve(const ScholtesRelaxationConfig& config);

    /**
     * @brief Solve with initial guess for warm-starting the first iteration
     * @param config Configuration parameters
     * @param x_init Initial guess for state trajectory (state_dim x horizon+1)
     * @param u_init Initial guess for control trajectory (control_dim x horizon)
     * @return Solution structure
     */
    ScholtesRelaxationSolution solveWithInitialGuess(const ScholtesRelaxationConfig& config,
                                                     const casadi::DM& x_init,
                                                     const casadi::DM& u_init);

   private:
    std::shared_ptr<MPCCProblem> problem_;

    /**
     * @brief Create IPOPT solver options dictionary
     */
    casadi::Dict createIPOPTOptions(const ScholtesRelaxationConfig& config);

    /**
     * @brief Build the NLP objective and constraints (symbolic)
     */
    void buildNLP(const ScholtesRelaxationConfig& config, casadi::SX& opt_vars, casadi::SX& obj,
                  casadi::SX& g, int& nx_total, int& nu_total, int& nc);

    /**
     * @brief Build variable bounds (fixed across all iterations)
     */
    void buildVariableBounds(const ScholtesRelaxationConfig& config, std::vector<double>& lbx,
                             std::vector<double>& ubx);

    /**
     * @brief Build constraint bounds for a given relaxation parameter t
     * @param config Configuration parameters
     * @param t Current relaxation parameter
     * @param lbg Output lower bounds for constraints
     * @param ubg Output upper bounds for constraints
     */
    void buildConstraintBounds(const ScholtesRelaxationConfig& config, double t,
                               std::vector<double>& lbg, std::vector<double>& ubg);

    /**
     * @brief Compute maximum complementarity violation from solution
     */
    double computeMaxComplementarity(const casadi::DM& x_sol,
                                     const ScholtesRelaxationConfig& config);

    /**
     * @brief Extract solution from optimization result
     */
    ScholtesRelaxationSolution extractSolution(const std::map<std::string, casadi::DM>& sol_map,
                                               const ScholtesRelaxationConfig& config, int nx_total,
                                               int nu_total);
};

}  // namespace aula
