#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "impact/lcp_problem.h"

namespace aula {

// IPOPT penalty baseline using the LCPProblem interface from impact_solver.
using LCPProblem = impact::LCPProblem;

/**
 * @brief Configuration for LCP Penalty Solver
 */
struct LCPPenaltyConfig {
    int horizon = 4;                       // Planning horizon
    double complementarity_penalty = 1e6;  // Penalty for complementarity violation
    double velocity_penalty = 1.0;         // Additional velocity damping
    int ipopt_max_iter = 500;              // Maximum IPOPT iterations
    int ipopt_print_level = 0;             // IPOPT verbosity (0 = silent)
    double ipopt_tol = 1e-5;               // Convergence tolerance
    bool print_time = false;               // Print timing information
};

/**
 * @brief Solution structure for LCP Penalty Solver
 */
struct LCPSolution {
    Eigen::VectorXd action;             // First command to execute (n_cmd)
    std::vector<double> full_solution;  // Full decision variable solution (for warm start)
    double cost;                        // Optimal cost
    std::string status;                 // Solver status string
    bool success;                       // Whether solver succeeded
    double solve_time;                  // Solution time in seconds
};

/**
 * @brief Penalty-based solver for LCP contact dynamics problems
 *
 * This solver handles problems with:
 * - Parametric contact information (gap function phi, contact Jacobian J)
 * - LCP dynamics: h * Q * vel = b + J^T * lam
 * - Complementarity: 0 <= lam ⊥ (J * vel + phi/h) >= 0
 *
 * The solver is built once and can be called repeatedly with different parameters.
 */
class LCPPenaltySolver {
   public:
    /**
     * @brief Construct solver for given problem
     * @param problem Shared pointer to LCP problem definition
     * @param config Solver configuration
     */
    LCPPenaltySolver(std::shared_ptr<LCPProblem> problem, const LCPPenaltyConfig& config);

    /**
     * @brief Solve the LCP problem with given runtime parameters
     * @param q0 Initial configuration (n_qpos)
     * @param phi_vec Gap function values (max_ncon * 4)
     * @param jac_mat Contact Jacobian (max_ncon * 4 x n_qvel)
     * @param target_p Target position (3)
     * @param target_q Target quaternion (4)
     * @param warm_start Optional warm start from previous solution
     * @return Solution structure
     */
    LCPSolution solve(const Eigen::VectorXd& q0, const Eigen::VectorXd& phi_vec,
                      const Eigen::MatrixXd& jac_mat, const Eigen::Vector3d& target_p,
                      const Eigen::Vector4d& target_q,
                      const std::vector<double>* warm_start = nullptr);

    /**
     * @brief Get the number of decision variables per time step
     */
    int getVarsPerStep() const { return vars_per_step_; }

    /**
     * @brief Get the total number of decision variables
     */
    int getTotalVars() const { return total_vars_; }

   private:
    std::shared_ptr<LCPProblem> problem_;
    LCPPenaltyConfig config_;
    casadi::Function solver_;

    // Cached bounds
    std::vector<double> lbw_, ubw_, lbg_, ubg_;
    int total_vars_, param_size_, vars_per_step_;

    void buildSolver();
};

}  // namespace aula
