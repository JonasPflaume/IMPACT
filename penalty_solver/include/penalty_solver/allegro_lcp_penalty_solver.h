#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "penalty_solver/allegro_lcp_problem.h"

namespace allegro_lcp {

/**
 * @brief Configuration for Allegro LCP Penalty Solver
 * Parameters matched from Python test_lcp_ipopt.py
 */
struct AllegroLCPPenaltyConfig {
    int horizon = 3;                       // From Python: mpc_horizon_ = 3
    double complementarity_penalty = 1e4;  // From Python: comp_penalty = 1e4
    double velocity_penalty = 1.0;         // From Python: velocity_penalty = 1.0 (commented out)
    int ipopt_max_iter = 50;               // From Python: ipopt_max_iter_ = 50
    int ipopt_print_level = 0;             // IPOPT verbosity (0 = silent)
    double ipopt_tol = 1e-6;               // From Python: ipopt.tol = 1e-6
    double acceptable_tol = 1e-6;          // From Python: ipopt.acceptable_tol = 1e-6
    int acceptable_iter = 5;               // From Python: ipopt.acceptable_iter = 5
    bool warm_start = true;                // From Python: warm_start_init_point = 'yes'
    bool print_time = false;               // Print timing information
};

/**
 * @brief Solution structure for Allegro LCP Penalty Solver
 */
struct AllegroLCPSolution {
    Eigen::VectorXd action;             // First command to execute (n_cmd = 16)
    std::vector<double> full_solution;  // Full primal decision variable solution
    std::vector<double> lam_g;          // Dual variables for constraints (for warm start)
    std::vector<double> lam_x;          // Dual variables for bounds (for warm start)
    double cost;                        // Optimal cost
    std::string status;                 // Solver status string
    bool success;                       // Whether solver succeeded
    double solve_time;                  // Solution time in seconds
};

/**
 * @brief Warm start structure for time-shifting
 */
struct AllegroWarmStart {
    std::vector<double> primal;  // Primal variables
    std::vector<double> dual_g;  // Constraint dual variables
    std::vector<double> dual_x;  // Bound dual variables
    bool valid = false;
};

/**
 * @brief Penalty-based solver for Allegro hand LCP contact dynamics
 *
 * This solver matches the Python test_lcp_ipopt.py implementation exactly:
 * - LCP dynamics: h * Q * vel = b + J^T * lam
 * - Complementarity: 0 <= lam ⊥ (J * vel + phi/h) >= 0
 * - Cost functions from Python init_cost_fns()
 *
 * The solver is built once and can be called repeatedly with different parameters.
 */
class AllegroLCPPenaltySolver {
   public:
    /**
     * @brief Construct solver for given problem
     * @param problem Shared pointer to Allegro LCP problem definition
     * @param config Solver configuration
     */
    AllegroLCPPenaltySolver(std::shared_ptr<AllegroLCPProblem> problem,
                            const AllegroLCPPenaltyConfig& config);

    /**
     * @brief Solve the LCP problem with given runtime parameters
     * @param q0 Initial configuration (23D: [obj_pos(3), obj_quat(4), robot_qpos(16)])
     * @param phi_vec Gap function values (max_ncon * 4 = 80)
     * @param jac_mat Contact Jacobian (80 x 22)
     * @param target_p Target position (3)
     * @param target_q Target quaternion (4)
     * @param warm_start Optional warm start from previous solution
     * @return Solution structure
     */
    AllegroLCPSolution solve(const Eigen::VectorXd& q0, const Eigen::VectorXd& phi_vec,
                             const Eigen::MatrixXd& jac_mat, const Eigen::Vector3d& target_p,
                             const Eigen::Vector4d& target_q,
                             const AllegroWarmStart* warm_start = nullptr);

    /**
     * @brief Time-shift warm start for next MPC iteration
     * @param current_sol Current solution
     * @return Shifted warm start for next iteration
     */
    AllegroWarmStart timeShiftWarmStart(const AllegroLCPSolution& current_sol);

    /**
     * @brief Get the number of decision variables per time step
     */
    int getVarsPerStep() const { return vars_per_step_; }

    /**
     * @brief Get the total number of decision variables
     */
    int getTotalVars() const { return total_vars_; }

    /**
     * @brief Get the number of constraints per time step
     */
    int getConstraintsPerStep() const { return g_per_step_; }

   private:
    std::shared_ptr<AllegroLCPProblem> problem_;
    AllegroLCPPenaltyConfig config_;
    casadi::Function solver_;

    // Cached bounds
    std::vector<double> lbw_, ubw_, lbg_, ubg_;
    std::vector<double> w0_;  // Default initial guess
    int total_vars_, param_size_, vars_per_step_, g_per_step_;

    void buildSolver();

    // Build CasADi functions for fingertip forward kinematics
    void buildFingertipKinematics();

    // Fingertip FK functions (from Python allegro_fkin.py)
    casadi::Function fftp_pos_fn_;  // First finger tip position
    casadi::Function mftp_pos_fn_;  // Middle finger tip position
    casadi::Function rftp_pos_fn_;  // Ring finger tip position
    casadi::Function thtp_pos_fn_;  // Thumb tip position
};

}  // namespace allegro_lcp
