#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>

#include "impact/aula_subproblem.h"
#include "impact/aula_config.h"
#include "impact/aula_solver.h"
#include "impact/lcp_problem.h"
#include "impact/lcp_stage.h"
#include "impact/stage_problem.h"

namespace impact {

/**
 * @brief Single-shooting subproblem and offsets for per-solve data.
 *
 * The state is rolled out symbolically from x_0 via StageProblem::step, so only
 * the controls are free. For LCP tasks x_0 = q0 and the runtime parameter p packs
 * the contact data (targets, phi, J) that change every MPC step; the symbolic
 * functions are built once.
 */
struct SingleShootingLayout {
    std::unique_ptr<AulaSubproblem> sub;
    int off_x0 = 0;  // initial state x_0 / q0 (rollout start)
    int off_p = 0;   // task runtime parameter p
};

/**
 * @brief Build a single-shooting subproblem from a StageProblem.
 *
 * z = [vec(U)]; the state is rolled out via step() so only the controls are free.
 * The cost is nonlinear in the controls (it sees the rolled state), so every row
 * is a penalty block (cost rows use rho = 1).
 */
SingleShootingLayout buildSingleShooting(const StageProblem& stage, const AulaConfig& config);

/**
 * @brief Solution of a single-shooting solve (trajectories + solver stats).
 */
struct SingleShootingSolution {
    Eigen::MatrixXd config_trajectory;    // n_qpos x (horizon + 1)
    Eigen::MatrixXd command_trajectory;   // n_cmd  x horizon
    Eigen::MatrixXd lambda_trajectory;    // n_lam  x horizon
    Eigen::MatrixXd velocity_trajectory;  // n_qvel x horizon
    Eigen::VectorXd first_command;

    double objective_value = 0.0;
    double dynamics_violation = 0.0;
    double equality_violation = 0.0;
    double inequality_violation = 0.0;
    double complementarity_violation = 0.0;
    double stationarity_violation = 0.0;
    // Tolerance-free MPCC complementarity certificate; see AulaResult.
    double comp_neg_G = 0.0;
    double comp_neg_H = 0.0;
    double comp_support_G = 0.0;
    double comp_support_H = 0.0;
    bool converged = false;
    int outer_iterations = 0;
    int total_inner_iterations = 0;
    int total_gn_iterations = 0;
    double solve_time = 0.0;
    std::string status_message;
};

/**
 * @brief Thin single-shooting front-end for LCP contact tasks.
 *
 * Wraps the task in an LCPStage, builds the subproblem once (per horizon), then
 * per solve writes the contact data + targets, resets/warm-starts the decision
 * vector and AuLa state, runs AulaSolver, and rolls q out from the optimized
 * velocities.
 */
class SingleShootingSolver {
   public:
    explicit SingleShootingSolver(std::shared_ptr<LCPProblem> problem)
        : problem_(std::move(problem)) {}

    SingleShootingSolution solve(const AulaConfig& config, const Eigen::VectorXd& q0,
                                 const Eigen::VectorXd& phi_vec, const Eigen::MatrixXd& jac_mat,
                                 const Eigen::Vector3d& target_p, const Eigen::Vector4d& target_q);

    SingleShootingSolution solveWithInitialGuess(
        const AulaConfig& config, const Eigen::VectorXd& q0, const Eigen::VectorXd& phi_vec,
        const Eigen::MatrixXd& jac_mat, const Eigen::Vector3d& target_p,
        const Eigen::Vector4d& target_q, const Eigen::MatrixXd& cmd_init,
        const Eigen::MatrixXd& lam_init, const Eigen::MatrixXd& vel_init);

   private:
    void ensureBuilt(const AulaConfig& config);

    std::shared_ptr<LCPProblem> problem_;
    std::unique_ptr<LCPStage> stage_;
    SingleShootingLayout layout_;
    // Reuse the solver so Gauss-Newton sparsity maps and saddle pattern analysis
    // survive across MPC steps on the cached subproblem.
    AulaSolver solver_;
    int horizon_built_ = -1;

    int n_qpos_ = 0, n_qvel_ = 0, n_cmd_ = 0, n_lam_ = 0, vars_per_step_ = 0;
    double h_ = 0.0;
};

}  // namespace impact
