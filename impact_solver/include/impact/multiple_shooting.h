#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>

#include "impact/aula_subproblem.h"
#include "impact/bcd_aula_config.h"
#include "impact/bcd_aula_solver.h"
#include "impact/problem.h"
#include "impact/stage_problem.h"

namespace impact {

/**
 * @brief Multiple-shooting subproblem and offsets for per-solve data.
 */
struct MultipleShootingLayout {
    std::unique_ptr<AulaSubproblem> sub;
    int off_x0 = 0;  // initial state x_0 (initial-condition equality block)
    int off_p = 0;   // task runtime parameter p (e.g. x_goal)
};

/**
 * @brief Build a multiple-shooting subproblem from a StageProblem.
 *
 * Decision vector z = [vec(X); vec(U)]; the state trajectory is free, dynamics are
 * enforced as the AuLa defect block X_{k+1} - step(X_k, U_k), and the initial
 * condition as a second equality block.
 */
MultipleShootingLayout buildMultipleShooting(const StageProblem& stage,
                                             const BCDAULAConfig& config);

/**
 * @brief Solution of a multiple-shooting solve (trajectories + solver stats).
 */
struct MultipleShootingSolution {
    Eigen::MatrixXd state_trajectory;    // nx x (horizon + 1)
    Eigen::MatrixXd control_trajectory;  // nu x horizon

    double objective_value = 0.0;
    double dynamics_violation = 0.0;
    double equality_violation = 0.0;
    double inequality_violation = 0.0;
    double complementarity_violation = 0.0;
    bool converged = false;
    int outer_iterations = 0;
    int total_inner_iterations = 0;
    double solve_time = 0.0;
    std::string status_message;
};

/**
 * @brief Thin multiple-shooting front-end for MPCC tasks.
 *
 * Wraps the task in an MPCCStage, builds the subproblem, flattens the initial
 * guess, runs BCDAULASolver, and reshapes z into (X, U).
 */
class MultipleShootingSolver {
   public:
    explicit MultipleShootingSolver(std::shared_ptr<MPCCProblem> problem)
        : problem_(std::move(problem)) {}

    MultipleShootingSolution solve(const BCDAULAConfig& config);
    MultipleShootingSolution solveWithInitialGuess(const BCDAULAConfig& config,
                                                   const Eigen::MatrixXd& x_init,
                                                   const Eigen::MatrixXd& u_init);

   private:
    std::shared_ptr<MPCCProblem> problem_;
};

}  // namespace impact
