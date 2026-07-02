#pragma once

#include <Eigen/Core>
#include <string>

#include "impact/aula_subproblem.h"
#include "impact/bcd_aula_config.h"
#include "impact/gauss_newton_solver.h"

namespace impact {

/**
 * @brief Terminal status of a solve.
 *
 * `converged` is kept for older callers and is equivalent to
 * `status == Converged`. The enum records whether a non-converged run simply hit
 * the outer-iteration limit or got stuck in the X-update linear algebra.
 */
enum class BCDAULAStatus { Converged, MaxIterations, LinearAlgebraFailure };

/**
 * @brief Result returned by BCDAULASolver.
 *
 * `z` is the raw decision vector. Shooting front-ends reshape it into task
 * trajectories. Reported constraint violations are unscaled.
 */
struct BCDAULAResult {
    Eigen::VectorXd z;

    double objective_value = 0.0;
    double dynamics_violation = 0.0;
    double equality_violation = 0.0;
    double inequality_violation = 0.0;
    double complementarity_violation = 0.0;  // ||G ∘ H||_inf

    bool converged = false;  // == (status == BCDAULAStatus::Converged)
    BCDAULAStatus status = BCDAULAStatus::MaxIterations;
    int outer_iterations = 0;
    int total_inner_iterations = 0;
    double solve_time = 0.0;
    std::string status_message;
};

/**
 * @brief Safeguarded Augmented-Lagrangian solver with an inner BCD loop.
 *
 * The solver consumes an AulaSubproblem, independent of the task that produced
 * it. In broad strokes it runs:
 *
 *   Outer (Algorithm 1): clip every dual block's multiplier to the safeguard box,
 *   solve the AuLa subproblem inexactly, dual-ascend each multiplier, then apply a
 *   per-block *safeguarded* penalty increase (rho grows only when a block's
 *   violation fails to decrease enough between outer iterations).
 *
 *   Inner (Algorithm 2): block coordinate descent alternating a damped
 *   Gauss-Newton X-update (GaussNewtonSolver) with the closed-form complementarity
 *   (Y,Z)-update (projectComplementarity), stopping on objective stagnation.
 *
 * The shooting builders decide which dual blocks exist; this loop only updates
 * the blocks it is given.
 */
class BCDAULASolver {
   public:
    BCDAULASolver() = default;

    /**
     * @brief Solve the subproblem in place starting from z_init.
     * @param sub      AuLa subproblem (dual-block AuLa state is mutated)
     * @param config   Solver hyper-parameters
     * @param z_init   Initial decision vector (size sub.numOpt())
     */
    BCDAULAResult solve(AulaSubproblem& sub, const BCDAULAConfig& config,
                        const Eigen::VectorXd& z_init);

   private:
    GaussNewtonSolver gn_;
};

}  // namespace impact
