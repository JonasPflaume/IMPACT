#pragma once

#include <Eigen/Core>
#include <functional>
#include <string>

namespace impact {

/**
 * @brief One Augmented-Lagrangian constraint channel.
 *
 * Each block owns its penalty rho and multiplier kappa, evaluates its scaled
 * residual c(z), and declares whether it is an equality or a one-sided
 * inequality. BCDAULASolver applies the same safeguarded update to every block.
 *
 *   Equality   : kappa <- kappa + rho * c ;   violation = ||c||_inf
 *   Inequality : kappa <- max(0, kappa + rho * c) ; violation = ||max(c,0)||_inf
 *                (one-sided g <= 0; two-sided bounds are stacked into one block)
 *
 * The complementarity channel is handled separately (CompBlock) because its
 * (Y,Z)-update is the closed-form projection, not a dual ascent.
 */
enum class DualKind { Equality, Inequality };

struct DualBlock {
    std::string name;
    DualKind kind = DualKind::Equality;
    int dim = 0;
    double scale = 1.0;  // conditioning scale already folded into c(z) and the residual
    double tol = 1e-5;   // unscaled convergence tolerance for this channel

    double rho = 1.0;       // penalty, mutated by the solver
    double rho_init = 1.0;  // build-time seed used for per-solve resets
    Eigen::VectorXd kappa;  // multiplier (Equality) or mu>=0 (Inequality)

    // Offsets of this block's kappa and rho inside the subproblem parameter buffer.
    int kappa_offset = 0;
    int rho_offset = 0;

    // Scaled constraint residual c(z) (the `scale` is already applied), supplied by
    // the subproblem builder as a CasADi-backed closure.
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> eval_scaled;
};

/**
 * @brief The complementarity channel: 0 <= G(z) ⊥ H(z) >= 0.
 *
 * Updated by the closed-form projection (projectComplementarity) on the slacks
 * (sG, sH), then a dual ascent on the split multipliers (kappaG, kappaH). The
 * split residual the penalty acts on is comp_scale * (G - sG) and (H - sH).
 */
struct CompBlock {
    std::string name = "comp";
    int dim = 0;
    double scale = 1.0;  // comp_scale
    double tol = 1e-5;   // outer complementarity tolerance
    double rho = 1.0;    // rho_comp (state)
    double rho_init = 1.0;  // build-time seed used for per-solve resets

    Eigen::VectorXd kappaG, kappaH;  // split multipliers (state)
    Eigen::VectorXd sG, sH;          // slacks (state, from projection)

    // Offsets inside the subproblem parameter buffer.
    int kappaG_offset = 0;
    int kappaH_offset = 0;
    int sG_offset = 0;
    int sH_offset = 0;
    int rho_offset = 0;
};

}  // namespace impact
