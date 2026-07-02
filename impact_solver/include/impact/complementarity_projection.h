#pragma once

#include <Eigen/Core>

namespace impact {

/**
 * @brief Closed-form (Y, Z)-update used inside the BCD loop.
 *
 * With the trajectory variables X fixed, the augmented-Lagrangian objective
 * decouples across complementarity pairs. For each scalar pair (y_i, z_i) under
 * the hard constraint 0 <= y_i ⊥ z_i >= 0, the exact minimiser is one of two
 * candidates (one branch per leg of the complementarity set):
 *
 *   Case 1 (z_i = 0):  y_i* = max(0, G_i + kappaG_i / (rho * s))
 *   Case 2 (y_i = 0):  z_i* = max(0, H_i + kappaH_i / (rho * s))
 *
 * where s = comp_scale conditions the complementarity residual. We evaluate the
 * penalty contribution of each candidate and keep the smaller one. Writing the
 * shifted residuals  aG = s*G + kappaG/rho,  aH = s*H + kappaH/rho,  the two
 * costs reduce to  cost1 = min(aG,0)^2 + aH^2  and  cost2 = aG^2 + min(aH,0)^2.
 *
 * This is an exact block minimisation, so it never increases the objective. The
 * implementation is vectorised over all complementarity pairs with Eigen arrays.
 *
 * @param G,H            Complementarity function values at the current X
 * @param kappaG,kappaH  Complementarity-split multipliers
 * @param rho_comp       Complementarity penalty
 * @param comp_scale     Complementarity conditioning scale (s)
 * @param[out] sG,sH     Updated slack variables (y, z)
 */
inline void projectComplementarity(const Eigen::VectorXd& G, const Eigen::VectorXd& H,
                                   const Eigen::VectorXd& kappaG, const Eigen::VectorXd& kappaH,
                                   double rho_comp, double comp_scale, Eigen::VectorXd& sG,
                                   Eigen::VectorXd& sH) {
    const double rs = rho_comp * comp_scale;

    // Per-candidate minimisers (projected onto the nonnegative leg).
    const Eigen::ArrayXd sG_case1 = (G.array() + kappaG.array() / rs).max(0.0);
    const Eigen::ArrayXd sH_case2 = (H.array() + kappaH.array() / rs).max(0.0);

    // Shifted residuals; costs of the two candidates.
    const Eigen::ArrayXd aG = comp_scale * G.array() + kappaG.array() / rho_comp;
    const Eigen::ArrayXd aH = comp_scale * H.array() + kappaH.array() / rho_comp;
    const Eigen::ArrayXd cost1 = aG.min(0.0).square() + aH.square();
    const Eigen::ArrayXd cost2 = aG.square() + aH.min(0.0).square();

    const Eigen::Array<bool, Eigen::Dynamic, 1> use_case1 = (cost1 <= cost2);
    sG = use_case1.select(sG_case1, 0.0).matrix();
    sH = use_case1.select(0.0, sH_case2).matrix();
}

}  // namespace impact
