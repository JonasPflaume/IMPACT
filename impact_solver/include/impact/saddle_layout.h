#pragma once

#include <vector>

namespace impact {

/**
 * @brief Row layout of one penalty block inside the stacked AuLa residual r(z;p).
 *
 * The Gauss-Newton residual is r = [r_cost; r_penalty], where each penalty block
 * b stores its rows as  r_b = sqrt(rho_b) * (scale_b * c_b(z) + kappa_b / rho_b).
 * To assemble the primal saddle system the solver needs, per block, where its
 * rows live, how many there are, and where to read its penalty rho_b from the
 * persistent parameter buffer.
 */
struct SaddleBlock {
    int row_start;         // first row of this block within the full residual r
    int count;             // number of rows in this block
    int rho_param_offset;  // index of rho_b inside the CasADi parameter buffer
};

/**
 * @brief Block description of the AuLa least-squares residual for the saddle solve.
 *
 * Splits r into the rho-independent cost rows [0, n_cost) and the penalty blocks
 * that follow. With this, the damped Gauss-Newton normal equations
 *   (J_cost^T J_cost + sum_b rho_b C_b^T C_b + lambda I) dz = -J^T r
 * can be solved in the equivalent primal-dual saddle form
 *   [ J_cost^T J_cost + (lambda+sigma) I   C^T            ] [dz]   [ -J_cost^T r_cost ]
 *   [ C                                  -diag(1/rho_b)   ] [y ] = [      -chat        ]
 * where C_b = J_b / sqrt(rho_b) and chat_b = r_b / sqrt(rho_b) are recovered from
 * the existing residual/Jacobian by de-scaling each penalty block. Only dz is used.
 */
struct SaddleLayout {
    int n_z = 0;     // number of primal variables (= numOpt)
    int n_cost = 0;  // number of cost residual rows (rows [0, n_cost) of r)
    int n_dual = 0;  // total penalty rows = sum of block counts
    std::vector<SaddleBlock> blocks;  // penalty blocks, in residual-row order
};

}  // namespace impact
