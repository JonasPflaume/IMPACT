#include "impact/gauss_newton_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace impact {

namespace {
// Finite test that still works with Release builds using -ffast-math. Some
// compilers may fold std::isfinite() away under -ffinite-math-only, so the step
// guards below inspect the IEEE-754 exponent bits directly.
inline bool isFiniteValue(double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

// Slot of entry (row, col) in a compressed column-major sparse matrix, or -1.
int findSlot(const Eigen::SparseMatrix<double>& A, int row, int col) {
    const int* outer = A.outerIndexPtr();
    const int* inner = A.innerIndexPtr();
    for (int s = outer[col]; s < outer[col + 1]; ++s)
        if (inner[s] == row) return s;
    return -1;
}

bool sameLayout(const impact::SaddleLayout& a, const impact::SaddleLayout& b) {
    if (a.n_z != b.n_z || a.n_cost != b.n_cost || a.n_dual != b.n_dual ||
        a.blocks.size() != b.blocks.size())
        return false;
    for (size_t i = 0; i < a.blocks.size(); ++i) {
        if (a.blocks[i].row_start != b.blocks[i].row_start ||
            a.blocks[i].count != b.blocks[i].count ||
            a.blocks[i].rho_param_offset != b.blocks[i].rho_param_offset)
            return false;
    }
    return true;
}
}  // namespace

void GaussNewtonSolver::init(casadi::Function residual, casadi::Function jacobian,
                             const GaussNewtonConfig& config, const SaddleLayout* saddle_layout) {
    // Fast path for repeated solves on the same subproblem. The sparsity maps,
    // saddle matrix pattern and symbolic LDLT analyses are still valid, so only
    // the config needs to be refreshed.
    const bool want_saddle = config.use_saddle && saddle_layout != nullptr;
    if (structure_ready_ && residual.get() == residual_func_.get() &&
        jacobian.get() == jacobian_func_.get() && want_saddle == use_saddle_ &&
        (!want_saddle || sameLayout(*saddle_layout, layout_))) {
        config_ = config;
        return;
    }

    residual_func_ = std::move(residual);
    jacobian_func_ = std::move(jacobian);
    config_ = config;

    const int n_x = static_cast<int>(jacobian_func_.numel_in(0));
    const int n_r = static_cast<int>(residual_func_.numel_out(0));

    // Build the fixed Jacobian sparsity pattern (values filled per iteration).
    const casadi::Sparsity sp = jacobian_func_.sparsity_out(0);
    std::vector<casadi_int> rows, cols;
    sp.get_triplet(rows, cols);

    J_.resize(n_r, n_x);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rows.size());
    for (size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(static_cast<int>(rows[k]), static_cast<int>(cols[k]), 1.0);
    }
    J_.setFromTriplets(triplets.begin(), triplets.end());
    J_.makeCompressed();

    // Map each CasADi nonzero (column-major triplet order) to its slot in the
    // compressed Eigen storage, so value updates are a flat scatter.
    casadi_to_eigen_nz_.resize(rows.size());
    const int* outer = J_.outerIndexPtr();
    const int* inner = J_.innerIndexPtr();
    for (size_t k = 0; k < rows.size(); ++k) {
        const int row = static_cast<int>(rows[k]);
        const int col = static_cast<int>(cols[k]);
        for (int slot = outer[col]; slot < outer[col + 1]; ++slot) {
            if (inner[slot] == row) {
                casadi_to_eigen_nz_[k] = slot;
                break;
            }
        }
    }

    reg_I_.resize(n_x, n_x);
    reg_I_.setIdentity();

    x_dm_ = casadi::DM::zeros(n_x);
    pattern_analyzed_ = false;

    use_saddle_ = false;
    if (config_.use_saddle && saddle_layout != nullptr) {
        initSaddle(*saddle_layout, rows, cols);
    }
    structure_ready_ = true;
}

void GaussNewtonSolver::updateJacobianValues(const casadi::DM& J_casadi) {
    const std::vector<double>& nz = J_casadi.nonzeros();
    double* values = J_.valuePtr();
    for (size_t k = 0; k < nz.size(); ++k) {
        values[casadi_to_eigen_nz_[k]] = nz[k];
    }
}

// Primal saddle backend.

void GaussNewtonSolver::initSaddle(const SaddleLayout& layout,
                                   const std::vector<casadi_int>& rows,
                                   const std::vector<casadi_int>& cols) {
    layout_ = layout;
    n_z_ = layout.n_z;
    n_cost_ = layout.n_cost;
    n_dual_ = layout.n_dual;
    const int N = n_z_ + n_dual_;
    const size_t nnz = rows.size();

    // Per dual row -> penalty rho parameter offset.
    dual_row_rho_off_.assign(n_dual_, -1);
    for (const SaddleBlock& b : layout.blocks) {
        const int d0 = b.row_start - n_cost_;
        for (int i = 0; i < b.count; ++i) dual_row_rho_off_[d0 + i] = b.rho_param_offset;
    }
    // Per CasADi Jacobian nonzero: cost row (-1), or its dual row/rho offset.
    casadi_nz_dual_row_.assign(nnz, -1);
    casadi_nz_rho_off_.assign(nnz, -1);
    for (size_t k = 0; k < nnz; ++k) {
        const int r = static_cast<int>(rows[k]);
        if (r >= n_cost_) {
            const int d = r - n_cost_;
            casadi_nz_dual_row_[k] = d;
            casadi_nz_rho_off_[k] = dual_row_rho_off_[d];
        }
    }

    // Constant cost Jacobian (cost residual is linear in z -> values are constant).
    casadi::DM z0 = casadi::DM::zeros(n_z_);
    casadi::DM p0 = casadi::DM::zeros(static_cast<casadi_int>(jacobian_func_.numel_in(1)));
    const std::vector<double> Jc = jacobian_func_(std::vector<casadi::DM>{z0, p0})[0].nonzeros();
    std::vector<Eigen::Triplet<double>> jcost_trip;
    for (size_t k = 0; k < nnz; ++k) {
        const int r = static_cast<int>(rows[k]);
        if (r < n_cost_) jcost_trip.emplace_back(r, static_cast<int>(cols[k]), Jc[k]);
    }
    J_cost_.resize(n_cost_, n_z_);
    J_cost_.setFromTriplets(jcost_trip.begin(), jcost_trip.end());
    J_cost_.makeCompressed();

    // H_cost = J_cost^T J_cost (constant), lower triangle only.
    Eigen::SparseMatrix<double> Hcost =
        (J_cost_.transpose() * J_cost_).triangularView<Eigen::Lower>();
    Hcost.makeCompressed();

    // Assemble the lower-triangular saddle pattern once:
    //   (1,1) H_cost lower + forced full diagonal | (2,1) C block | (2,2) diagonal.
    std::vector<Eigen::Triplet<double>> trip;
    for (int c = 0; c < Hcost.outerSize(); ++c)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Hcost, c); it; ++it)
            trip.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
    for (int i = 0; i < n_z_; ++i) trip.emplace_back(i, i, 0.0);  // ensure (1,1) diagonal exists
    for (size_t k = 0; k < nnz; ++k) {
        const int r = static_cast<int>(rows[k]);
        if (r >= n_cost_)
            trip.emplace_back(n_z_ + (r - n_cost_), static_cast<int>(cols[k]), 0.0);  // C block
    }
    for (int d = 0; d < n_dual_; ++d) trip.emplace_back(n_z_ + d, n_z_ + d, 0.0);  // (2,2) diagonal

    M_.resize(N, N);
    M_.setFromTriplets(trip.begin(), trip.end());  // sums duplicate (i,i) -> H_cost diag + 0
    M_.makeCompressed();

    // Build value-update maps into M_'s compressed storage.
    casadi_nz_to_M_C_.assign(nnz, -1);
    for (size_t k = 0; k < nnz; ++k) {
        const int r = static_cast<int>(rows[k]);
        if (r >= n_cost_)
            casadi_nz_to_M_C_[k] = findSlot(M_, n_z_ + (r - n_cost_), static_cast<int>(cols[k]));
    }
    M_diag1_slot_.assign(n_z_, -1);
    H_cost_diag_.assign(n_z_, 0.0);
    for (int i = 0; i < n_z_; ++i) {
        const int s = findSlot(M_, i, i);
        M_diag1_slot_[i] = s;
        H_cost_diag_[i] = M_.valuePtr()[s];  // constant H_cost(i,i)
    }
    M_diag2_slot_.assign(n_dual_, -1);
    for (int d = 0; d < n_dual_; ++d) M_diag2_slot_[d] = findSlot(M_, n_z_ + d, n_z_ + d);
    saddle_dual_scale_.assign(n_dual_, 1.0);
    saddle_C_row_norm_sq_.assign(n_dual_, 0.0);

    saddle_pattern_analyzed_ = false;
    use_saddle_ = true;
}

void GaussNewtonSolver::updateSaddleValues(const casadi::DM& J_casadi,
                                           const std::vector<double>& p) {
    const std::vector<double>& nz = J_casadi.nonzeros();
    double* val = M_.valuePtr();

    std::fill(saddle_C_row_norm_sq_.begin(), saddle_C_row_norm_sq_.end(), 0.0);

    // C block before equilibration: C_b = J_b / sqrt(rho_b). The active-set mask
    // is already baked into J, so inactive inequality rows get zero values here.
    for (size_t k = 0; k < nz.size(); ++k) {
        const int slot = casadi_nz_to_M_C_[k];
        if (slot < 0) continue;  // cost row -> folded into the constant H_cost
        const int d = casadi_nz_dual_row_[k];
        const double c = nz[k] / std::sqrt(p[casadi_nz_rho_off_[k]]);
        val[slot] = c;
        saddle_C_row_norm_sq_[d] += c * c;
    }

    // Symmetric congruence scaling S M S with S = diag(I, s_d). In exact
    // arithmetic this preserves dz, keeps the sparsity pattern fixed, and avoids
    // tiny isolated -1/rho pivots for inactive or weak dual rows.
    for (int d = 0; d < n_dual_; ++d) {
        const double inv_rho = 1.0 / p[dual_row_rho_off_[d]];
        double s = 1.0;
        if (config_.saddle_equilibrate_dual) {
            const double row_scale = std::sqrt(saddle_C_row_norm_sq_[d] + inv_rho);
            s = (row_scale > 0.0) ? std::max(1.0, 1.0 / row_scale) : 1.0;
        }
        saddle_dual_scale_[d] = s;
        val[M_diag2_slot_[d]] = -(s * s) * inv_rho;
    }
    for (size_t k = 0; k < nz.size(); ++k) {
        const int slot = casadi_nz_to_M_C_[k];
        if (slot < 0) continue;
        val[slot] *= saddle_dual_scale_[casadi_nz_dual_row_[k]];
    }
}

void GaussNewtonSolver::buildSaddleRhs(const std::vector<double>& p, Eigen::VectorXd& rhs) const {
    // RHS = [ -J_cost^T r_cost ; -chat ], chat_d = r_pen_d / sqrt(rho_d). It is
    // independent of the damping lambda, so retries in the same GN iteration share it.
    rhs.resize(n_z_ + n_dual_);
    rhs.head(n_z_).noalias() = -(J_cost_.transpose() * r_curr_.head(n_cost_));
    for (int d = 0; d < n_dual_; ++d)
        rhs[n_z_ + d] =
            -saddle_dual_scale_[d] * r_curr_[n_cost_ + d] / std::sqrt(p[dual_row_rho_off_[d]]);
}

bool GaussNewtonSolver::solveSaddleStep(double lambda, const Eigen::VectorXd& rhs,
                                        Eigen::VectorXd& dx) {
    double* val = M_.valuePtr();
    // (1,1) diagonal = H_cost(i,i) + lambda + sigma_primal (strictly PD).
    const double diag_add = lambda + config_.sigma_primal;
    for (int i = 0; i < n_z_; ++i) val[M_diag1_slot_[i]] = H_cost_diag_[i] + diag_add;

    if (!saddle_pattern_analyzed_) {
        ldlt_saddle_.analyzePattern(M_);
        saddle_pattern_analyzed_ = true;
    }
    ldlt_saddle_.factorize(M_);
    if (ldlt_saddle_.info() != Eigen::Success) return false;

    Eigen::VectorXd sol = ldlt_saddle_.solve(rhs);
    if (ldlt_saddle_.info() != Eigen::Success) return false;
    for (int it = 0; it < config_.saddle_refinement_steps; ++it) {
        const Eigen::VectorXd residual = rhs - M_ * sol;
        if (residual.norm() <= 1e-10 * std::max(1.0, rhs.norm())) break;
        const Eigen::VectorXd correction = ldlt_saddle_.solve(residual);
        if (ldlt_saddle_.info() != Eigen::Success || !correction.allFinite()) return false;
        sol += correction;
    }
    if (!sol.allFinite()) return false;
    dx = sol.head(n_z_);
    return true;
}

// Solver loop.

double GaussNewtonSolver::minimize(const casadi::DM& params, Eigen::VectorXd& x, double grad_tol) {
    const int n_x = static_cast<int>(x.size());
    const double gtol = (grad_tol > 0.0) ? grad_tol : config_.grad_tol;
    std::vector<double>& x_buf = x_dm_.nonzeros();
    const std::vector<double>& p = params.nonzeros();

    // Evaluate residual at x_eval, writing it into r_out; returns ||r||^2.
    // Leaves x_buf holding x_eval, so the Jacobian can be taken at the same point.
    auto residual_at = [&](const Eigen::VectorXd& x_eval, Eigen::VectorXd& r_out) -> double {
        std::copy(x_eval.data(), x_eval.data() + n_x, x_buf.begin());
        casadi::DM r = residual_func_(std::vector<casadi::DM>{x_dm_, params})[0];
        r_out = Eigen::Map<const Eigen::VectorXd>(r.ptr(), r.numel());
        return r_out.squaredNorm();
    };

    // Refresh cached matrices from the CasADi Jacobian at x_buf: the full J_ for
    // the normal path and M_'s C/dual blocks for the saddle path.
    auto refresh_jacobian = [&]() {
        const casadi::DM J_casadi = jacobian_func_(std::vector<casadi::DM>{x_dm_, params})[0];
        updateJacobianValues(J_casadi);
        if (use_saddle_) updateSaddleValues(J_casadi, p);
    };

    // Residual and Jacobian at the starting point.
    double f_val = residual_at(x, r_curr_);
    refresh_jacobian();

    // Damping always starts small; a hard previous solve never leaks in.
    double lambda = config_.lambda_init;
    double nu = 2.0;  // Geometric raise factor on rejection (Nielsen).
    Eigen::VectorXd r_trial;

    // X-update failure tracking for reporting; it does not change control flow.
    last_x_update_failed_ = false;
    bool had_la_failure = false;  // some LM attempt's factorisation reported info != Success
    bool any_accept = false;      // at least one LM step was accepted this call

    for (int iter = 0; iter < config_.max_iter; ++iter) {
        const Eigen::VectorXd grad = 2.0 * (J_.transpose() * r_curr_);  // = grad f
        const double grad_inf = grad.lpNorm<Eigen::Infinity>();
        if (config_.print_level >= 1) {
            std::cout << "    GN " << iter << ": f=" << f_val << ", ||grad||_inf=" << grad_inf
                      << ", lambda=" << lambda << (use_saddle_ ? " [saddle]" : "") << std::endl;
        }
        if (grad_inf < gtol) break;

        bool accepted = false;
        double step_norm = 0.0;
        bool la_failed_this_iter = false;  // LDLT/saddle factorisation failed this GN iteration

        if (use_saddle_) {
            // Primal saddle step: conditioning is much less sensitive to large rho.
            buildSaddleRhs(p, saddle_rhs_);  // lambda-independent, shared across retries
            for (int attempt = 0; attempt < config_.max_damping_tries; ++attempt) {
                Eigen::VectorXd dx;
                if (!solveSaddleStep(lambda, saddle_rhs_, dx)) {
                    had_la_failure = true;
                    la_failed_this_iter = true;
                    lambda = std::min(lambda * nu, config_.lambda_max);
                    nu *= 2.0;
                    continue;
                }
                const Eigen::VectorXd x_new = x + dx;
                const double f_new = residual_at(x_new, r_trial);  // x_buf now holds x_new

                // Predicted reduction of the GN model ||r + J dx||^2, computed from
                // J dx (no explicit J^T J): f - m(dx) = -2 r.(J dx) - ||J dx||^2.
                const Eigen::VectorXd Jdx = J_ * dx;
                const double predicted = -2.0 * r_curr_.dot(Jdx) - Jdx.squaredNorm();
                const double gain = (predicted > 0.0) ? (f_val - f_new) / predicted : -1.0;

                if (isFiniteValue(f_new) && f_new < f_val && gain > 0.0) {
                    x = x_new;
                    step_norm = dx.norm();
                    r_curr_.swap(r_trial);
                    refresh_jacobian();  // x_buf already holds x_new
                    f_val = f_new;
                    const double shrink =
                        std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * gain - 1.0, 3.0));
                    lambda = std::max(lambda * shrink, config_.lambda_min);
                    nu = 2.0;
                    accepted = true;
                    break;
                }
                lambda = std::min(lambda * nu, config_.lambda_max);
                nu *= 2.0;
            }
        } else {
            // Classical normal-equations step: (2 J^T J + lambda I).
            const Eigen::SparseMatrix<double> JtJ = J_.transpose() * J_;

            // Across damping retries only lambda changes, so build H once per GN
            // iteration and, on rejection, shift just its diagonal instead of
            // re-forming the sum. reg_I_ makes every diagonal entry structurally
            // present, so the coeffRef updates never touch the (analyse-once) pattern.
            Eigen::SparseMatrix<double> H = 2.0 * JtJ + lambda * reg_I_;
            double lambda_in_H = lambda;
            if (!pattern_analyzed_) {
                ldlt_.analyzePattern(H);
                pattern_analyzed_ = true;
            }

            for (int attempt = 0; attempt < config_.max_damping_tries; ++attempt) {
                if (lambda != lambda_in_H) {
                    const double d_lambda = lambda - lambda_in_H;
                    for (int k = 0; k < n_x; ++k) H.coeffRef(k, k) += d_lambda;
                    lambda_in_H = lambda;
                }
                ldlt_.factorize(H);
                if (ldlt_.info() != Eigen::Success) {
                    had_la_failure = true;
                    la_failed_this_iter = true;
                    lambda = std::min(lambda * nu, config_.lambda_max);
                    nu *= 2.0;
                    continue;
                }

                const Eigen::VectorXd dx = ldlt_.solve(-grad);
                const Eigen::VectorXd x_new = x + dx;
                const double f_new = residual_at(x_new, r_trial);  // x_buf now holds x_new

                // Predicted reduction of the GN model m(dx) = ||r + J dx||^2:
                //   f - m(dx) = -(grad . dx) - dx^T (J^T J) dx
                const double predicted = -grad.dot(dx) - dx.dot(JtJ * dx);
                const double gain = (predicted > 0.0) ? (f_val - f_new) / predicted : -1.0;

                if (isFiniteValue(f_new) && f_new < f_val && gain > 0.0) {
                    // Accept: reuse the trial residual (no re-eval), refresh the
                    // Jacobian (x_buf already holds x_new), lower the damping.
                    x = x_new;
                    step_norm = dx.norm();
                    r_curr_.swap(r_trial);
                    refresh_jacobian();
                    f_val = f_new;
                    const double shrink =
                        std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * gain - 1.0, 3.0));
                    lambda = std::max(lambda * shrink, config_.lambda_min);
                    nu = 2.0;
                    accepted = true;
                    break;
                }

                // Reject (non-finite or no decrease): increase damping, reuse J/grad.
                lambda = std::min(lambda * nu, config_.lambda_max);
                nu *= 2.0;
            }
        }

        // Steepest-descent fallback for a failed LM factorisation. If every LM
        // retry fails or is rejected, take a backtracking step along -grad so the
        // X-update can still make progress before trying LM again.
        if (!accepted && la_failed_this_iter && grad_inf >= gtol) {
            const double gg = grad.squaredNorm();
            const Eigen::VectorXd Jg = J_ * grad;        // sparse matvec
            const double curv = 2.0 * Jg.squaredNorm();  // = g^T (2 J^T J) g >= 0
            double alpha = (curv > 0.0) ? gg / curv : 1.0 / (1.0 + grad_inf);
            if (!isFiniteValue(alpha) || alpha <= 0.0) alpha = 1.0 / (1.0 + grad_inf);
            for (int ls = 0; ls < config_.max_damping_tries; ++ls) {
                const double f_new = residual_at(x - alpha * grad, r_trial);  // x_buf <- x_new
                if (isFiniteValue(f_new) && f_new < f_val - 1e-4 * alpha * gg) {  // Armijo
                    x -= alpha * grad;
                    step_norm = alpha * std::sqrt(gg);
                    r_curr_.swap(r_trial);
                    refresh_jacobian();  // x_buf already holds x_new
                    f_val = f_new;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
        }

        any_accept |= accepted;
        if (!accepted) break;  // No descent even along -grad (e.g. non-finite Jacobian).
        if (step_norm < config_.step_tol) break;
    }

    // Treat the X-update as stuck only when a factorisation failed and no step was
    // accepted. Transient failures recovered by damping are not terminal.
    last_x_update_failed_ = had_la_failure && !any_accept;
    return f_val;
}

}  // namespace impact
