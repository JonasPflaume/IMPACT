#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <casadi/casadi.hpp>
#include <vector>

#include "impact/saddle_layout.h"

namespace impact {

/**
 * @brief Configuration for the adaptive Levenberg-Marquardt least-squares solver.
 *
 * Damping is adjusted automatically by a gain-ratio (trust-region) rule: a step
 * that reduces the objective is accepted and the damping is lowered; a step that
 * does not is rejected and the damping is raised. The damping always starts from
 * `lambda_init` at the beginning of each minimize() call, so a large value from a
 * previous (harder) solve never carries over into the next one.
 */
struct GaussNewtonConfig {
    int max_iter = 50;          // Maximum outer Gauss-Newton iterations
    double grad_tol = 1e-6;     // Stop when ||2 J^T r||_inf below this
    double step_tol = 1e-8;     // Stop when ||step||_2 below this
    double lambda_init = 1e-5;  // Initial LM damping, reset on every minimize()
    double lambda_min = 1e-12;  // Lower clamp for damping
    double lambda_max = 1e12;   // Upper clamp for damping
    int max_damping_tries = 20;  // Reject/raise-damping attempts per iteration
    int print_level = 0;         // 0 silent, >=1 per-iteration trace

    // Linear-algebra backend (see SaddleLayout). When use_saddle is set and a
    // layout is supplied to init(), each LM step is computed from the primal
    // saddle system instead of explicitly forming the penalty Schur complement
    // rho*C^T*C. sigma_primal is the proximal floor added to the (1,1) block to
    // keep the system strictly quasidefinite.
    bool use_saddle = false;
    double sigma_primal = 1e-8;
    bool saddle_equilibrate_dual = true;  // congruence scaling; preserves dz and sparsity
    int saddle_refinement_steps = 0;      // optional residual-correction solves after saddle LDLT
};

/**
 * @brief Standalone damped Gauss-Newton solver for nonlinear least squares.
 *
 * Minimises  f(x) = || r(x; p) ||^2  for a fixed parameter vector p, where the
 * residual r and its (sparse) Jacobian J = dr/dx are supplied as CasADi
 * functions with signature  (x, p) -> r  and  (x, p) -> J.
 *
 * The Levenberg-Marquardt step solves the damped normal equations
 *
 *     (2 J^T J + lambda * I) dx = -2 J^T r
 *
 * via a sparse LDLT factorisation, and a gain-ratio rule accepts or rejects the
 * step while adapting lambda. The Jacobian sparsity pattern is fixed across
 * calls, so the sparse structure and symbolic factorisation are analysed once.
 *
 * This module knows nothing about the MPCC/AuLa structure: callers bake all
 * multipliers, penalties and slack variables into the parameter vector p.
 */
class GaussNewtonSolver {
   public:
    GaussNewtonSolver() = default;

    /**
     * @brief Bind the residual and Jacobian functions and pre-allocate caches.
     * @param residual  CasADi function (x, p) -> r  (dense residual vector)
     * @param jacobian  CasADi function (x, p) -> J  (sparse Jacobian dr/dx)
     * @param config    Solver hyper-parameters
     */
    void init(casadi::Function residual, casadi::Function jacobian,
              const GaussNewtonConfig& config, const SaddleLayout* saddle_layout = nullptr);

    /**
     * @brief Minimise ||r(x; params)||^2 in place starting from x.
     * @param params  Fixed parameter vector for this solve
     * @param[in,out] x  Initial guess on input, optimum on output
     * @param grad_tol  Gradient inf-norm stop tolerance for this call; if <= 0
     *                  the configured grad_tol is used. Lets the outer loop run
     *                  an inexact (forcing-sequence) inner solve.
     * @return Final objective value f = ||r||^2
     */
    double minimize(const casadi::DM& params, Eigen::VectorXd& x, double grad_tol = -1.0);

    /// True if the most recent minimize() could not produce a single accepted step
    /// because every damped factorisation failed (a genuinely stuck X-update, as
    /// opposed to a transient failure the damping recovered from). Lets the outer
    /// solver distinguish a linear-algebra breakdown from slow convergence.
    bool lastXUpdateFailed() const { return last_x_update_failed_; }

   private:
    bool last_x_update_failed_ = false;
    // True once init() has built the structural caches; enables the same-structure
    // fast path on repeated init() calls (persistent-subproblem MPC solves).
    bool structure_ready_ = false;

    casadi::Function residual_func_;
    casadi::Function jacobian_func_;
    GaussNewtonConfig config_;

    // Fixed Jacobian sparsity, mapped from CasADi's nonzero order to the
    // compressed Eigen sparse storage (computed once in init()).
    std::vector<int> casadi_to_eigen_nz_;
    Eigen::SparseMatrix<double> J_;     // Pre-allocated Jacobian (fixed pattern)
    Eigen::SparseMatrix<double> reg_I_;  // Identity for LM damping
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
    bool pattern_analyzed_ = false;

    // Reusable input buffer for x (mutated in place to avoid reallocation).
    casadi::DM x_dm_;
    // Current residual r(x; p), kept across iterations so accepted steps avoid
    // a redundant residual evaluation.
    Eigen::VectorXd r_curr_;

    // Refresh J_ values from a freshly evaluated CasADi Jacobian (zero pattern change).
    void updateJacobianValues(const casadi::DM& J_casadi);

    // Primal saddle backend, built in init() when use_saddle && layout.
    bool use_saddle_ = false;
    SaddleLayout layout_;
    int n_z_ = 0, n_cost_ = 0, n_dual_ = 0;

    // Saddle matrix M = [[H_cost + (lambda+sigma)I, C^T], [C, -diag(1/rho)]],
    // stored lower-triangular (column-major) for a quasidefinite LDL^T.
    Eigen::SparseMatrix<double> M_;
    Eigen::SparseMatrix<double> J_cost_;  // constant cost Jacobian (n_cost x n_z)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_saddle_;
    bool saddle_pattern_analyzed_ = false;

    // Value-update maps into M_'s compressed storage (computed once in init):
    std::vector<int> casadi_nz_to_M_C_;   // per CasADi Jac nz -> M slot in C block (-1 if cost row)
    std::vector<int> casadi_nz_dual_row_; // per CasADi Jac nz -> dual row (-1 if cost row)
    std::vector<int> casadi_nz_rho_off_;  // per CasADi Jac nz -> rho param offset (-1 if cost row)
    std::vector<int> M_diag1_slot_;       // (1,1) diagonal slots, size n_z
    std::vector<double> H_cost_diag_;     // constant H_cost diagonal, size n_z
    std::vector<int> M_diag2_slot_;       // (2,2) diagonal slots, size n_dual
    std::vector<int> dual_row_rho_off_;   // rho param offset per dual row, size n_dual
    std::vector<double> saddle_dual_scale_;      // dual congruence scale S_d
    std::vector<double> saddle_C_row_norm_sq_;   // row norm of the unscaled C block

    void initSaddle(const SaddleLayout& layout, const std::vector<casadi_int>& rows,
                    const std::vector<casadi_int>& cols);
    // Refresh M_'s C block and -1/rho diagonal from a fresh CasADi Jacobian + params.
    void updateSaddleValues(const casadi::DM& J_casadi, const std::vector<double>& p);
    // Assemble the (lambda-independent) saddle RHS at the current residual/scaling,
    // once per GN iteration; the damping retries reuse it.
    void buildSaddleRhs(const std::vector<double>& p, Eigen::VectorXd& rhs) const;
    // Solve the saddle system for the LM step dx at the given damping; returns false
    // if the factorisation failed. Fills the (1,1) diagonal with H_cost + lambda + sigma.
    bool solveSaddleStep(double lambda, const Eigen::VectorXd& rhs, Eigen::VectorXd& dx);
    Eigen::VectorXd saddle_rhs_;  // per-iteration RHS buffer (avoids reallocation)
};

}  // namespace impact
