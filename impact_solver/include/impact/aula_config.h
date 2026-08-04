#pragma once

#include <Eigen/Core>

namespace impact {

/**
 * @brief Configuration shared by the augmented-Lagrangian solver and its builders.
 *
 * The inner loop is block-coordinate descent: the complementarity slacks are frozen
 * at their closed-form projection while the trajectory variables take a
 * Gauss-Newton step, both inside the AuLa outer loop.
 *
 * The per-type penalty, scale and tolerance fields initialize the dual blocks
 * created by a shooting builder. Builders ignore fields for constraint types
 * their transcription does not contain.
 */
struct AulaConfig {
    // Problem setup.
    int horizon = 50;        // Planning horizon T
    Eigen::VectorXd x_0;     // Initial state (fixed via AuLa equality in multiple shooting)
    Eigen::VectorXd x_goal;  // Target state

    // Cost weights for least-squares stage and terminal terms.
    double stage_cost_weight = 1e-3;        // ||u_t||^2
    double stage_state_cost_weight = 0.0;   // ||x_t - x_goal||^2
    double control_rate_weight = 0.0;       // ||u_{t+1} - u_t||^2
    double final_cost_weight = 100.0;       // ||x_T - x_goal||^2

    // Initial AuLa penalties, grouped by constraint type.
    double rho_fix_point_init = 1.0;
    double rho_dynamics_init = 1.0;
    double rho_eq_init = 1.0;
    double rho_ineq_init = 1.0;  // also seeds single-shooting control bounds (ineq channel)
    double rho_comp_init = 1.0;
    double rho_max = 1e6;                  // Penalty cap
    double rho_scale = 1.1;               // Penalty increase factor gamma > 1
    double penalty_decrease_ratio = 0.5;  // Safeguard ratio eta in [0,1]

    // Balanced initial penalties (Birgin & Martinez, Practical Augmented Lagrangian
    // Methods, sec. 12.1). When on, the rho_*_init seeds above are ignored and each
    // dual block's initial EFFECTIVE penalty rho_eff = rho * scale^2 is computed at
    // z_init as
    //
    //     rho_eff = clip(2 max(1, |f(z0)|) / ||c(z0)||^2, clip_min, clip_max)
    //
    // with c the block's UNSCALED residual (violating part only for inequalities;
    // distance to the complementarity set for comp blocks). A block whose initial
    // residual is below 1 carries no scale information; it keeps its build-time
    // rho_*_init seed -- the only prior there is -- and is balanced at the first
    // outer iteration that hands it a measurable residual. Working in effective units makes the
    // whole scheme invariant under both the objective's scaling and the per-type
    // conditioning scales -- a nominal-rho window under-enforces any block with
    // scale << 1 (rho_max * scale^2 can sit orders below every other channel),
    // which is what produced mode-averaged chattering plans at small comp_scale.
    // In auto mode the safeguarded growth cap is per-block for the same reason:
    // max(rho_max, clip_max / scale^2). The safeguarded update still grows rho,
    // so this changes where the penalty path starts, never where it can end.
    //
    // Off by default, because it is a *replacement* for per-task penalty tuning and
    // the drivers in experiments/ are tuned. Measured A/B over all ten drivers with
    // nothing else changed: every task still converges, but the objective is 3.2x
    // worse in geometric mean and the wall time 1.6x higher, with the worst cases on
    // cart_transporter (33x objective, 14x time). It reaches a solution in fewer
    // outer iterations on several tasks -- it just reaches a worse one.
    //
    // Turn it on for a problem you have *not* tuned: it balances each block's
    // effective penalty against the objective instead of asking for a rho_max and a
    // rho_scale that only a sweep would find.
    bool auto_rho_init = false;
    double auto_rho_clip_min = 1e-2;  // clip floor for rho_eff = rho * scale^2
    double auto_rho_clip_max = 1e2;   // clip ceiling for rho_eff

    // Multiplier safeguard, applied before each inner solve.
    double safeguard_factor = 1e6;  // kappa in [-s, s], mu in [0, s]

    // Per-type constraint conditioning scales.
    double fix_point_scale = 1.0;
    double dynamics_scale = 1.0;
    double eq_scale = 1.0;
    double ineq_scale = 1.0;
    double comp_scale = 1.0;

    // Outer loop.
    int max_outer_iters = 1000;
    double outer_tol_h = 1e-5;     // Equality / dynamics feasibility tolerance
    double outer_tol_g = 1e-5;     // Inequality feasibility tolerance
    double outer_tol_comp = 1e-5;  // Complementarity tolerance

    // Inner loop (BCD sweeps).
    int max_inner_iters = 50;
    double inner_tol_init = 1e-2;   // Stagnation tolerance, early outer iters
    double inner_tol_final = 1e-3;  // Stagnation tolerance, later outer iters
    // Outer iteration at which the stagnation tolerance steps from inner_tol_init to
    // their midpoint, and then to inner_tol_final. The schedule is a ramp, not a
    // cliff: an early subproblem is far from the solution and solving it tightly
    // wastes inner work, while a late one needs the tight tolerance for the outer
    // loop to make progress at all.
    int inner_tol_ramp_start = 3;
    int inner_tol_ramp_end = 8;

    bool check_stationarity = false;
    // Compare rho-conditioned complementarity (scale^2 * G_i H_i) against outer_tol_comp
    // instead of the raw product. NOTE: MPCCDescription::conditioned_complementarity
    // defaults to *true*, so a subproblem built directly through buildMPCC() conditions
    // by default while one built through a shooting builder takes this field's false.
    // The two entry points therefore apply different convergence tests, and both
    // defaults are in active use, so neither is changed here.
    bool conditioned_complementarity = false;
    double stationarity_tol = 1e-5;
    int max_stagnation_restarts = 0;

    // Solve the GN X-step loosely while feasibility is far off, then tighten as
    // the outer loop improves: grad_tol = min(cap, max(newton_tol, factor * viol)).
    bool use_forcing_sequence = true;
    double forcing_cap = 1e-2;     // loosest gradient tolerance the sequence will ask for
    double forcing_factor = 0.1;   // fraction of the previous outer violation

    // Inner Gauss-Newton X-update (see GaussNewtonConfig).
    int newton_max_iter = 50;
    double newton_tol = 1e-6;             // Gradient inf-norm tolerance
    double newton_step_tol = 1e-8;        // Step-norm tolerance
    double newton_regularization = 1e-5;  // Levenberg-Marquardt damping, reset each minimize()
    // Levenberg-Marquardt damping envelope and retry budget. Defaults are
    // GaussNewtonConfig's, so leaving them alone reproduces the solver exactly; they
    // are surfaced here because they are the knobs a caller reaches for when a
    // problem's conditioning sits far from the 1e-5 .. 1e12 window.
    double newton_lambda_min = 1e-12;
    double newton_lambda_max = 1e12;
    int newton_max_damping_tries = 20;

    // Compile CasADi residual/Jacobian functions through a C compiler instead of
    // interpreting them. Useful for repeated solves; off by default.
    bool jit = false;

    // X-update linear-algebra backend (see SaddleLayout). The saddle form is the
    // default because it is much better conditioned than the normal equations on
    // dense single-shooting problems.
    bool use_saddle = true;
    double saddle_sigma_primal = 1e-8;   // primal proximal floor on the (1,1) block
    bool saddle_equilibrate_dual = true;  // diagonal dual-row scaling
    int saddle_refinement_steps = 0;      // optional iterative refinement steps

    // Single-shooting command bounds, handled as an AuLa inequality channel.
    bool use_cmd_bounds = false;
    Eigen::VectorXd cmd_lower;
    Eigen::VectorXd cmd_upper;

    // Initialisation / verbosity.
    bool use_constant_state_init = false;  // false: linear x_0->x_goal interp (multiple shooting)
    int print_level = 1;                   // 0 silent, 1 outer summary, 2 inner, 3 inner GN
};

}  // namespace impact
