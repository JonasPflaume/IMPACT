#pragma once

#include <Eigen/Core>

namespace impact {

/**
 * @brief Configuration shared by the BCD-AuLa solver and its builders.
 *
 * The per-type penalty, scale and tolerance fields initialize the dual blocks
 * created by a shooting builder. Builders ignore fields for constraint types
 * their transcription does not contain.
 */
struct BCDAULAConfig {
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

    // Inner BCD loop.
    int max_inner_iters = 50;
    double inner_tol_init = 1e-2;   // Stagnation tolerance, early outer iters
    double inner_tol_final = 1e-3;  // Stagnation tolerance, later outer iters

    bool check_stationarity = false;
    bool conditioned_complementarity = false;
    double stationarity_tol = 1e-5;
    int max_stagnation_restarts = 0;

    // Solve the GN X-step loosely while feasibility is far off, then tighten as
    // the outer loop improves.
    bool use_forcing_sequence = true;

    // Gauss-Newton X-update.
    int newton_max_iter = 50;
    double newton_tol = 1e-6;             // Gradient inf-norm tolerance
    double newton_step_tol = 1e-8;        // Step-norm tolerance
    double newton_regularization = 1e-5;  // Levenberg-Marquardt damping

    // Compile CasADi residual/Jacobian functions through a C compiler instead of
    // interpreting them. Useful for repeated solves; off by default.
    bool jit = false;

    // X-update linear-algebra backend (see SaddleLayout). The saddle form is the
    // default because it is much better conditioned than the normal equations on
    // dense single-shooting problems.
    bool use_saddle = true;
    double saddle_sigma_primal = 1e-8;    // primal proximal floor on the (1,1) block
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
