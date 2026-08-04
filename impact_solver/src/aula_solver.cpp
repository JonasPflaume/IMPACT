#include "impact/aula_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "impact/complementarity_projection.h"

namespace impact {

namespace {

// Per-block safeguarded penalty update. rho is raised only when a constraint is
// still above tolerance and did not decrease enough since the previous outer
// iteration.
double safeguardedPenalty(double rho, double v_new, double v_old, double tol, double eta,
                          double gamma, double rho_max) {
    const bool violating = v_new >= tol;
    const bool insufficient_decrease = v_new > eta * v_old;  // false at k=0 (v_old = +inf)
    if (violating && insufficient_decrease) return std::min(rho * gamma, rho_max);
    return rho;
}

double infNorm(const Eigen::VectorXd& v) { return v.size() ? v.lpNorm<Eigen::Infinity>() : 0.0; }

int totalCompDim(const std::vector<CompBlock>& comps) {
    int n = 0;
    for (const CompBlock& c : comps) n += c.dim;
    return n;
}

}  // namespace

AulaResult AulaSolver::solve(AulaSubproblem& sub, const AulaConfig& config,
                                   const Eigen::VectorXd& z_init) {
    // steady_clock, not high_resolution_clock: libstdc++ aliases the latter to
    // system_clock, which is not monotonic, so an NTP step mid-solve can make the
    // reported solve_time jump or go negative.
    const auto t_start = std::chrono::steady_clock::now();

    if (z_init.size() != sub.numOpt() || !z_init.allFinite())
        throw std::invalid_argument("AulaSolver::solve: invalid initial decision vector");
    if (config.max_outer_iters <= 0 || config.max_inner_iters <= 0 ||
        config.newton_max_iter <= 0)
        throw std::invalid_argument("AulaSolver::solve: iteration limits must be positive");
    if (!(config.rho_scale > 1.0) || !(config.rho_max > 0.0) ||
        !(config.penalty_decrease_ratio >= 0.0 && config.penalty_decrease_ratio <= 1.0))
        throw std::invalid_argument("AulaSolver::solve: invalid penalty configuration");

    // Bind the Gauss-Newton X-solver to this subproblem.
    GaussNewtonConfig gn_cfg;
    gn_cfg.max_iter = config.newton_max_iter;
    gn_cfg.grad_tol = config.newton_tol;
    gn_cfg.step_tol = config.newton_step_tol;
    gn_cfg.lambda_init = config.newton_regularization;
    gn_cfg.lambda_min = config.newton_lambda_min;
    gn_cfg.lambda_max = config.newton_lambda_max;
    gn_cfg.max_damping_tries = config.newton_max_damping_tries;
    gn_cfg.print_level = (config.print_level >= 3) ? 1 : 0;
    gn_cfg.use_saddle = config.use_saddle;
    gn_cfg.sigma_primal = config.saddle_sigma_primal;
    gn_cfg.saddle_equilibrate_dual = config.saddle_equilibrate_dual;
    gn_cfg.saddle_refinement_steps = config.saddle_refinement_steps;
    const SaddleLayout layout = sub.saddleLayout();
    gn_.init(sub.residualFunction(), sub.jacobianFunction(), gn_cfg,
             config.use_saddle ? &layout : nullptr);
    // The solver instance is reused across MPC steps, so the timers have to be
    // zeroed per solve rather than per construction.
    gn_.resetTimers();

    Eigen::VectorXd z = z_init;

    std::vector<DualBlock>& blocks = sub.dualBlocks();
    std::vector<CompBlock>& comps = sub.compBlocks();
    const int n_comp = totalCompDim(comps);

    // Previous violations for safeguarded penalty updates and the forcing sequence.
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> v_prev(blocks.size(), INF);
    std::vector<double> v_comp_prev(comps.size(), INF);
    double outer_viol_prev = 1e3;

    const double s = config.safeguard_factor;
    const double eta = config.penalty_decrease_ratio;
    const double gamma = config.rho_scale;

    Eigen::VectorXd G(n_comp), H(n_comp);

    // Balanced initial penalties (Birgin-Martinez): pick each block's rho_0 so that
    // at z_init the penalty term rho * ||c||^2 matches the objective's magnitude,
    // instead of trusting the build-time rho_*_init seeds. The balance is only
    // informative when the block starts measurably infeasible (||c||^2 > 1); a
    // (near-)feasible block carries no scale information at z_init, so it starts at
    // the geometric midpoint of the clip window (the scale-free neutral choice) and
    // is balanced later, at the first outer iteration that gives it a measurable
    // residual -- see the deferred pass at the end of the outer loop. Without that
    // split, a feasible cold start (e.g. constant-state init) drives the guarded
    // formula 2 max(1,|f|) / max(1, ||c||^2) to the clip ceiling, and a penalty
    // that stiff freezes the first subproblems onto the feasible manifold before
    // the objective has moved anywhere. rho_init is overwritten alongside rho, so
    // stagnation restarts reset to the same balanced value.
    std::vector<char> rho_balanced;       // empty unless auto_rho_init
    std::vector<char> rho_comp_balanced;  // empty unless auto_rho_init
    // Everything below is in EFFECTIVE-stiffness units. The penalty a block exerts
    // goes as rho * scale^2 (its rows are sqrt(rho) * scale * c), so a nominal-rho
    // clip window silently under-enforces any block with scale << 1: at
    // comp_scale = 0.005 a nominal ceiling of 1e2 is an effective ceiling of
    // 2.5e-3, the complementarity set stays relaxed forever, and the optimizer
    // exploits it with mode-averaged chattering plans. Balancing, clipping and
    // capping rho_eff = rho * scale^2 makes the whole scheme invariant under the
    // conditioning scales, which then do the only job they should: scaling rows.
    auto sq = [](double s) { return s > 0.0 ? s * s : 1.0; };
    auto balancedRhoEff = [&](double d2u, double f_abs) {
        return std::clamp(2.0 * std::max(1.0, f_abs) / d2u, config.auto_rho_clip_min,
                          config.auto_rho_clip_max);
    };
    // Unscaled squared norm of a comp block's split residual at the current slacks;
    // at kappa = 0, sG/sH from the closed-form projection make this exactly the
    // squared distance of (G, H) to the complementarity set.
    auto compResidual2 = [](const CompBlock& comp, const Eigen::VectorXd& Gv,
                            const Eigen::VectorXd& Hv, int off) {
        double d2 = 0.0;
        for (int i = 0; i < comp.dim; ++i) {
            const double rg = Gv(off + i) - comp.sG(i);
            const double rh = Hv(off + i) - comp.sH(i);
            d2 += rg * rg + rh * rh;
        }
        return d2;
    };
    if (config.auto_rho_init) {
        const double f0 = std::abs(sub.evalTaskObjective(z));
        rho_balanced.assign(blocks.size(), 1);
        rho_comp_balanced.assign(comps.size(), 1);
        for (size_t i = 0; i < blocks.size(); ++i) {
            DualBlock& b = blocks[i];
            if (b.dim == 0) continue;
            Eigen::VectorXd c = b.eval_scaled(z);
            if (b.kind == DualKind::Inequality) c = c.cwiseMax(0.0);
            const double d2u = c.squaredNorm() / sq(b.scale);  // unscaled units
            rho_balanced[i] = d2u > 1.0;
            // A block below the unit residual carries no scale information; the
            // build-time seed is the only prior there is, and it holds exactly
            // until the deferred pass below sees a measurable residual.
            b.rho = rho_balanced[i] ? balancedRhoEff(d2u, f0) / sq(b.scale) : b.rho_init;
        }
        if (n_comp > 0) {
            sub.evalGH(z, G, H);
            int off = 0;
            for (size_t ci = 0; ci < comps.size(); ++ci) {
                CompBlock& comp = comps[ci];
                if (comp.dim <= 0) continue;
                // Distance^2 of (G, H) to the complementarity set, per pair the
                // smaller branch cost from projectComplementarity. Needs no rho,
                // which breaks the projection -> rho_0 -> projection cycle.
                double d2u = 0.0;
                for (int i = 0; i < comp.dim; ++i) {
                    const double aG = G(off + i);
                    const double aH = H(off + i);
                    const double nG = std::min(aG, 0.0), nH = std::min(aH, 0.0);
                    d2u += std::min(nG * nG + aH * aH, aG * aG + nH * nH);
                }
                rho_comp_balanced[ci] = d2u > 1.0;
                comp.rho = rho_comp_balanced[ci]
                               ? balancedRhoEff(d2u, f0) / sq(comp.scale)
                               : comp.rho_init;
                off += comp.dim;
            }
        }
    }

    if (config.print_level >= 1) {
        std::cout << std::string(60, '=') << "\nBCD-AuLa (IMPACT) Solver\n"
                  << std::string(60, '=') << "\nVariables: " << sub.numOpt()
                  << ", complementarity pairs: " << n_comp;
        for (const DualBlock& b : blocks)
            if (b.dim > 0) std::cout << ", " << b.name << ": " << b.dim;
        std::cout << std::endl;
    }

    bool converged = false;
    bool any_la_failure = false;  // a stuck X-update occurred at some inner iteration
    const bool check_stationarity = sub.checkStationarity();
    const bool conditioned_comp = sub.conditionedComplementarity();
    int stagnant_outer = 0;
    int stagnation_restarts = 0;
    int outer_iter = 0;
    int total_inner = 0;
    int total_gn = 0;

    for (outer_iter = 0; outer_iter < config.max_outer_iters; ++outer_iter) {
        const Eigen::VectorXd z_outer_start = z;
        // Clip multipliers before the inner solve.
        for (DualBlock& b : blocks) {
            if (b.dim == 0) continue;
            if (b.kind == DualKind::Inequality)
                b.kappa = b.kappa.cwiseMax(0.0).cwiseMin(s);
            else
                b.kappa = b.kappa.cwiseMax(-s).cwiseMin(s);
        }
        for (CompBlock& comp : comps) {
            if (comp.dim <= 0) continue;
            comp.kappaG = comp.kappaG.cwiseMax(-s).cwiseMin(s);
            comp.kappaH = comp.kappaH.cwiseMax(-s).cwiseMin(s);
        }
        sub.syncParams();

        const double inner_tol =
            (outer_iter < config.inner_tol_ramp_start)
                ? config.inner_tol_init
                : (outer_iter < config.inner_tol_ramp_end)
                      ? 0.5 * (config.inner_tol_init + config.inner_tol_final)
                      : config.inner_tol_final;

        // Inner BCD loop: alternate the GN X-update and the closed-form slack update.
        // The parameter buffer is synced before the loop and after every slack
        // update, so each X-update always sees the current slacks/multipliers.
        // phi_prev tracks Φ(w^(j)) at the previous completed sweep (initially Φ(w^(0))
        // under the freshly safeguarded multipliers), so the stagnation test below
        // measures the FULL sweep decrease Φ(w^(j)) − Φ(w^(j+1)) of paper Alg. 2
        // (X-update + slack update), not just the slack-block part.
        double phi_prev = sub.evalAugmentedObjective(z);
        for (int inner = 0; inner < config.max_inner_iters; ++inner) {
            ++total_inner;

            const double gn_tol =
                config.use_forcing_sequence
                    ? std::min(config.forcing_cap,
                               std::max(config.newton_tol,
                                        config.forcing_factor * outer_viol_prev))
                    : -1.0;
            gn_.minimize(sub.params(), z, gn_tol);  // X-update
            any_la_failure |= gn_.lastXUpdateFailed();
            total_gn += gn_.lastIterations();

            if (n_comp > 0) {
                sub.evalGH(z, G, H);  // (Y,Z)-update
                if (comps.size() == 1) {
                    CompBlock& comp = comps.front();
                    projectComplementarity(G, H, comp.kappaG, comp.kappaH, comp.rho, comp.scale,
                                           comp.sG, comp.sH);
                } else {
                    int off = 0;
                    for (CompBlock& comp : comps) {
                        if (comp.dim <= 0) continue;
                        const Eigen::VectorXd Gb = G.segment(off, comp.dim);
                        const Eigen::VectorXd Hb = H.segment(off, comp.dim);
                        projectComplementarity(Gb, Hb, comp.kappaG, comp.kappaH, comp.rho,
                                               comp.scale, comp.sG, comp.sH);
                        off += comp.dim;
                    }
                }
                sub.syncParams();
            }
            const double phi_S = sub.evalAugmentedObjective(z);
            if (std::abs(phi_prev - phi_S) < inner_tol) break;
            phi_prev = phi_S;
        }

        // Dual ascent and scaled violation measurement for each standard block.
        std::vector<double> v_scaled(blocks.size(), 0.0);
        for (size_t i = 0; i < blocks.size(); ++i) {
            DualBlock& b = blocks[i];
            if (b.dim == 0) continue;
            const Eigen::VectorXd c = b.eval_scaled(z);  // already scaled
            if (b.kind == DualKind::Inequality) {
                b.kappa = (b.kappa + b.rho * c).cwiseMax(0.0);
                v_scaled[i] = c.cwiseMax(0.0).lpNorm<Eigen::Infinity>();
            } else {
                b.kappa += b.rho * c;
                v_scaled[i] = infNorm(c);
            }
        }

        // Complementarity dual ascent on the split G/H residuals.
        double comp_prod = 0.0, comp_termination = 0.0, v_comp_scaled = 0.0;
        std::vector<double> comp_prod_blocks(comps.size(), 0.0);
        std::vector<double> comp_termination_blocks(comps.size(), 0.0);
        std::vector<double> v_comp_scaled_blocks(comps.size(), 0.0);
        if (n_comp > 0) {
            if (comps.size() == 1) {
                CompBlock& comp = comps.front();
                const Eigen::VectorXd comp_res_G = comp.scale * (G - comp.sG);
                const Eigen::VectorXd comp_res_H = comp.scale * (H - comp.sH);
                comp.kappaG += comp.rho * comp_res_G;
                comp.kappaH += comp.rho * comp_res_H;
                comp_prod = (G.cwiseProduct(H)).lpNorm<Eigen::Infinity>();
                v_comp_scaled = std::max(infNorm(comp_res_G), infNorm(comp_res_H));
                comp_prod_blocks[0] = comp_prod;
                comp_termination_blocks[0] =
                    conditioned_comp ? comp.scale * comp.scale * comp_prod : comp_prod;
                comp_termination = comp_termination_blocks[0];
                v_comp_scaled_blocks[0] = v_comp_scaled;
            } else {
                int off = 0;
                for (size_t ci = 0; ci < comps.size(); ++ci) {
                    CompBlock& comp = comps[ci];
                    if (comp.dim <= 0) continue;
                    const Eigen::VectorXd Gb = G.segment(off, comp.dim);
                    const Eigen::VectorXd Hb = H.segment(off, comp.dim);
                    const Eigen::VectorXd comp_res_G = comp.scale * (Gb - comp.sG);
                    const Eigen::VectorXd comp_res_H = comp.scale * (Hb - comp.sH);
                    comp.kappaG += comp.rho * comp_res_G;
                    comp.kappaH += comp.rho * comp_res_H;
                    comp_prod_blocks[ci] = (Gb.cwiseProduct(Hb)).lpNorm<Eigen::Infinity>();
                    comp_prod = std::max(comp_prod, comp_prod_blocks[ci]);
                    comp_termination_blocks[ci] =
                        conditioned_comp ? comp.scale * comp.scale * comp_prod_blocks[ci]
                                         : comp_prod_blocks[ci];
                    comp_termination =
                        std::max(comp_termination, comp_termination_blocks[ci]);
                    v_comp_scaled_blocks[ci] = std::max(infNorm(comp_res_G), infNorm(comp_res_H));
                    v_comp_scaled = std::max(v_comp_scaled, v_comp_scaled_blocks[ci]);
                    off += comp.dim;
                }
            }
        }

        // Update penalties from unscaled violations. The safeguard uses the same
        // units as the convergence check; otherwise a small conditioning scale can
        // leave rho unchanged while the true violation is still above tolerance.
        // In auto mode the cap is per-block and never below the effective clip
        // ceiling: a scale << 1 block must be allowed to reach rho_eff =
        // auto_rho_clip_max, or rho_max (a nominal number) silently re-imposes the
        // under-enforcement the effective-units scheme exists to remove. max()
        // with rho_max so scale >= 1 blocks keep their configured headroom.
        auto rhoCap = [&](double scale) {
            return config.auto_rho_init
                       ? std::max(config.rho_max, config.auto_rho_clip_max / sq(scale))
                       : config.rho_max;
        };
        for (size_t i = 0; i < blocks.size(); ++i) {
            DualBlock& b = blocks[i];
            if (b.dim == 0) continue;
            const double vu = b.scale > 0.0 ? v_scaled[i] / b.scale : v_scaled[i];
            b.rho = safeguardedPenalty(b.rho, vu, v_prev[i], b.tol, eta, gamma,
                                       rhoCap(b.scale));
            v_prev[i] = vu;
        }
        for (size_t ci = 0; ci < comps.size(); ++ci) {
            CompBlock& comp = comps[ci];
            if (comp.dim <= 0) continue;
            const double vcu = conditioned_comp
                                   ? v_comp_scaled_blocks[ci]
                                   : (comp.scale > 0.0
                                          ? v_comp_scaled_blocks[ci] / comp.scale
                                          : v_comp_scaled_blocks[ci]);
            comp.rho =
                safeguardedPenalty(comp.rho, vcu, v_comp_prev[ci], comp.tol, eta, gamma,
                                   rhoCap(comp.scale));
            v_comp_prev[ci] = vcu;
        }

        // Deferred balancing (auto_rho_init): a block that started (near-)feasible
        // was seeded with the scale-free neutral value; the first outer iteration
        // that hands it a measurable residual supplies the scale the balance formula
        // needs, and the block is balanced once, here. max() keeps any safeguard
        // growth that already happened. Costs one residual eval per still-unbalanced
        // block per outer iteration, and nothing once every block is balanced.
        if (config.auto_rho_init) {
            bool pending = false;
            for (char done : rho_balanced) pending |= !done;
            for (char done : rho_comp_balanced) pending |= !done;
            if (pending) {
                const double f_cur = std::abs(sub.evalTaskObjective(z));
                for (size_t i = 0; i < blocks.size(); ++i) {
                    if (rho_balanced[i]) continue;
                    DualBlock& b = blocks[i];
                    Eigen::VectorXd c = b.eval_scaled(z);
                    if (b.kind == DualKind::Inequality) c = c.cwiseMax(0.0);
                    const double d2u = c.squaredNorm() / sq(b.scale);
                    if (d2u <= 1.0) continue;
                    b.rho = std::max(b.rho, balancedRhoEff(d2u, f_cur) / sq(b.scale));
                    rho_balanced[i] = 1;
                }
                // G and H still hold their values at the current z from the last
                // inner-loop evaluation, and sG/sH the matching projection.
                int off = 0;
                for (size_t ci = 0; ci < comps.size(); ++ci) {
                    const CompBlock& c_ro = comps[ci];
                    if (c_ro.dim <= 0) continue;
                    if (!rho_comp_balanced[ci]) {
                        const double d2u = compResidual2(c_ro, G, H, off);
                        if (d2u > 1.0) {
                            CompBlock& comp = comps[ci];
                            comp.rho = std::max(comp.rho,
                                                balancedRhoEff(d2u, f_cur) / sq(comp.scale));
                            rho_comp_balanced[ci] = 1;
                        }
                    }
                    off += c_ro.dim;
                }
            }
        }

        auto unscale = [](double v, double scale) { return scale > 0.0 ? v / scale : v; };
        bool feasible = true;
        double max_unscaled = 0.0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].dim == 0) continue;
            const double vu = unscale(v_scaled[i], blocks[i].scale);
            max_unscaled = std::max(max_unscaled, vu);
            if (vu >= blocks[i].tol) feasible = false;
        }
        double comp_res_u = 0.0;
        for (size_t ci = 0; ci < comps.size(); ++ci) {
            const CompBlock& comp = comps[ci];
            if (comp.dim <= 0) continue;
            const double block_res_test =
                conditioned_comp ? v_comp_scaled_blocks[ci]
                                 : unscale(v_comp_scaled_blocks[ci], comp.scale);
            comp_res_u = std::max(comp_res_u, block_res_test);
            if (comp_termination_blocks[ci] >= comp.tol || block_res_test >= comp.tol)
                feasible = false;
        }

        outer_viol_prev = std::max({max_unscaled, comp_termination, comp_res_u});

        double stationarity = check_stationarity ? INF : 0.0;
        if (check_stationarity && feasible) {
            sub.syncParams();
            stationarity = sub.evalAugmentedGradientInf(z);
        }
        const bool stationary = !check_stationarity ||
                                (feasible && stationarity < sub.stationarityTol());

        if (config.print_level >= 1) {
            std::cout << std::scientific << std::setprecision(3) << "Outer " << std::setw(4)
                      << outer_iter << ":";
            for (size_t i = 0; i < blocks.size(); ++i)
                if (blocks[i].dim > 0)
                    std::cout << " " << blocks[i].name << "="
                              << unscale(v_scaled[i], blocks[i].scale);
            if (n_comp > 0 && comps.size() == 1) {
                std::cout << " comp=" << comp_prod << " comp_res=" << comp_res_u
                          << " | rho_comp=" << comps.front().rho;
            } else if (n_comp > 0) {
                std::cout << " comp=" << comp_prod << " comp_res=" << comp_res_u;
                for (size_t ci = 0; ci < comps.size(); ++ci) {
                    const CompBlock& comp = comps[ci];
                    if (comp.dim <= 0) continue;
                    std::cout << " " << comp.name << "=(" << comp_prod_blocks[ci] << ","
                              << unscale(v_comp_scaled_blocks[ci], comp.scale)
                              << ",rho=" << comp.rho << ")";
                }
            }
            if (check_stationarity) std::cout << " stationarity=" << stationarity;
            std::cout << std::endl;
        }

        if (feasible && stationary) {
            converged = true;
            break;
        }

        if (stagnation_restarts < sub.maxStagnationRestarts()) {
            const double step_inf = (z - z_outer_start).lpNorm<Eigen::Infinity>();
            const double step_floor = 1e-10 * (1.0 + z.lpNorm<Eigen::Infinity>());
            stagnant_outer = (step_inf <= step_floor) ? stagnant_outer + 1 : 0;
            bool severely_infeasible = false;
            for (size_t i = 0; i < blocks.size(); ++i) {
                if (blocks[i].dim == 0) continue;
                const double vu = unscale(v_scaled[i], blocks[i].scale);
                severely_infeasible |= vu > 10.0 * blocks[i].tol;
            }
            for (size_t ci = 0; ci < comps.size(); ++ci) {
                if (comps[ci].dim == 0) continue;
                severely_infeasible |=
                    comp_termination_blocks[ci] > 10.0 * comps[ci].tol ||
                    v_comp_scaled_blocks[ci] > 10.0 * comps[ci].tol;
            }
            const bool severely_nonstationary = check_stationarity && feasible &&
                                                stationarity > 10.0 * sub.stationarityTol();
            if (stagnant_outer >= 3 &&
                (severely_infeasible || severely_nonstationary)) {
                Eigen::VectorXd direction = -sub.evalTaskGradient(z);
                const double d_inf = direction.size()
                                         ? direction.lpNorm<Eigen::Infinity>()
                                         : 0.0;
                if (!direction.allFinite() || d_inf <= 1e-12) {
                    direction.resize(z.size());
                    for (int i = 0; i < z.size(); ++i)
                        direction(i) = ((i + stagnation_restarts) % 2 == 0) ? 1.0 : -1.0;
                } else {
                    direction /= d_inf;
                }
                for (int i = 0; i < z.size(); ++i)
                    z(i) += 1e-2 * std::max(1.0, std::abs(z(i))) * direction(i);

                for (DualBlock& b : blocks) {
                    b.kappa.setZero();
                    b.rho = b.rho_init;
                }
                if (n_comp > 0) {
                    sub.evalGH(z, G, H);
                    int comp_off = 0;
                    for (CompBlock& comp : comps) {
                        comp.kappaG.setZero();
                        comp.kappaH.setZero();
                        comp.rho = comp.rho_init;
                        const Eigen::VectorXd Gb = G.segment(comp_off, comp.dim);
                        const Eigen::VectorXd Hb = H.segment(comp_off, comp.dim);
                        projectComplementarity(Gb, Hb, comp.kappaG, comp.kappaH,
                                               comp.rho, comp.scale, comp.sG, comp.sH);
                        comp_off += comp.dim;
                    }
                }
                std::fill(v_prev.begin(), v_prev.end(), INF);
                std::fill(v_comp_prev.begin(), v_comp_prev.end(), INF);
                outer_viol_prev = 1e3;
                stagnant_outer = 0;
                ++stagnation_restarts;
                sub.syncParams();
            }
        }
    }

    // Fill the public result fields.
    AulaResult res;
    res.z = z;
    res.converged = converged;
    res.outer_iterations = std::min(outer_iter + 1, config.max_outer_iters);
    res.total_inner_iterations = total_inner;
    res.total_gn_iterations = total_gn;
    res.objective_value = sub.evalTaskObjective(z);
    if (check_stationarity) res.constraint_violations.reserve(blocks.size() + comps.size());
    if (check_stationarity) res.stationarity_violation = sub.evalAugmentedGradientInf(z);

    if (n_comp > 0) {
        sub.evalGH(z, G, H);
        res.complementarity_violation = (G.cwiseProduct(H)).lpNorm<Eigen::Infinity>();
        res.comp_neg_G = infNorm(G.cwiseMin(0.0));
        res.comp_neg_H = infNorm(H.cwiseMin(0.0));
        int comp_off = 0;
        for (const CompBlock& comp : comps) {
            if (comp.dim <= 0) continue;
            const Eigen::VectorXd Gb = G.segment(comp_off, comp.dim);
            const Eigen::VectorXd Hb = H.segment(comp_off, comp.dim);
            res.comp_support_G = std::max(
                res.comp_support_G,
                infNorm(Gb.cwiseMax(0.0).cwiseMin(comp.kappaG.cwiseAbs())));
            res.comp_support_H = std::max(
                res.comp_support_H,
                infNorm(Hb.cwiseMax(0.0).cwiseMin(comp.kappaH.cwiseAbs())));
            comp_off += comp.dim;
        }
    }
    auto unscale = [](double v, double scale) { return scale > 0.0 ? v / scale : v; };
    for (const DualBlock& b : blocks) {
        if (b.dim == 0) continue;
        const Eigen::VectorXd c = b.eval_scaled(z);
        const double vu = unscale(b.kind == DualKind::Inequality
                                      ? c.cwiseMax(0.0).lpNorm<Eigen::Infinity>()
                                      : infNorm(c),
                                  b.scale);
        if (b.name == "dynamics") res.dynamics_violation = vu;
        else if (b.name == "equality") res.equality_violation = vu;
        else if (b.name == "inequality") res.inequality_violation = vu;
        if (check_stationarity) res.constraint_violations.push_back({b.name, vu, vu});
    }
    if (check_stationarity && n_comp > 0) {
        int comp_off = 0;
        for (const CompBlock& comp : comps) {
            const Eigen::VectorXd Gb = G.segment(comp_off, comp.dim);
            const Eigen::VectorXd Hb = H.segment(comp_off, comp.dim);
            const double product = (Gb.cwiseProduct(Hb)).lpNorm<Eigen::Infinity>();
            const double split = std::max(
                infNorm(Gb - comp.sG), infNorm(Hb - comp.sH));
            res.constraint_violations.push_back({comp.name, product, split});
            comp_off += comp.dim;
        }
    }

    res.solve_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    res.eval_time = gn_.evalSeconds();
    res.factor_time = gn_.factorSeconds();
    if (converged) {
        res.status = BCDAULAStatus::Converged;
        res.status_message = "Converged";
    } else if (any_la_failure) {
        res.status = BCDAULAStatus::LinearAlgebraFailure;
        res.status_message = "Linear-algebra failure: X-update factorisation stuck";
    } else {
        res.status = BCDAULAStatus::MaxIterations;
        res.status_message = check_stationarity && std::isfinite(res.stationarity_violation)
                                 ? "Max outer iterations reached (feasibility or stationarity)"
                                 : "Max outer iterations reached";
    }

    if (config.print_level >= 1) {
        std::cout << std::string(60, '=') << "\n"
                  << (converged ? "CONVERGED" : "STOPPED (max iters)") << " in " << res.solve_time
                  << "s, " << res.outer_iterations << " outer / " << total_inner << " inner\n"
                  << std::string(60, '=') << std::endl;
    }
    return res;
}

}  // namespace impact
