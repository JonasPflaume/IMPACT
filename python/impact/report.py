"""Formatting for what a solve did: the planner tag, the settings echo, the
machine-readable ``RESULT`` line.

These live in the library rather than in the CLI because the tag is not
cosmetic -- it names the output directory (``results/<task>/<tag>/``) and goes
into the trajectory file's ``# Planner:`` header, which is what the visualizers
and the sweep scripts key off. A script that solves without going through the
CLI has to produce the same strings.

Two measurement traps are pinned here rather than left to callers, because
getting them wrong produces numbers that look algorithmic but are not:

1. The reported stationarity is the same quantity the inner Gauss-Newton solver
   stops on, so a stationarity target has to drag ``newton_tol`` and the inner
   stagnation tolerance down with it (see :func:`tighten_to_stationarity`).
   Otherwise *both* inner solvers appear to floor near the default, for reasons
   that have nothing to do with either algorithm.
2. Classifying complementarity index sets needs an active-set threshold, and at
   these accuracy levels the threshold would decide the verdict by itself.
   :func:`result_line` therefore reports the tolerance-free support residuals
   from :class:`~impact.AulaResult` instead.
"""

from __future__ import annotations

__all__ = ["planner_tag", "settings_line", "result_line", "tighten_to_stationarity"]


def planner_tag(config, bcd_tag: str = "bcd_aula") -> str:
    """The tag naming what actually ran, used in output paths."""
    return bcd_tag


def settings_line(config) -> str:
    """One line naming the inner solver and the tolerances actually in force."""
    return (f"Inner solver: BCD, "
            f"stat_tol={config.stationarity_tol if config.check_stationarity else 0.0}, "
            f"tol={config.outer_tol_comp}, newton_tol={config.newton_tol}, "
            f"inner_tol_final={config.inner_tol_final}, max_outer={config.max_outer_iters}, "
            f"backend={'saddle' if config.use_saddle else 'normal-equations'}")


def result_line(mode: str, s, goal_err: float = 0.0) -> str:
    """The ``RESULT key=value ...`` line the sweep scripts parse."""
    return (f"RESULT mode={mode} converged={int(s.converged)} "
            f"objective={s.objective_value:.6e} goal_err={goal_err:.6e} "
            f"dynamics={s.dynamics_violation:.6e} "
            f"comp_prod={s.complementarity_violation:.6e} "
            f"neg_G={s.comp_neg_G:.6e} neg_H={s.comp_neg_H:.6e} "
            f"supp_G={s.comp_support_G:.6e} supp_H={s.comp_support_H:.6e} "
            f"stationarity={s.stationarity_violation:.6e} "
            f"outer={s.outer_iterations} inner={s.total_inner_iterations} "
            f"gn={s.total_gn_iterations} time={s.solve_time:.6e}")


def tighten_to_stationarity(config, stat_tol: float):
    """Ask for a stationarity certificate, and move the tolerances it depends on.

    Setting ``stationarity_tol`` alone is the trap in the module docstring: the
    inner solver would still stop at its own default and the certificate would
    measure that default rather than either algorithm. Callers that want the
    inner tolerances left alone should set the fields directly.
    """
    config.check_stationarity = True
    config.stationarity_tol = stat_tol
    config.newton_tol = 0.1 * stat_tol
    config.inner_tol_final = 1e-3 * stat_tol
    config.inner_tol_init = max(config.inner_tol_init, config.inner_tol_final)
    return config
