"""Planar Push-T manipulation.

    State   x = [px, py, theta]                     T-block pose
    Control u = [cx, cy, lam(8), v(7), w(7)]        body-frame contact point,
                                                    signed friction forces, and
                                                    the |.| slack pair

The T-shape is described by seven face offsets. Each face's signed distance from
the contact point is split as ``v_i - w_i`` with ``v_i, w_i >= 0`` and
``v_i ⊥ w_i``, which makes ``v_i + w_i`` the absolute distance -- the standard
complementarity encoding of ``|.|``, and the reason the control vector carries
14 slacks.

Complementarity (43 pairs):
  * ``v ⊥ w`` (7) -- the absolute-value encoding above;
  * ``lam_mag ⊥ gap`` (8) -- a face carries force only when it is in contact;
  * ``lam_mag[i] ⊥ lam_mag[j]``, i < j (28) -- at most one face active at a time.

Port of ``experiments/push_t/push_t.{h,cpp}``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import casadi as ca
import numpy as np

from impact import AulaConfig
from impact.report import planner_tag
from impact.stage import MPCCProblem

from ..common import Result

__all__ = ["PushT", "config", "solve"]


class PushT(MPCCProblem):
    def __init__(self, mass: float = 0.1, gravity: float = 9.8, friction: float = 0.4,
                 limit_surface_c: float = 0.4, half_side: float = 0.05,
                 limit_surface_dc: float = 2.6429, dt: float = 0.05):
        self.m = mass
        self.g = gravity
        self.mu = friction
        self.c = limit_surface_c
        self.l = half_side
        self.dc = limit_surface_dc
        self.r = 2.8 * half_side  # contact radius constraint
        self.dt = dt

    state_dim = property(lambda self: 3)
    control_dim = property(lambda self: 24)
    comp_dim = property(lambda self: 43)
    eq_dim = property(lambda self: 7)
    ineq_dim = property(lambda self: 4)
    time_step = property(lambda self: self.dt)

    # -- control layout ----------------------------------------------------
    @staticmethod
    def _split(u):
        return u[0], u[1], u[2:10], u[10:17], u[17:24]  # cx, cy, lam(8), v(7), w(7)

    def dynamics(self, x, u):
        constant = 1.0 / (self.mu * self.m * self.g)
        theta = x[2]
        cx, cy, lam, _, _ = self._split(u)
        # lam is 1-indexed in the model: sum_x collects lam2/4/6/8, sum_y lam1/3/5/7.
        sum_x = lam[1] + lam[3] + lam[5] + lam[7]
        sum_y = lam[0] + lam[2] + lam[4] + lam[6]
        cos_th, sin_th = ca.cos(theta), ca.sin(theta)
        dpx = constant * (sum_x * cos_th - sum_y * sin_th)
        dpy = constant * (sum_x * sin_th + sum_y * cos_th)
        dtheta = constant / (self.c * self.r) * (-cy * sum_x + cx * sum_y)
        return ca.vertcat(dpx, dpy, dtheta)

    def _lam_mag(self, lam):
        """Signed forces mapped to nonnegative magnitudes."""
        return ca.vertcat(-lam[0], -lam[1], lam[2], -lam[3], lam[4], lam[5], lam[6], lam[7])

    def _gap(self, cy, v, w):
        l, dc = self.l, self.dc
        a = [v[i] + w[i] for i in range(7)]  # |.| per face
        return ca.vertcat(
            (4 - dc) * l - cy,
            a[0] + a[1] + a[2] - 1.0 * l,
            a[0] + a[2] + a[3] - 1.5 * l,
            a[2] + a[3] + a[4] - 3.0 * l,
            a[3] + a[4] + a[5] - 1.0 * l,
            a[2] + a[4] + a[5] - 3.0 * l,
            a[2] + a[5] + a[6] - 1.5 * l,
            a[1] + a[2] + a[6] - 1.0 * l)

    def G(self, x, u):
        cx, cy, lam, v, w = self._split(u)
        lam_mag = self._lam_mag(lam)
        terms = [v[i] for i in range(7)]                       # (A) v ⊥ w
        terms += [lam_mag[i] for i in range(8)]                # (B) lam_mag ⊥ gap
        for i in range(8):                                     # (C) mutual exclusion
            for j in range(i + 1, 8):
                terms.append(lam_mag[i])
        return ca.vertcat(*terms)

    def H(self, x, u):
        cx, cy, lam, v, w = self._split(u)
        lam_mag = self._lam_mag(lam)
        gap = self._gap(cy, v, w)
        terms = [w[i] for i in range(7)]
        terms += [gap[i] for i in range(8)]
        for i in range(8):
            for j in range(i + 1, 8):
                terms.append(lam_mag[j])
        return ca.vertcat(*terms)

    def eq(self, x, u):
        # Each face's signed offset equals v_i - w_i.
        cx, cy, lam, v, w = self._split(u)
        l, dc = self.l, self.dc
        return ca.vertcat(
            (v[0] - w[0]) - (cx - 2 * l),
            (v[1] - w[1]) - (cy - (4 - dc) * l),
            (v[2] - w[2]) - (cy - (3 - dc) * l),
            (v[3] - w[3]) - (cx - 0.5 * l),
            (v[4] - w[4]) - (cy + dc * l),
            (v[5] - w[5]) - (cx + 0.5 * l),
            (v[6] - w[6]) - (cx + 2 * l))

    def ineq(self, x, u):
        # Contact-point bounds, in the g <= 0 convention.
        cx, cy, lam, v, w = self._split(u)
        l, dc = self.l, self.dc
        return ca.vertcat(-2 * l - cx,          # cx >= -2l
                          cx - 2 * l,           # cx <=  2l
                          -dc * l - cy,         # cy >= -dc*l
                          cy - (4 - dc) * l)    # cy <= (4-dc)*l

    def control_lower_bounds(self):
        # No explicit bounds; the contact-point limits are the inequality block.
        return np.full(24, -np.inf)

    def control_upper_bounds(self):
        return np.full(24, np.inf)


def config(horizon: int = 50, start: Sequence[float] = (0.0, 0.0, 0.0),
           goal: Sequence[float] = (0.05, 0.05, 1.5708)) -> AulaConfig:
    """The tuned settings, from ``experiments/push_t/push_t_impact_multiple.cpp``.

    The conditioning scales are not incidental -- they were arrived at by
    measurement, and stripping them measurably degrades the solution here (the
    objective comes out 7.6x worse). See ``examples/README.md``.
    """
    c = AulaConfig()
    c.horizon = horizon
    c.stage_cost_weight = 0.01
    c.stage_state_cost_weight = 0.0
    c.control_rate_weight = 0.0
    c.final_cost_weight = 100.0
    all_scale = 25.0
    c.fix_point_scale = c.dynamics_scale = c.eq_scale = c.ineq_scale = all_scale
    c.comp_scale = 0.1
    c.rho_max = 1000.0
    c.rho_scale = 1.05
    c.max_outer_iters = 1000
    c.outer_tol_h = 1e-5
    c.outer_tol_comp = 1e-5
    c.outer_tol_g = 1e-5
    c.max_inner_iters = 50
    c.inner_tol_init = 5e-3
    c.inner_tol_final = 1e-3
    c.newton_max_iter = 200
    c.newton_tol = 1e-6
    c.newton_regularization = 5e-5
    # Quiet as a library call; the CLI turns the per-outer trace back on
    # (--print-level, default 1), which is what the C++ drivers print.
    c.print_level = 0
    c.x_0 = np.asarray(start, dtype=float)
    c.x_goal = np.asarray(goal, dtype=float)
    return c


def solve(cfg: Optional[AulaConfig] = None) -> Result:
    """Solve it through the multiple-shooting front end."""
    from impact.shooting import MultipleShootingSolver

    cfg = config() if cfg is None else cfg
    solution = MultipleShootingSolver(PushT()).solve(cfg)
    goal = np.asarray(cfg.x_goal, dtype=float).ravel()
    state = solution.state_trajectory
    labels = "px, py, theta"
    return Result(
        name="push_t", solution=solution, planner=planner_tag(cfg),
        state=state, control=solution.control_trajectory,
        start=np.asarray(cfg.x_0, dtype=float).ravel(), goal=goal,
        goal_error=float(np.max(np.abs(state[:, -1] - goal))),
        file=dict(title="Push-T AuLa Trajectory", start_label=labels,
                  goal_label=labels, state_label=labels, control_label=""),
        rows=(("final state", str(np.round(state[:, -1], 6))),
              ("goal", str(np.round(goal, 6)))))
