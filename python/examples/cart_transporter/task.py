"""Cargo on a cart, with Coulomb friction between them.

    State   x = [x1, x2, x1_dot, x2_dot]   cargo and cart positions and velocities
    Control u = [f, u_cart, v, w]          friction force, cart control force, and
                                           the slack pair for |x1_dot - x2_dot|

The cargo can slide on the cart. The relative velocity is split as ``v - w`` with
``v, w >= 0``, and the three complementarity pairs encode the Coulomb law:

  1. ``v ⊥ w``                   -- it cannot slide both ways at once;
  2. ``w ⊥ (mu m1 g - f)``       -- sliding backward pins friction to its upper limit;
  3. ``v ⊥ (f + mu m1 g)``       -- sliding forward pins it to its lower limit.

Sticking is the remaining case, where both slacks vanish and the friction force is
free inside the cone.

Port of ``experiments/cart_transporter/cart_transporter.{h,cpp}``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import casadi as ca
import numpy as np

from impact import AulaConfig
from impact.report import planner_tag
from impact.stage import MPCCProblem

from ..common import Result

__all__ = ["CartTransporter", "config", "solve"]


class CartTransporter(MPCCProblem):
    def __init__(self, cargo_mass: float = 0.1, cart_mass: float = 0.2, friction: float = 0.2,
                 gravity: float = 9.81, gap_length: float = 1.0, dt: float = 0.02):
        self.m1 = cargo_mass
        self.m2 = cart_mass
        self.mu = friction
        self.g = gravity
        self.l = gap_length  # half gap: the cargo must stay within |x1 - x2| <= l
        self.dt = dt

    state_dim = property(lambda self: 4)
    control_dim = property(lambda self: 4)
    comp_dim = property(lambda self: 3)
    eq_dim = property(lambda self: 1)
    ineq_dim = property(lambda self: 4)
    time_step = property(lambda self: self.dt)

    @property
    def _friction_limit(self) -> float:
        return self.mu * self.m1 * self.g

    def dynamics(self, x, u):
        x1_dot, x2_dot = x[2], x[3]
        f, u_cart = u[0], u[1]
        return ca.vertcat(x1_dot, x2_dot, f / self.m1, (u_cart - f) / self.m2)

    def G(self, x, u):
        f, u_cart, v, w = u[0], u[1], u[2], u[3]
        return ca.vertcat(v, w, v)

    def H(self, x, u):
        f, u_cart, v, w = u[0], u[1], u[2], u[3]
        lim = self._friction_limit
        return ca.vertcat(w, lim - f, f + lim)

    def eq(self, x, u):
        # x1_dot - x2_dot = v - w.
        x1_dot, x2_dot = x[2], x[3]
        v, w = u[2], u[3]
        return ca.vertcat(x1_dot - x2_dot - v + w)

    def ineq(self, x, u):
        # g <= 0 convention.
        x1, x2 = x[0], x[1]
        f = u[0]
        lim = self._friction_limit
        return ca.vertcat(-(lim - f),               # f <= mu m1 g
                          -(f + lim),               # f >= -mu m1 g
                          -(x1 - x2 + self.l),      # x1 - x2 >= -l
                          -(self.l - (x1 - x2)))    # x1 - x2 <=  l

    def control_lower_bounds(self):
        return np.array([-np.inf, -np.inf, 0.0, 0.0])  # v, w are slacks

    def control_upper_bounds(self):
        return np.full(4, np.inf)


def config(horizon: int = 300, start: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
           goal: Sequence[float] = (1.0, 1.0, 0.0, 0.0)) -> AulaConfig:
    """The tuned settings, from
    ``experiments/cart_transporter/cart_transporter_impact_multiple.cpp``.

    The longest horizon here (300 knots) and the only task started from a
    replicated initial state. See ``examples/README.md`` for what the scales do.
    """
    c = AulaConfig()
    c.horizon = horizon
    c.stage_cost_weight = 1e-6
    c.stage_state_cost_weight = 0.0
    c.final_cost_weight = 5000.0
    c.fix_point_scale = c.dynamics_scale = c.eq_scale = c.ineq_scale = 1.0
    c.comp_scale = 0.002
    c.rho_max = 100000.0
    c.rho_scale = 1.5
    c.max_outer_iters = 1000
    c.outer_tol_h = 1e-5
    c.outer_tol_comp = 1e-5
    c.outer_tol_g = 1e-5
    c.max_inner_iters = 10
    c.inner_tol_init = 1e-2
    c.inner_tol_final = 1e-3
    c.newton_max_iter = 100
    c.newton_tol = 1e-6
    c.newton_regularization = 1e-5
    c.use_constant_state_init = True
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
    solution = MultipleShootingSolver(CartTransporter()).solve(cfg)
    goal = np.asarray(cfg.x_goal, dtype=float).ravel()
    state = solution.state_trajectory
    labels = "x1, x2, x1_dot, x2_dot"
    return Result(
        name="cart_transporter", solution=solution, planner=planner_tag(cfg),
        state=state, control=solution.control_trajectory,
        start=np.asarray(cfg.x_0, dtype=float).ravel(), goal=goal,
        goal_error=float(np.max(np.abs(state[:, -1] - goal))),
        file=dict(title="Cart Transporter AuLa Trajectory", start_label=labels,
                  goal_label=labels, state_label=labels, control_label="f, u, v, w"),
        rows=(("final state", str(np.round(state[:, -1], 6))),
              ("goal", str(np.round(goal, 6)))))
