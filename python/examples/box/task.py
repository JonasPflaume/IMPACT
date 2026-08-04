"""Planar box pushing.

    State   x = [px, py, theta]                       box pose
    Control u = [cx, cy, ld1y, ld2x, ld3y, ld4x]      body-frame contact point,
                                                      then the four side forces

Quasi-static planar dynamics through the limit surface. The complementarity
constraints select which side/contact-force combination is active: four pairs tie
each side force to the contact point staying inside that side, and six more
enforce mutual exclusion between sides.

Port of ``experiments/box/box_pushing.{h,cpp}``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import casadi as ca
import numpy as np

from impact import AulaConfig
from impact.report import planner_tag
from impact.stage import MPCCProblem

from ..common import Result

__all__ = ["BoxPushing", "config", "solve"]


class BoxPushing(MPCCProblem):
    def __init__(self, mass: float = 0.1, gravity: float = 9.81, friction: float = 0.5,
                 integration_constant: float = 0.5, width: float = 0.3, height: float = 0.4,
                 dt: float = 0.05):
        self.m = mass
        self.g = gravity
        self.mu = friction
        self.c = integration_constant
        self.a = width
        self.b = height
        self.r = math.sqrt(width * width + height * height)  # characteristic contact distance
        self.dt = dt

    state_dim = property(lambda self: 3)
    control_dim = property(lambda self: 6)
    comp_dim = property(lambda self: 10)
    time_step = property(lambda self: self.dt)

    def dynamics(self, x, u):
        constant = 1.0 / (self.mu * self.m * self.g)
        theta = x[2]
        cx, cy, ld1y, ld2x, ld3y, ld4x = (u[0], u[1], u[2], u[3], u[4], u[5])
        dpx = constant * ((ld2x + ld4x) * ca.cos(theta) - (ld1y + ld3y) * ca.sin(theta))
        dpy = constant * ((ld2x + ld4x) * ca.sin(theta) + (ld1y + ld3y) * ca.cos(theta))
        dtheta = constant / (self.c * self.r) * (-cy * (ld2x + ld4x) + cx * (ld1y + ld3y))
        return ca.vertcat(dpx, dpy, dtheta)

    def G(self, x, u):
        cx, cy, ld1y, ld2x, ld3y, ld4x = (u[0], u[1], u[2], u[3], u[4], u[5])
        return ca.vertcat(
            ld1y,    # force at left side >= 0
            ld2x,    # force at right side >= 0
            -ld3y,   # force at bottom side >= 0 (negated)
            -ld4x,   # force at top side >= 0 (negated)
            ld1y,    # force coupling constraints
            ld1y, ld1y, ld2x, ld2x, -ld3y)

    def H(self, x, u):
        cx, cy, ld1y, ld2x, ld3y, ld4x = (u[0], u[1], u[2], u[3], u[4], u[5])
        return ca.vertcat(
            cy + self.b,   # contact point within box (left)
            cx + self.a,   # contact point within box (right)
            self.b - cy,   # contact point within box (bottom)
            self.a - cx,   # contact point within box (top)
            ld2x,          # force coupling (if left active, right compatible)
            -ld3y,         # force coupling (if left active, bottom compatible)
            -ld4x,         # force coupling (if left active, top compatible)
            -ld3y,         # force coupling (if right active, bottom compatible)
            -ld4x,         # force coupling (if right active, top compatible)
            -ld4x)         # force coupling (if bottom active, top compatible)


def config(horizon: int = 50, start: Sequence[float] = (0.0, 0.0, 0.0),
           goal: Sequence[float] = (0.1, 0.1, 1.0)) -> AulaConfig:
    """The tuned settings, from ``experiments/box/box_impact_multiple.cpp``.

    The conditioning scales and the penalty schedule are not incidental -- they
    were arrived at by measurement, and both are the driver's own numbers, so this
    solves the same problem the C++ binary does.
    """
    c = AulaConfig()
    c.horizon = horizon
    c.stage_cost_weight = 0.001
    c.final_cost_weight = 100.0
    all_scale = 25.0
    c.fix_point_scale = c.dynamics_scale = c.eq_scale = c.ineq_scale = all_scale
    c.comp_scale = 0.1
    c.rho_max = 200.0
    c.rho_scale = 1.05
    c.max_outer_iters = 500
    c.outer_tol_h = 1e-5
    c.outer_tol_comp = 1e-5
    c.max_inner_iters = 50
    c.inner_tol_init = 1e-2
    c.inner_tol_final = 1e-3
    c.newton_max_iter = 50
    c.newton_tol = 1e-6
    c.newton_regularization = 2e-5
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
    solution = MultipleShootingSolver(BoxPushing()).solve(cfg)
    goal = np.asarray(cfg.x_goal, dtype=float).ravel()
    state = solution.state_trajectory
    labels = "px, py, theta"
    return Result(
        name="box", solution=solution, planner=planner_tag(cfg),
        state=state, control=solution.control_trajectory,
        start=np.asarray(cfg.x_0, dtype=float).ravel(), goal=goal,
        goal_error=float(np.max(np.abs(state[:, -1] - goal))),
        file=dict(title="Box Pushing AuLa Trajectory", start_label=labels,
                  goal_label=labels, state_label=labels,
                  control_label="cx, cy, ld1y, ld2x, ld3y, ld4x"),
        rows=(("final state", str(np.round(state[:, -1], 6))),
              ("goal", str(np.round(goal, 6)))))
