"""Planar single-point pushing of a rigid disk.

Quasi-static pusher-slider with pusher/slider friction::

    State   x = [qx, qy, sx, sy]      disk center q, pusher point s
    Control u = [f_n, f_t, vx, vy]    normal force, tangential (friction) force,
                                      pusher velocity

The disk is symmetric, so its orientation is decoupled from the signed-distance
field, the contact geometry and the goal, and is not tracked. The tangential
friction force still steers the disk translationally through the isotropic limit
surface, so contact friction is fully modelled for the task.

Contact geometry (world frame). With ``d = s - q`` and ``r = ||d||`` the smooth
signed distance is ``phi(x) = r - R``. The pusher pushes along the inward normal
``-d/r`` and may apply a tangential force along ``(-dy, dx)/r``; the quasi-static
limit surface maps the net contact force to the disk velocity ``q̇ = c_trans F``
with ``c_trans = 1/(mu_s m g)``. The pusher moves kinematically, ``ṡ = v``.

The complementarity ``0 <= f_n ⊥ phi >= 0`` both activates the normal force only
on contact and, through ``phi >= 0``, keeps the pusher outside the disk -- which
is what forces a valid trajectory to route around it. The friction cone
``|f_t| <= mu_c f_n`` is a stage inequality.

Port of ``experiments/push_circle/push_circle.{h,cpp}``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from impact import AulaConfig
from impact.report import planner_tag
from impact.stage import MPCCProblem, MPCCStage

from ..common import Result

__all__ = ["PushCircle", "DISK_DIM", "scenario", "config", "solve"]

#: Leading state components that make up the task goal (the disk center).
DISK_DIM = 2


class PushCircle(MPCCProblem):
    def __init__(self, mass: float = 1.0, gravity: float = 9.81, ground_friction: float = 0.1,
                 pusher_friction: float = 0.1, radius: float = 0.3, dt: float = 0.05):
        self.m = mass
        self.g = gravity
        self.mu_s = ground_friction   # slider/ground friction (limit surface)
        self.mu_c = pusher_friction   # pusher/slider friction (contact friction cone)
        self.R = radius
        self.dt = dt

    state_dim = property(lambda self: 4)
    control_dim = property(lambda self: 4)
    comp_dim = property(lambda self: 1)
    eq_dim = property(lambda self: 0)
    ineq_dim = property(lambda self: 2)  # friction cone
    time_step = property(lambda self: self.dt)

    # -- geometry ----------------------------------------------------------
    def _offset(self, x):
        """(dx, dy, ||d||) from the disk center to the pusher.

        The +1e-9 under the square root only matters if the pusher reaches the
        disk center, which non-penetration prevents, so it never distorts contact.
        """
        dx = x[2] - x[0]
        dy = x[3] - x[1]
        return dx, dy, ca.sqrt(dx * dx + dy * dy + 1e-9)

    def dynamics(self, x, u):
        # Quasi-static limit surface q̇ = c_trans * F, with the contact force
        # F = f_n * n_in + f_t * t expressed directly in the world frame.
        c_trans = 1.0 / (self.mu_s * self.m * self.g)
        fn, ft, vx, vy = u[0], u[1], u[2], u[3]
        dx, dy, nrm = self._offset(x)
        Fx = (-fn * dx - ft * dy) / nrm
        Fy = (-fn * dy + ft * dx) / nrm
        return ca.vertcat(c_trans * Fx, c_trans * Fy, vx, vy)

    def G(self, x, u):
        return ca.vertcat(u[0])  # f_n

    def H(self, x, u):
        _, _, nrm = self._offset(x)
        return ca.vertcat(nrm - self.R)  # phi

    def ineq(self, x, u):
        # Coulomb friction cone, g <= 0 convention:
        #   |f_t| <= mu_c f_n  =>  f_t - mu_c f_n <= 0,  -f_t - mu_c f_n <= 0.
        # With f_n = 0 (no contact) this collapses to f_t = 0.
        fn, ft = u[0], u[1]
        return ca.vertcat(ft - self.mu_c * fn, -ft - self.mu_c * fn)


# ------------------------------------------------------------------ scenario --


def scenario(distance: float = 1.5, angle_deg: float = 225.0,
             goal: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The driver's scenario: ``(x_0, x_goal)``.

    The disk starts at the origin and the pusher at ``distance`` in direction
    ``angle_deg``; the disk's goal defaults to the pusher's own start, so pushing
    from the initial contact drives the disk the wrong way and the pusher has to
    orbit it.

    The pusher's terminal target is the contact point behind the disk along the
    direction of travel. The terminal knot carries no per-stage contact constraint,
    so giving the pusher a target there is what keeps that knot non-penetrating.
    """
    ang = np.deg2rad(angle_deg)
    tx, ty = distance * np.cos(ang), distance * np.sin(ang)
    gx, gy = (tx, ty) if goal is None else (float(goal[0]), float(goal[1]))
    norm = math.hypot(gx, gy)
    dx, dy = (gx / norm, gy / norm) if norm > np.finfo(float).eps else (np.cos(ang), np.sin(ang))
    radius = PushCircle().R
    return (np.array([0.0, 0.0, tx, ty]),
            np.array([gx, gy, gx - radius * dx, gy - radius * dy]))


def config(horizon: int = 100, distance: float = 1.5, angle: float = 225.0,
           goal: Optional[Sequence[float]] = None) -> AulaConfig:
    """The tuned settings, from ``experiments/push_circle/push_circle_impact_multiple.cpp``."""
    c = AulaConfig()
    c.horizon = horizon
    c.use_constant_state_init = True  # replicate the start: the hard init
    c.stage_cost_weight = 1e-2
    c.final_cost_weight = 100.0
    all_scale = 10.0
    c.fix_point_scale = c.dynamics_scale = c.eq_scale = c.ineq_scale = all_scale
    c.comp_scale = 1.0
    c.rho_max = 200.0
    c.rho_scale = 1.05
    c.max_outer_iters = 800
    c.outer_tol_h = 1e-5
    c.outer_tol_g = 1e-5
    c.outer_tol_comp = 1e-5
    c.max_inner_iters = 50
    c.inner_tol_init = 1e-2
    c.inner_tol_final = 1e-3
    c.newton_max_iter = 100
    c.newton_tol = 1e-5
    c.newton_regularization = 1e-5
    # Quiet as a library call; the CLI turns the per-outer trace back on
    # (--print-level, default 1), which is what the C++ drivers print.
    c.print_level = 0
    c.x_0, c.x_goal = scenario(distance, angle, goal)
    return c


def solve(cfg: Optional[AulaConfig] = None, mode: str = "multiple") -> Result:
    """Solve it, and measure the go-around the trajectory had to perform."""
    from impact.shooting import MultipleShootingSolver, SingleShootingSolver

    cfg = config() if cfg is None else cfg
    problem = PushCircle()
    front_end = MultipleShootingSolver if mode == "multiple" else SingleShootingSolver
    solution = front_end(problem).solve(cfg)

    x0 = np.asarray(cfg.x_0, dtype=float).ravel()
    goal = np.asarray(cfg.x_goal, dtype=float).ravel()
    state = solution.state_trajectory
    disk, pusher = state[:DISK_DIM, :], state[DISK_DIM:, :]
    # How far the pusher swept around the disk: the go-around, measured.
    angles = np.unwrap(np.arctan2(*(pusher - disk)[::-1, :]))
    contact = int(np.argmax(solution.control_trajectory[0] > 1e-3))

    tag = planner_tag(cfg) + ("_single" if mode == "single" else "")
    return Result(
        name="push_circle", solution=solution, planner=tag, state=state,
        control=solution.control_trajectory, start=x0, goal=goal[:DISK_DIM],
        goal_error=float(np.max(np.abs(disk[:, -1] - goal[:DISK_DIM]))),
        file=dict(title="Disk Pushing AuLa Trajectory",
                  preamble=[("Disk Radius", [problem.R])],
                  start_label="qx, qy, sx, sy", goal_label="qx, qy",
                  state_label="qx, qy, sx, sy", control_label="fn, ft, vx, vy"),
        rows=(("final disk", str(np.round(disk[:, -1], 5))),
              ("min pusher-disk gap",
               f"{float(np.linalg.norm(pusher - disk, axis=0).min()):.5f} "
               f"(disk radius {problem.R})"),
              ("pusher swept angle",
               f"{np.rad2deg(abs(angles[-1] - angles[0])):.1f} deg   <- the go-around"),
              ("first contact at step", str(contact))))
