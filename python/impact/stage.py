"""Per-stage task interfaces and the two adapters onto them.

A trajectory task is reduced to :class:`StageProblem` before it reaches the
solver: dimensions, a symbolic stage model, and a cost residual. The shooting
transcription is chosen *outside* the task (see :mod:`impact.shooting`), so the
same stage description serves both single and multiple shooting.

Two adapters cover the shapes a task usually has:

* :class:`MPCCStage` -- an explicit-ODE task with the config-driven quadratic
  cost.
* :class:`LCPStage` -- a contact task whose dynamics are algebraic (the LCP force
  balance) and whose state update is pure kinematics.

The repository's ``python/examples/<task>/task.py`` files have worked instances of
both, but nothing here depends on them: this package knows no tasks.

Ports of ``impact/stage_problem.h``, ``impact/mpcc_stage.h`` and
``impact/lcp_stage.h``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import casadi as ca
import numpy as np

__all__ = [
    "StageProblem",
    "MPCCProblem",
    "MPCCStage",
    "LCPProblem",
    "LCPStage",
    "quat_dcm",
]


class StageProblem(ABC):
    """Per-stage description consumed by the shooting builders.

    Symbolic methods are evaluated once, when the subproblem is built, never
    inside the solver loop. ``p`` is the per-solve runtime parameter vector -- the
    data that changes between solves, such as goals or contact data; tasks without
    any return ``runtime_param_dim == 0``.
    """

    # -- dimensions ---------------------------------------------------------
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """nx -- state x, subject to the shooting choice."""

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """nu -- controls plus always-free auxiliaries (e.g. lam / vel)."""

    @property
    @abstractmethod
    def comp_dim(self) -> int:
        """Complementarity pairs per stage."""

    @property
    def eq_dim(self) -> int:
        return 0

    @property
    def ineq_dim(self) -> int:
        return 0

    @property
    def dynamics_residual_dim(self) -> int:
        """Algebraic dynamics rows (the LCP force balance)."""
        return 0

    @property
    def runtime_param_dim(self) -> int:
        return 0

    @property
    @abstractmethod
    def time_step(self) -> float:
        ...

    @property
    def cost_is_linear(self) -> bool:
        """True when the multiple-shooting cost residual is linear in z.

        The objective is then quadratic and the builder can put the cost in the
        constant saddle block. Single shooting rolls out the state, so this only
        affects the multiple-shooting path.
        """
        return False

    # -- symbolic stage model ----------------------------------------------
    @abstractmethod
    def step(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """State evolution x_{k+1} = step(x_k, u_k, p)."""

    @abstractmethod
    def G(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """First leg of 0 <= G ⊥ H >= 0."""

    @abstractmethod
    def H(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """Second leg of 0 <= G ⊥ H >= 0."""

    def dynamics_residual(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """Algebraic dynamics residual r_dyn(x, u, p) = 0."""
        return ca.SX(0, 1)

    def eq(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        return ca.SX(0, 1)

    def ineq(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        return ca.SX(0, 1)

    @abstractmethod
    def cost_residual(self, X: Sequence[ca.SX], U: Sequence[ca.SX], p: ca.SX) -> ca.SX:
        """Full nonlinear-least-squares cost residual; objective = ||this||^2.

        The task owns its cost and emits the rows in its own order.
        """


class MPCCProblem(ABC):
    """An explicit-ODE MPCC task: dx/dt = f(x, u) plus complementarity.

    Unlike the C++ interface this returns CasADi expressions rather than
    ``casadi::Function`` objects. Calling an SX function inlines its graph anyway,
    so the assembled expression is the same one; writing the expression directly
    is simply what reads naturally in Python.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def control_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def comp_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def time_step(self) -> float:
        ...

    @abstractmethod
    def dynamics(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """dx/dt = f(x, u)."""

    @abstractmethod
    def G(self, x: ca.SX, u: ca.SX) -> ca.SX:
        ...

    @abstractmethod
    def H(self, x: ca.SX, u: ca.SX) -> ca.SX:
        ...

    @property
    def eq_dim(self) -> int:
        return 0

    @property
    def ineq_dim(self) -> int:
        return 0

    def eq(self, x: ca.SX, u: ca.SX) -> ca.SX:
        return ca.SX(0, 1)

    def ineq(self, x: ca.SX, u: ca.SX) -> ca.SX:
        return ca.SX(0, 1)

    # -- convenience -------------------------------------------------------
    def dynamics_function(self) -> ca.Function:
        """(x, u) -> f, for callers that want to roll a trajectory out numerically."""
        x = ca.SX.sym("x", self.state_dim)
        u = ca.SX.sym("u", self.control_dim)
        return ca.Function("dynamics", [x, u], [self.dynamics(x, u)])


class MPCCStage(StageProblem):
    """Encode an :class:`MPCCProblem` as a :class:`StageProblem`.

    ``step`` is explicit Euler; the cost is the config-driven quadratic (stage
    ``||u||^2``, optional ``||x - goal||^2``, optional control rate, terminal
    ``||x_T - goal||^2``). The runtime parameter is ``p = x_goal``.
    """

    def __init__(self, problem: MPCCProblem, config):
        self.problem = problem
        self._nx = problem.state_dim
        self._nu = problem.control_dim
        self._nc = problem.comp_dim
        self._neq = problem.eq_dim
        self._nineq = problem.ineq_dim
        self._dt = problem.time_step
        self.stage_cost_weight = config.stage_cost_weight
        self.stage_state_cost_weight = config.stage_state_cost_weight
        self.control_rate_weight = config.control_rate_weight
        self.final_cost_weight = config.final_cost_weight

    state_dim = property(lambda self: self._nx)
    control_dim = property(lambda self: self._nu)
    comp_dim = property(lambda self: self._nc)
    eq_dim = property(lambda self: self._neq)
    ineq_dim = property(lambda self: self._nineq)
    time_step = property(lambda self: self._dt)
    runtime_param_dim = property(lambda self: self._nx)  # x_goal
    cost_is_linear = property(lambda self: True)  # quadratic objective

    def step(self, x, u, p):
        return x + self._dt * self.problem.dynamics(x, u)

    def G(self, x, u, p):
        return self.problem.G(x, u)

    def H(self, x, u, p):
        return self.problem.H(x, u)

    def eq(self, x, u, p):
        return self.problem.eq(x, u)

    def ineq(self, x, u, p):
        return self.problem.ineq(x, u)

    def cost_residual(self, X, U, p):
        # p == x_goal. Row order: per stage [sqrt(stage)*u, optional
        # sqrt(state)*(x - goal)], optional control-rate rows, then the terminal
        # sqrt(final)*(x_T - goal).
        horizon = len(U)
        rows: List[ca.SX] = []
        for t in range(horizon):
            rows.append(math.sqrt(self.stage_cost_weight) * U[t])
            if self.stage_state_cost_weight > 0.0:
                rows.append(math.sqrt(self.stage_state_cost_weight) * (X[t] - p))
        if self.control_rate_weight > 0.0:
            for t in range(horizon - 1):
                rows.append(math.sqrt(self.control_rate_weight) * (U[t + 1] - U[t]))
        rows.append(math.sqrt(self.final_cost_weight) * (X[horizon] - p))
        return ca.vertcat(*rows)


def quat_dcm(q: ca.SX) -> ca.SX:
    """Direction-cosine matrix of a (w, x, y, z) quaternion."""
    return ca.vertcat(
        ca.horzcat(1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]),
                   2 * (q[1] * q[3] + q[0] * q[2])),
        ca.horzcat(2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]),
                   2 * (q[2] * q[3] - q[0] * q[1])),
        ca.horzcat(2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
                   1 - 2 * (q[1] * q[1] + q[2] * q[2])))


class LCPProblem(ABC):
    """A single-shooting LCP contact task.

    The configuration q is rolled out from velocities; the free variables per step
    are ``[cmd, lam, vel]``. The force balance appears as an algebraic dynamics
    residual rather than a state update, which is what lets a contact task reuse
    the same stage interface as an ODE task.
    """

    @property
    @abstractmethod
    def config_dim(self) -> int:
        """n_qpos."""

    @property
    @abstractmethod
    def velocity_dim(self) -> int:
        """n_qvel."""

    @property
    @abstractmethod
    def command_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def max_contacts(self) -> int:
        ...

    @property
    @abstractmethod
    def time_step(self) -> float:
        ...

    @abstractmethod
    def inertia_matrix(self) -> np.ndarray:
        """Q (n_qvel x n_qvel)."""

    @abstractmethod
    def gravity_bias(self) -> np.ndarray:
        """b_gravity (n_qvel)."""

    def robot_stiffness(self) -> np.ndarray:
        return 100.0 * np.eye(self.command_dim)

    # Cost weights.
    @property
    @abstractmethod
    def control_cost_weight(self) -> float:
        ...

    @property
    @abstractmethod
    def contact_cost_weight(self) -> float:
        ...

    @property
    @abstractmethod
    def grasp_closure_weight(self) -> float:
        ...

    @property
    @abstractmethod
    def velocity_penalty(self) -> float:
        ...

    @property
    @abstractmethod
    def position_cost_weight(self) -> float:
        ...

    @property
    @abstractmethod
    def orientation_cost_weight(self) -> float:
        ...

    @property
    @abstractmethod
    def final_cost_multiplier(self) -> float:
        ...

    @property
    def final_position_weight(self) -> float:
        return self.position_cost_weight

    @property
    def final_orientation_weight(self) -> float:
        return self.orientation_cost_weight

    @property
    def num_fingertips(self) -> int:
        return 3

    def fingertip_positions_sx(self, q: ca.SX) -> ca.SX:
        return q[7:7 + 3 * self.num_fingertips]

    def integrate_state(self, q: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError("integrate_state not implemented")


class LCPStage(StageProblem):
    """Adapter from an :class:`LCPProblem` to :class:`StageProblem`.

    The mapping is a DAE-style MPCC::

        x = q                                   (configuration)
        u = [cmd, lam, vel]                     (controls + always-free contact aux)
        step(x, u) = integrate(q, vel)          (quaternion-aware kinematics)
        dynamics_residual = Q*vel - b/h - J'λ/h (LCP force balance, algebraic)
        G = lam,  H = J*vel + phi/h             (contact complementarity)
        ineq = [cmd_lo - cmd; cmd - cmd_hi]     (control bounds)

    ``p = [target_p(3), target_q(4), phi(n_lam), vec(J)(n_lam*n_qvel)]`` changes
    every MPC step; the symbolic functions are built once.
    """

    def __init__(self, problem: LCPProblem, config):
        self.problem = problem
        self.n_qpos = problem.config_dim
        self.n_qvel = problem.velocity_dim
        self.n_cmd = problem.command_dim
        self.n_lam = problem.max_contacts * 4
        self.n_ftp = problem.num_fingertips
        self.h = problem.time_step
        self.control_w = problem.control_cost_weight
        self.contact_w = problem.contact_cost_weight
        self.grasp_w = problem.grasp_closure_weight
        self.vel_w = problem.velocity_penalty
        self.final_mult = problem.final_cost_multiplier
        self.final_pos_w = problem.final_position_weight
        self.final_ori_w = problem.final_orientation_weight
        self.Q = np.asarray(problem.inertia_matrix(), dtype=float)
        self.robot_stiff = np.asarray(problem.robot_stiffness(), dtype=float)
        self.b_o = np.asarray(problem.gravity_bias(), dtype=float)[:3]

        cmd_lower = np.asarray(config.cmd_lower, dtype=float).ravel()
        cmd_upper = np.asarray(config.cmd_upper, dtype=float).ravel()
        self.use_cmd = bool(config.use_cmd_bounds and cmd_lower.size == self.n_cmd
                            and cmd_upper.size == self.n_cmd)
        self.cmd_lower = cmd_lower if self.use_cmd else None
        self.cmd_upper = cmd_upper if self.use_cmd else None

    state_dim = property(lambda self: self.n_qpos)
    control_dim = property(lambda self: self.n_cmd + self.n_lam + self.n_qvel)
    comp_dim = property(lambda self: self.n_lam)
    ineq_dim = property(lambda self: 2 * self.n_cmd if self.use_cmd else 0)
    dynamics_residual_dim = property(lambda self: self.n_qvel)
    runtime_param_dim = property(lambda self: 3 + 4 + self.n_lam + self.n_lam * self.n_qvel)
    time_step = property(lambda self: self.h)

    # -- layout helpers -----------------------------------------------------
    def cmd_of(self, u):
        return u[0:self.n_cmd]

    def lam_of(self, u):
        return u[self.n_cmd:self.n_cmd + self.n_lam]

    def vel_of(self, u):
        return u[self.n_cmd + self.n_lam:self.n_cmd + self.n_lam + self.n_qvel]

    def target_p_of(self, p):
        return p[0:3]

    def target_q_of(self, p):
        return p[3:7]

    def phi_of(self, p):
        return p[7:7 + self.n_lam]

    def jac_of(self, p):
        return ca.reshape(p[7 + self.n_lam:7 + self.n_lam + self.n_lam * self.n_qvel],
                          self.n_lam, self.n_qvel)

    # -- stage model --------------------------------------------------------
    def step(self, x, u, p):
        vel = self.vel_of(u)
        obj_pos, obj_quat, robot_q = x[0:3], x[3:7], x[7:self.n_qpos]
        v_lin, v_ang, v_robot = vel[0:3], vel[3:6], vel[6:self.n_qvel]
        next_pos = obj_pos + self.h * v_lin
        next_robot = robot_q + self.h * v_robot
        Hqb = ca.vertcat(
            ca.horzcat(-obj_quat[1], obj_quat[0], obj_quat[3], -obj_quat[2]),
            ca.horzcat(-obj_quat[2], -obj_quat[3], obj_quat[0], obj_quat[1]),
            ca.horzcat(-obj_quat[3], obj_quat[2], -obj_quat[1], obj_quat[0]))
        next_quat = obj_quat + 0.5 * self.h * ca.mtimes(Hqb.T, v_ang)
        next_quat = next_quat / ca.norm_2(next_quat)
        return ca.vertcat(next_pos, next_quat, next_robot)

    def G(self, x, u, p):
        return self.lam_of(u)

    def H(self, x, u, p):
        return ca.mtimes(self.jac_of(p), self.vel_of(u)) + self.phi_of(p) / self.h

    def dynamics_residual(self, x, u, p):
        Q_sx = ca.SX(ca.DM(self.Q))
        stiff_sx = ca.SX(ca.DM(self.robot_stiff))
        lhs = ca.mtimes(Q_sx, self.vel_of(u))
        b_r = ca.mtimes(stiff_sx, self.cmd_of(u))
        b_full = ca.vertcat(ca.DM(self.b_o), ca.SX.zeros(3, 1), b_r)
        rhs = b_full / self.h + ca.mtimes(self.jac_of(p).T, self.lam_of(u)) / self.h
        return lhs - rhs

    def ineq(self, x, u, p):
        cmd = self.cmd_of(u)
        return ca.vertcat(ca.DM(self.cmd_lower) - cmd, cmd - ca.DM(self.cmd_upper))

    def cost_residual(self, X, U, p):
        horizon = len(U)
        res: List[ca.SX] = []
        for k in range(horizon):
            q_next = X[k + 1]
            res.append(math.sqrt(self.control_w) * self.cmd_of(U[k]))
            obj_pos_next = q_next[0:3]
            ftp = self.problem.fingertip_positions_sx(q_next)
            for f in range(self.n_ftp):
                res.append(math.sqrt(self.contact_w) * (obj_pos_next - ftp[3 * f:3 * f + 3]))
            dcm = quat_dcm(q_next[3:7])
            grasp_sum = ca.SX.zeros(3, 1)
            for f in range(self.n_ftp):
                v_f = ca.mtimes(dcm.T, ftp[3 * f:3 * f + 3] - obj_pos_next)
                grasp_sum = grasp_sum + v_f / ca.norm_2(v_f)
            res.append(math.sqrt(self.grasp_w) * grasp_sum)
            res.append(math.sqrt(self.vel_w) * self.vel_of(U[k])[0:6])

        # Terminal position + quaternion-log error.
        q_final = X[horizon]
        target_p, target_q = self.target_p_of(p), self.target_q_of(p)
        res.append(math.sqrt(self.final_mult * self.final_pos_w) * (q_final[0:3] - target_p))
        qf = q_final[3:7]
        w1, x1, y1, z1 = qf[0], qf[1], qf[2], qf[3]
        w2, x2, y2, z2 = target_q[0], target_q[1], target_q[2], target_q[3]
        qrw = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
        qrx = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
        qry = -w1 * y2 + y1 * w2 - z1 * x2 + x1 * z2
        qrz = -w1 * z2 + z1 * w2 - x1 * y2 + y1 * x2
        vnorm = ca.sqrt(qrx * qrx + qry * qry + qrz * qrz + 1e-12)
        theta = 2 * ca.atan2(vnorm, ca.fabs(qrw))
        qscale = ca.if_else(vnorm > 1e-6, theta / vnorm, 2.0)
        quat_log = ca.vertcat(qscale * qrx, qscale * qry, qscale * qrz)
        res.append(math.sqrt(self.final_mult * self.final_ori_w) * quat_log)
        return ca.vertcat(*res)
