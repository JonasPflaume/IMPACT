"""Shooting transcriptions: stage description in, solver-ready subproblem out.

Multiple shooting keeps the state trajectory free and enforces the dynamics as
defect equalities; single shooting rolls the state out symbolically so only the
controls are free. Both end in :func:`impact.mpcc.build_mpcc`, and the *order* in
which they add their blocks is part of the transcription -- it fixes the residual
row layout the saddle solver indexes into -- so it is reproduced here exactly as
in ``multiple_shooting.cpp`` / ``single_shooting.cpp``.

One asymmetry is deliberate and easy to misread: multiple shooting calls the
state-defect block ``dynamics`` and the algebraic force balance ``physics``,
while single shooting has no defect block and calls the force balance
``dynamics``. Renaming either would change which violation a driver reads back
out of ``AulaResult.dynamics_violation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import casadi as ca
import numpy as np

from . import _impact_core as _core
from .mpcc import BlockOptions, MPCCDescription, build_mpcc
from .stage import LCPProblem, LCPStage, MPCCProblem, MPCCStage, StageProblem

__all__ = [
    "ShootingLayout",
    "build_multiple_shooting",
    "build_single_shooting",
    "TrajectorySolution",
    "ShootingSolution",
    "LCPSolution",
    "MultipleShootingSolver",
    "SingleShootingSolver",
    "LCPSingleShootingSolver",
]


@dataclass
class ShootingLayout:
    """An assembled subproblem plus the offsets of its per-solve data."""

    subproblem: "_core.Subproblem"
    off_p: int = 0   # task runtime parameter (goal / contact data)
    off_x0: int = 0  # initial state
    n_opt: int = 0
    n_params: int = 0
    # The CasADi functions behind the subproblem, for inspection and for the
    # parity comparison against the C++ builders. See BuiltMPCC.
    residual: Optional["ca.Function"] = None
    jacobian: Optional["ca.Function"] = None
    gh: Optional["ca.Function"] = None


def _apply_common(desc: MPCCDescription, config) -> None:
    desc.check_stationarity = config.check_stationarity
    desc.conditioned_complementarity = config.conditioned_complementarity
    desc.stationarity_tol = config.stationarity_tol
    desc.max_stagnation_restarts = config.max_stagnation_restarts
    desc.jit = config.jit


def build_multiple_shooting(stage: StageProblem, config) -> ShootingLayout:
    """z = [vec(X); vec(U)], dynamics as defect equalities, x_0 as its own block."""
    nx = stage.state_dim
    nu = stage.control_dim
    nc = stage.comp_dim
    neq_s = stage.eq_dim
    nineq_s = stage.ineq_dim
    ndynres_s = stage.dynamics_residual_dim
    np_rt = stage.runtime_param_dim
    horizon = config.horizon

    X = ca.SX.sym("X", nx, horizon + 1)
    U = ca.SX.sym("U", nu, horizon)
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    Xlist = [X[:, t] for t in range(horizon + 1)]
    Ulist = [U[:, t] for t in range(horizon)]

    p_rt = ca.SX.sym("p_rt", max(np_rt, 1))
    x_0 = ca.SX.sym("x_0", nx)

    G_list, H_list, h_list, dynres_list, eq_list, ineq_list = [], [], [], [], [], []
    for t in range(horizon):
        xt, ut = Xlist[t], Ulist[t]
        G_list.append(stage.G(xt, ut, p_rt))
        H_list.append(stage.H(xt, ut, p_rt))
        h_list.append(Xlist[t + 1] - stage.step(xt, ut, p_rt))  # dynamics defect
        if ndynres_s > 0:
            dynres_list.append(stage.dynamics_residual(xt, ut, p_rt))
        if neq_s > 0:
            eq_list.append(stage.eq(xt, ut, p_rt))
        if nineq_s > 0:
            ineq_list.append(stage.ineq(xt, ut, p_rt))

    desc = MPCCDescription(z=z, p=ca.vertcat(p_rt, x_0),
                           cost=stage.cost_residual(Xlist, Ulist, p_rt),
                           cost_is_linear=stage.cost_is_linear)
    _apply_common(desc, config)

    if nc > 0:
        desc.add_complementarity("comp", G_list, H_list,
                                 BlockOptions(config.comp_scale, config.rho_comp_init,
                                              config.outer_tol_comp))
    desc.add_equality("dynamics", h_list,
                      BlockOptions(config.dynamics_scale, config.rho_dynamics_init,
                                   config.outer_tol_h))
    if ndynres_s > 0:
        desc.add_equality("physics", dynres_list,
                          BlockOptions(config.dynamics_scale, config.rho_dynamics_init,
                                       config.outer_tol_h))
    if neq_s > 0:
        desc.add_equality("equality", eq_list,
                          BlockOptions(config.eq_scale, config.rho_eq_init, config.outer_tol_h))
    if nineq_s > 0:
        desc.add_inequality("inequality", ineq_list,
                            BlockOptions(config.ineq_scale, config.rho_ineq_init,
                                         config.outer_tol_g))
    desc.add_equality("fix_point", Xlist[0] - x_0,
                      BlockOptions(config.fix_point_scale, config.rho_fix_point_init,
                                   config.outer_tol_h))

    built = build_mpcc(desc)
    return ShootingLayout(subproblem=built.subproblem, off_p=built.off_p,
                          off_x0=built.off_p + max(np_rt, 1), n_opt=built.n_opt,
                          n_params=built.n_params, residual=built.residual,
                          jacobian=built.jacobian, gh=built.gh)


def build_single_shooting(stage: StageProblem, config) -> ShootingLayout:
    """z = [vec(U)]; the state is rolled out from x_0 via ``step``."""
    horizon = config.horizon
    nx = stage.state_dim
    nu = stage.control_dim
    nc = stage.comp_dim
    ndynres_s = stage.dynamics_residual_dim
    nineq_s = stage.ineq_dim
    neq_s = stage.eq_dim
    np_rt = max(stage.runtime_param_dim, 1)

    n_opt = nu * horizon
    z = ca.SX.sym("z", n_opt)
    Ulist = [z[k * nu:(k + 1) * nu] for k in range(horizon)]

    x_0 = ca.SX.sym("x_0", nx)
    p_rt = ca.SX.sym("p_rt", np_rt)

    Xlist = [x_0]
    for k in range(horizon):
        Xlist.append(stage.step(Xlist[k], Ulist[k], p_rt))

    G_list, H_list, dynres_list, ineq_list, eq_list = [], [], [], [], []
    for k in range(horizon):
        G_list.append(stage.G(Xlist[k], Ulist[k], p_rt))
        H_list.append(stage.H(Xlist[k], Ulist[k], p_rt))
        if ndynres_s > 0:
            dynres_list.append(stage.dynamics_residual(Xlist[k], Ulist[k], p_rt))
        if nineq_s > 0:
            ineq_list.append(stage.ineq(Xlist[k], Ulist[k], p_rt))
        if neq_s > 0:
            eq_list.append(stage.eq(Xlist[k], Ulist[k], p_rt))

    desc = MPCCDescription(z=z, p=ca.vertcat(p_rt, x_0),
                           cost=stage.cost_residual(Xlist, Ulist, p_rt),
                           # The rolled-out state makes the cost nonlinear in z.
                           cost_is_linear=False)
    _apply_common(desc, config)

    if ndynres_s > 0:
        desc.add_equality("dynamics", dynres_list,
                          BlockOptions(config.dynamics_scale, config.rho_dynamics_init,
                                       config.outer_tol_h))
    if nc > 0:
        desc.add_complementarity("comp", G_list, H_list,
                                 BlockOptions(config.comp_scale, config.rho_comp_init,
                                              config.outer_tol_comp))
    if nineq_s > 0:
        desc.add_inequality("inequality", ineq_list,
                            BlockOptions(config.ineq_scale, config.rho_ineq_init,
                                         config.outer_tol_g))
    if neq_s > 0:
        desc.add_equality("equality", eq_list,
                          BlockOptions(config.eq_scale, config.rho_eq_init, config.outer_tol_h))

    built = build_mpcc(desc)
    return ShootingLayout(subproblem=built.subproblem, off_p=built.off_p,
                          off_x0=built.off_p + np_rt, n_opt=built.n_opt,
                          n_params=built.n_params, residual=built.residual,
                          jacobian=built.jacobian, gh=built.gh)


# --------------------------------------------------------------------- results --


@dataclass
class TrajectorySolution:
    """Solver statistics shared by both shooting front-ends."""

    objective_value: float = 0.0
    dynamics_violation: float = 0.0
    equality_violation: float = 0.0
    inequality_violation: float = 0.0
    complementarity_violation: float = 0.0
    stationarity_violation: float = 0.0
    # Tolerance-free MPCC complementarity certificate: a W-stationary point drives
    # all four to zero, so no active-set threshold has to be chosen to read them.
    comp_neg_G: float = 0.0
    comp_neg_H: float = 0.0
    comp_support_G: float = 0.0
    comp_support_H: float = 0.0
    converged: bool = False
    outer_iterations: int = 0
    total_inner_iterations: int = 0
    total_gn_iterations: int = 0
    solve_time: float = 0.0
    # Where solve_time went: CasADi evaluations vs sparse factorizations. The rest
    # is assembly, the complementarity projection and outer-loop bookkeeping.
    eval_time: float = 0.0
    factor_time: float = 0.0
    status_message: str = ""
    z: Optional[np.ndarray] = None

    @classmethod
    def _from_result(cls, r) -> "TrajectorySolution":
        return cls(objective_value=r.objective_value, dynamics_violation=r.dynamics_violation,
                   equality_violation=r.equality_violation,
                   inequality_violation=r.inequality_violation,
                   complementarity_violation=r.complementarity_violation,
                   stationarity_violation=r.stationarity_violation, comp_neg_G=r.comp_neg_G,
                   comp_neg_H=r.comp_neg_H, comp_support_G=r.comp_support_G,
                   comp_support_H=r.comp_support_H, converged=r.converged,
                   outer_iterations=r.outer_iterations,
                   total_inner_iterations=r.total_inner_iterations,
                   total_gn_iterations=r.total_gn_iterations,
                   solve_time=r.solve_time,
                   eval_time=r.eval_time, factor_time=r.factor_time,
                   status_message=r.status_message, z=np.asarray(r.z).ravel())


@dataclass
class ShootingSolution(TrajectorySolution):
    """What both MPCC shooting front-ends return: a state and a control trajectory."""

    state_trajectory: Optional[np.ndarray] = None    # nx x (horizon + 1)
    control_trajectory: Optional[np.ndarray] = None  # nu x horizon


@dataclass
class LCPSolution(TrajectorySolution):
    config_trajectory: Optional[np.ndarray] = None    # n_qpos x (horizon + 1)
    command_trajectory: Optional[np.ndarray] = None   # n_cmd  x horizon
    lambda_trajectory: Optional[np.ndarray] = None    # n_lam  x horizon
    velocity_trajectory: Optional[np.ndarray] = None  # n_qvel x horizon
    first_command: Optional[np.ndarray] = None


# ------------------------------------------------------------------- front-ends --


class _MPCCShootingSolver:
    """Common half of the two MPCC front-ends.

    Both wrap the task in a stage, build a subproblem, write ``x_goal``/``x_0``
    into the parameter buffer at the offsets the builder chose, solve, and slice z
    back into trajectories. Only the transcription differs, so only the parts that
    differ are left to the subclasses. Callers never see an offset or a raw z.
    """

    #: Set by the subclasses; picks the transcription.
    _build = staticmethod(build_multiple_shooting)

    def __init__(self, problem: MPCCProblem, stage_factory=MPCCStage):
        self.problem = problem
        # push_circle overrides only the terminal cost row, so the front-ends take
        # a stage factory rather than hard-coding MPCCStage.
        self.stage_factory = stage_factory

    def _assemble(self, config):
        """Build the subproblem and write the per-solve parameters into it.

        Returns the runtime parameter vector as well as the subproblem: single
        shooting has to roll the state out with the same ``p`` the solver saw, and
        re-deriving it at the call site is exactly the kind of duplicated offset
        arithmetic this front-end exists to remove.
        """
        stage = self.stage_factory(self.problem, config)
        layout = type(self)._build(stage, config)
        sub = layout.subproblem

        goal = np.asarray(config.x_goal, dtype=float).ravel()
        n_rt = stage.runtime_param_dim
        if n_rt and goal.size != n_rt:
            raise ValueError(
                f"{type(self).__name__}: this stage takes a {n_rt}-element runtime "
                f"parameter but config.x_goal has {goal.size}. Stages whose runtime "
                "parameter is not the goal need their own front-end.")
        p_value = goal if n_rt else np.zeros(1)
        sub.set_param_value(layout.off_p, p_value)
        sub.set_param_value(layout.off_x0, np.asarray(config.x_0, dtype=float).ravel())
        return stage, sub, p_value

    def _default_state_guess(self, config) -> np.ndarray:
        """Constant-start replication or a straight line to the goal, per the config."""
        nx = self.problem.state_dim
        x_0 = np.asarray(config.x_0, dtype=float).ravel()
        x_goal = np.asarray(config.x_goal, dtype=float).ravel()
        x_init = np.empty((nx, config.horizon + 1))
        for k in range(config.horizon + 1):
            if config.use_constant_state_init:
                x_init[:, k] = x_0
            else:
                a = k / config.horizon
                x_init[:, k] = (1.0 - a) * x_0 + a * x_goal
        return x_init


class MultipleShootingSolver(_MPCCShootingSolver):
    """Multiple shooting: X and U are both free, dynamics are defect equalities.

        solution = MultipleShootingSolver(BoxPushing()).solve(config)
        solution.state_trajectory    # nx x (horizon + 1)
    """

    _build = staticmethod(build_multiple_shooting)

    def solve(self, config) -> ShootingSolution:
        u_init = np.zeros((self.problem.control_dim, config.horizon))
        return self.solve_with_initial_guess(config, self._default_state_guess(config), u_init)

    def solve_with_initial_guess(self, config, x_init: np.ndarray,
                                 u_init: np.ndarray) -> ShootingSolution:
        nx, nu = self.problem.state_dim, self.problem.control_dim
        horizon = config.horizon
        nx_total = nx * (horizon + 1)

        _, sub, _ = self._assemble(config)

        z = np.empty(sub.num_opt)
        for k in range(horizon + 1):
            z[k * nx:(k + 1) * nx] = x_init[:, k]
        for k in range(horizon):
            z[nx_total + k * nu:nx_total + (k + 1) * nu] = u_init[:, k]

        result = _core.Solver().solve(sub, config, z)

        sol = ShootingSolution._from_result(result)
        zz = sol.z
        sol.state_trajectory = zz[:nx_total].reshape(horizon + 1, nx).T.copy()
        sol.control_trajectory = zz[nx_total:nx_total + nu * horizon].reshape(horizon, nu).T.copy()
        return sol


class SingleShootingSolver(_MPCCShootingSolver):
    """Single shooting: only U is free; the state is a function of it.

    Returns the same :class:`ShootingSolution` multiple shooting does, so the two
    transcriptions are interchangeable at the call site. The state trajectory is
    reconstructed by rolling ``stage.step`` forward from ``x_0`` -- the *same* map
    the builder inlined symbolically, taken from the stage rather than re-derived,
    so the reported states are the ones the solver actually constrained.
    """

    _build = staticmethod(build_single_shooting)

    def solve(self, config) -> ShootingSolution:
        return self.solve_with_initial_guess(
            config, np.zeros((self.problem.control_dim, config.horizon)))

    def solve_with_initial_guess(self, config, u_init: np.ndarray) -> ShootingSolution:
        nx, nu = self.problem.state_dim, self.problem.control_dim
        horizon = config.horizon

        stage, sub, p_value = self._assemble(config)
        result = _core.Solver().solve(sub, config, np.asarray(u_init, dtype=float).T.ravel())

        sol = ShootingSolution._from_result(result)
        control = sol.z[:nu * horizon].reshape(horizon, nu).T.copy()
        sol.control_trajectory = control

        x_sym = ca.SX.sym("x", nx)
        u_sym = ca.SX.sym("u", nu)
        p_sym = ca.SX.sym("p", p_value.size)
        step = ca.Function("step", [x_sym, u_sym, p_sym], [stage.step(x_sym, u_sym, p_sym)])

        state = np.empty((nx, horizon + 1))
        state[:, 0] = np.asarray(config.x_0, dtype=float).ravel()
        for k in range(horizon):
            state[:, k + 1] = np.asarray(step(state[:, k], control[:, k], p_value)).ravel()
        sol.state_trajectory = state
        return sol


class LCPSingleShootingSolver:
    """Single-shooting front-end for LCP contact tasks.

    The subproblem is built once per horizon and reused: for receding-horizon use
    the contact data (``phi``, ``J``) and targets are per-solve *parameters*, not
    part of the structure, so nothing symbolic is rebuilt between MPC steps. The
    ``_core.Solver`` instance is kept for the same reason -- it holds the Jacobian
    sparsity maps and the saddle symbolic factorisation.
    """

    def __init__(self, problem: LCPProblem):
        self.problem = problem
        self._layout: Optional[ShootingLayout] = None
        self._solver = _core.Solver()
        self._horizon_built = -1
        self.n_qpos = problem.config_dim
        self.n_qvel = problem.velocity_dim
        self.n_cmd = problem.command_dim
        self.n_lam = problem.max_contacts * 4
        self.h = problem.time_step
        self.vars_per_step = self.n_cmd + self.n_lam + self.n_qvel

    def _ensure_built(self, config) -> None:
        if self._horizon_built == config.horizon:
            return
        stage = LCPStage(self.problem, config)
        self._layout = build_single_shooting(stage, config)
        self._horizon_built = config.horizon

    def solve(self, config, q0, phi_vec, jac_mat, target_p, target_q,
              cmd_init=None, lam_init=None, vel_init=None) -> LCPSolution:
        self._ensure_built(config)
        layout = self._layout
        sub = layout.subproblem
        horizon = config.horizon

        q0 = np.asarray(q0, dtype=float).ravel()
        phi_vec = np.asarray(phi_vec, dtype=float).ravel()
        jac_mat = np.asarray(jac_mat, dtype=float)
        if cmd_init is None:
            cmd_init = np.zeros((self.n_cmd, horizon))
        if lam_init is None:
            lam_init = np.zeros((self.n_lam, horizon))
        if vel_init is None:
            vel_init = np.zeros((self.n_qvel, horizon))

        # p = [target_p(3), target_q(4), phi(n_lam), vec(J)(n_lam*n_qvel)].
        # column-major vec(J), matching the Eigen default the C++ front-end writes.
        p = np.concatenate([np.asarray(target_p, dtype=float).ravel(),
                            np.asarray(target_q, dtype=float).ravel(), phi_vec,
                            jac_mat.reshape(-1, order="F")])
        sub.set_param_value(layout.off_p, p)
        sub.set_param_value(layout.off_x0, q0)
        # Every solve starts from a clean AuLa state; an MPC step must not inherit
        # the previous step's multipliers or its escalated penalties.
        sub.reset_aula_state()

        z = np.empty(sub.num_opt)
        for k in range(horizon):
            o = k * self.vars_per_step
            z[o:o + self.n_cmd] = cmd_init[:, k]
            z[o + self.n_cmd:o + self.n_cmd + self.n_lam] = lam_init[:, k]
            z[o + self.n_cmd + self.n_lam:o + self.vars_per_step] = vel_init[:, k]

        result = self._solver.solve(sub, config, z)

        sol = LCPSolution._from_result(result)
        zz = sol.z
        cmd = np.empty((self.n_cmd, horizon))
        lam = np.empty((self.n_lam, horizon))
        vel = np.empty((self.n_qvel, horizon))
        for k in range(horizon):
            o = k * self.vars_per_step
            cmd[:, k] = zz[o:o + self.n_cmd]
            lam[:, k] = zz[o + self.n_cmd:o + self.n_cmd + self.n_lam]
            vel[:, k] = zz[o + self.n_cmd + self.n_lam:o + self.vars_per_step]
        sol.command_trajectory = cmd
        sol.lambda_trajectory = lam
        sol.velocity_trajectory = vel

        q_traj = np.empty((self.n_qpos, horizon + 1))
        q_traj[:, 0] = q0
        for k in range(horizon):
            q_traj[:, k + 1] = self.problem.integrate_state(q_traj[:, k], vel[:, k], self.h)
        sol.config_trajectory = q_traj
        sol.first_command = cmd[:, 0].copy()
        return sol
