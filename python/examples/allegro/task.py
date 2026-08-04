"""Allegro hand in-hand manipulation, as a single-shooting LCP task.

    q   = [obj_pos(3), obj_quat(4), robot_qpos(16)]                    (23)
    v   = [obj_lin_vel(3), obj_ang_vel(3), robot_vel(16)]              (22)
    cmd = joint-angle increment for the 16-DOF hand                    (16)

The contact set is not modelled symbolically: MuJoCo supplies the gap ``phi`` and
the linearised-cone Jacobian ``J`` at the current configuration, and they enter
the subproblem as runtime parameters (see ``sim.py``, next door). What is
symbolic is everything that has to be differentiated -- the force balance
``Q v - b/h - J' lambda/h = 0``, the complementarity ``lambda ⊥ (J v + phi/h)``,
the quaternion-aware kinematics, and the fingertip forward kinematics in the cost.

The fingertip FK is written out here rather than queried from MuJoCo because the
cost needs it *symbolically*, as a function of the decision variables, not as a
number at the current state.

Port of ``penalty_solver/src/allegro_lcp_problem.cpp``.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import casadi as ca
import numpy as np

from impact import AulaConfig
from impact.stage import LCPProblem

from ..common import Result, trajectory_io

__all__ = ["AllegroLCPProblem", "AllegroParameters", "allegro_solver_config", "config",
           "model_path", "solve"]


# ---------------------------------------------------------- homogeneous transforms --


def _ttmat(x: float, y: float, z: float) -> ca.SX:
    T = ca.SX.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def _quattmat(w: float, x: float, y: float, z: float) -> ca.SX:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    T = ca.SX.zeros(4, 4)
    T[0, 0] = 1 - 2 * (y * y + z * z)
    T[0, 1] = 2 * (x * y - w * z)
    T[0, 2] = 2 * (x * z + w * y)
    T[1, 0] = 2 * (x * y + w * z)
    T[1, 1] = 1 - 2 * (x * x + z * z)
    T[1, 2] = 2 * (y * z - w * x)
    T[2, 0] = 2 * (x * z - w * y)
    T[2, 1] = 2 * (y * z + w * x)
    T[2, 2] = 1 - 2 * (x * x + y * y)
    T[3, 3] = 1
    return T


def _rxtmat(a) -> ca.SX:
    T = ca.SX.eye(4)
    T[1, 1], T[1, 2] = ca.cos(a), -ca.sin(a)
    T[2, 1], T[2, 2] = ca.sin(a), ca.cos(a)
    return T


def _rytmat(a) -> ca.SX:
    T = ca.SX.eye(4)
    T[0, 0], T[0, 2] = ca.cos(a), ca.sin(a)
    T[2, 0], T[2, 2] = -ca.sin(a), ca.cos(a)
    return T


def _rztmat(a) -> ca.SX:
    T = ca.SX.eye(4)
    T[0, 0], T[0, 1] = ca.cos(a), -ca.sin(a)
    T[1, 0], T[1, 1] = ca.sin(a), ca.cos(a)
    return T


def _pos(T: ca.SX) -> ca.SX:
    return ca.vertcat(T[0, 3], T[1, 3], T[2, 3])


def _chain(*mats: ca.SX) -> ca.SX:
    out = mats[0]
    for m in mats[1:]:
        out = ca.mtimes(out, m)
    return out


@dataclass
class AllegroParameters:
    n_qpos: int = 23
    n_qvel: int = 22
    n_cmd: int = 16
    max_contacts: int = 20
    h: float = 0.1

    obj_mass: float = 0.01
    gravity: Tuple[float, ...] = (0.0, 0.0, -9.8, 0.0, 0.0, 0.0)
    #: Object "inertia": 50*I translational, 0.1*I rotational. These are
    #: quasi-dynamic conditioning weights, not the physical inertia.
    obj_inertia_linear: float = 50.0
    obj_inertia_angular: float = 0.1
    robot_stiffness: float = 1.0

    # Cost weights. The universal set from the 17-object x 10-seed sweep; the
    # earlier per-object schedule was removed because one grid-searched set beat
    # it on overall success rate.
    position_cost_weight: float = 0.0     # contact-based control: no pose tracking
    quaternion_cost_weight: float = 0.0
    contact_cost_weight: float = 1.0
    grasp_closure_weight: float = 0.0
    control_cost_weight: float = 0.2      # retuned 0.1 -> 0.2, +stability
    velocity_penalty: float = 0.4         # retuned 0.1 -> 0.4, +stability
    final_cost_multiplier: float = 10.0
    final_position_weight: float = 100.0
    final_quaternion_weight: float = 9.0
    cmd_bound: float = 0.1


class AllegroLCPProblem(LCPProblem):
    def __init__(self, params: Optional[AllegroParameters] = None):
        self.params = params if params is not None else AllegroParameters()

    config_dim = property(lambda self: self.params.n_qpos)
    velocity_dim = property(lambda self: self.params.n_qvel)
    command_dim = property(lambda self: self.params.n_cmd)
    max_contacts = property(lambda self: self.params.max_contacts)
    time_step = property(lambda self: self.params.h)

    control_cost_weight = property(lambda self: self.params.control_cost_weight)
    contact_cost_weight = property(lambda self: self.params.contact_cost_weight)
    grasp_closure_weight = property(lambda self: self.params.grasp_closure_weight)
    velocity_penalty = property(lambda self: self.params.velocity_penalty)
    position_cost_weight = property(lambda self: self.params.position_cost_weight)
    orientation_cost_weight = property(lambda self: self.params.quaternion_cost_weight)
    final_cost_multiplier = property(lambda self: self.params.final_cost_multiplier)
    final_position_weight = property(lambda self: self.params.final_position_weight)
    final_orientation_weight = property(lambda self: self.params.final_quaternion_weight)
    num_fingertips = property(lambda self: 4)

    def inertia_matrix(self) -> np.ndarray:
        p = self.params
        Q = np.zeros((p.n_qvel, p.n_qvel))
        Q[:3, :3] = p.obj_inertia_linear * np.eye(3)
        Q[3:6, 3:6] = p.obj_inertia_angular * np.eye(3)
        Q[6:, 6:] = self.robot_stiffness()
        return Q

    def robot_stiffness(self) -> np.ndarray:
        return self.params.robot_stiffness * np.eye(self.params.n_cmd)

    def gravity_bias(self) -> np.ndarray:
        # Only the object's 6 rows carry gravity; the robot's rows get
        # stiffness*cmd, which the stage adds symbolically from the decision
        # variables rather than baking in here.
        b = np.zeros(self.params.n_qvel)
        b[:6] = self.params.obj_mass * np.asarray(self.params.gravity, dtype=float)
        return b

    def fingertip_positions_sx(self, q: ca.SX) -> ca.SX:
        """Forward kinematics of the four fingertips from ``q``.

        Joint blocks: index 7:11, middle 11:15, ring 15:19, thumb 19:23.
        """
        ff, mf, rf, th = q[7:11], q[11:15], q[15:19], q[19:23]
        sq2 = math.sqrt(2.0)
        t_palm = _quattmat(0.0, 1.0 / sq2, 0.0, 1.0 / sq2)

        ff_base = _chain(t_palm, _ttmat(0, 0.0435, -0.001542),
                         _quattmat(0.999048, -0.0436194, 0, 0))
        ff_tip = _chain(ff_base, _rztmat(ff[0]), _ttmat(0, 0, 0.0164),
                        _rytmat(ff[1]), _ttmat(0, 0, 0.054),
                        _rytmat(ff[2]), _ttmat(0, 0, 0.0384),
                        _rytmat(ff[3]), _ttmat(0, 0, 0.0384))

        mf_base = _chain(t_palm, _ttmat(0, 0, 0.0007))
        mf_tip = _chain(mf_base, _rztmat(mf[0]), _ttmat(0, 0, 0.0164),
                        _rytmat(mf[1]), _ttmat(0, 0, 0.054),
                        _rytmat(mf[2]), _ttmat(0, 0, 0.0384),
                        _rytmat(mf[3]), _ttmat(0, 0, 0.0384))

        rf_base = _chain(t_palm, _ttmat(0, -0.0435, -0.001542),
                         _quattmat(0.999048, 0.0436194, 0, 0))
        rf_tip = _chain(rf_base, _rztmat(rf[0]), _ttmat(0, 0, 0.0164),
                        _rytmat(rf[1]), _ttmat(0, 0, 0.054),
                        _rytmat(rf[2]), _ttmat(0, 0, 0.0384),
                        _rytmat(rf[3]), _ttmat(0, 0, 0.0384))

        th_base = _chain(t_palm, _ttmat(-0.0182, 0.019333, -0.045987),
                         _quattmat(0.477714, -0.521334, -0.521334, -0.477714))
        th_tip = _chain(th_base, _rxtmat(-th[0]), _ttmat(-0.027, 0.005, 0.0399),
                        _rztmat(th[1]), _ttmat(0, 0, 0.0177),
                        _rytmat(th[2]), _ttmat(0, 0, 0.0514),
                        _rytmat(th[3]), _ttmat(0, 0, 0.054))

        return ca.vertcat(_pos(ff_tip), _pos(mf_tip), _pos(rf_tip), _pos(th_tip))

    def integrate_state(self, q: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        """Roll the configuration forward, quaternion included.

        The quaternion is *not* renormalised, matching the optimizer's own
        kinematics -- renormalising here would make the rolled-out trajectory
        disagree with the one the solver constrained.
        """
        p = self.params
        q = np.asarray(q, dtype=float)
        vel = np.asarray(vel, dtype=float)
        q_next = np.empty(p.n_qpos)
        q_next[0:3] = q[0:3] + dt * vel[0:3]

        quat = q[3:7]
        omega = vel[3:6]
        H_T = np.array([[-quat[1], -quat[2], -quat[3]],
                        [quat[0], -quat[3], quat[2]],
                        [quat[3], quat[0], -quat[1]],
                        [-quat[2], quat[1], quat[0]]])
        q_next[3:7] = quat + 0.5 * dt * H_T @ omega
        q_next[7:7 + p.n_cmd] = q[7:7 + p.n_cmd] + dt * vel[6:6 + p.n_cmd]
        return q_next


def allegro_solver_config(params: Optional[AllegroParameters] = None,
                          horizon: int = 4) -> AulaConfig:
    """The universal solver settings from ``allegro_impact_single.cpp``."""
    p = params if params is not None else AllegroParameters()
    c = AulaConfig()
    c.horizon = horizon
    c.rho_dynamics_init = 1.0
    c.rho_comp_init = 1.0
    c.rho_ineq_init = 1.0   # command bounds live on the inequality channel
    c.rho_max = 1e3
    c.rho_scale = 5.0
    c.dynamics_scale = 25.0
    c.comp_scale = 1.0
    c.max_outer_iters = 10
    c.outer_tol_h = 1e-3
    c.outer_tol_comp = 1e-3
    c.outer_tol_g = 1e-3
    c.max_inner_iters = 5
    c.newton_max_iter = 30
    c.newton_step_tol = 1e-5
    c.newton_tol = 1e-5
    c.newton_regularization = 1e-6
    c.print_level = 0
    # Command bounds as an AuLa inequality channel rather than box bounds.
    c.use_cmd_bounds = True
    c.cmd_lower = np.full(p.n_cmd, -p.cmd_bound)
    c.cmd_upper = np.full(p.n_cmd, p.cmd_bound)
    return c


#: The same settings under the name every example's ``main.py`` calls.
config = allegro_solver_config


def model_path(object: str = "cube", model: Optional[str] = None):
    """``resources/xmls/env_allegro_<object>.xml``, or an explicit path."""
    if model:
        path = pathlib.Path(model)
    else:
        root = trajectory_io.CHECKOUT_ROOT
        if root is None:
            raise SystemExit("model= is required outside a source checkout "
                             "(the MuJoCo XMLs live in the repository's resources/)")
        path = root / "resources" / "xmls" / f"env_allegro_{object}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
    return path


def solve(cfg: Optional[AulaConfig] = None, *, object: str = "cube",
          model: Optional[str] = None, max_steps: int = 100, frame_skip: int = 50,
          mu: float = 0.5, target_yaw_deg: float = 90.0, viewer: bool = False,
          video: Optional[str] = None, video_stride: int = 1, camera: str = "demo-cam",
          width: int = 640, height: int = 480, fps: int = 20) -> Result:
    """Run the receding-horizon MPC loop against MuJoCo.

    The only closed-loop example here, so the "solution" it reports is the whole
    rollout: ``iterations`` counts executed MPC steps and ``solve_time`` is their
    sum, not the last solve's. The subproblem is built once -- ``phi`` and the
    contact Jacobian are per-solve *parameters* -- which is what makes the loop
    run at a control rate at all.
    """
    import os

    path = model_path(object, model)
    if max_steps <= 0 or video_stride <= 0:
        raise ValueError("max_steps and video_stride must be positive")
    # MuJoCo picks its GL backend at import time, so this has to precede it.
    if video and not viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")

    from impact.shooting import LCPSingleShootingSolver

    from .sim import AllegroSimulator, quaternion_from_rpy
    from .viz import GifRecorder

    cfg = config() if cfg is None else cfg
    initial_robot = np.array([0.125, 1.13, 1.45, 1.24, -0.02, 0.445, 1.17, 1.5,
                              -0.459, 1.54, 1.11, 1.23, 0.638, 1.85, 1.5, 1.26])
    initial_object = np.array([0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0])
    target_position = np.array([0.0, 0.0, 0.05])
    target_quaternion = quaternion_from_rpy(np.deg2rad(target_yaw_deg))

    sim = AllegroSimulator(str(path), frame_skip=frame_skip, mu=mu)
    sim.reset(initial_robot, initial_object)
    sim.set_goal_pose(target_position, target_quaternion)
    if viewer:
        sim.init_rendering()
    recorder = (GifRecorder(sim.model, video, camera=camera, width=width, height=height,
                            fps=fps) if video else None)
    if recorder:
        recorder.capture(sim.data)

    solver = LCPSingleShootingSolver(AllegroLCPProblem(AllegroParameters()))
    solution, solve_times, commands = None, [], []
    states = [sim.state()]
    all_converged = True

    try:
        for step in range(max_steps):
            if viewer and sim.should_close():
                print("viewer closed by user")
                break
            contacts = sim.detect_contacts()
            args = (cfg, sim.state(), contacts.phi, contacts.jac,
                    target_position, target_quaternion)
            if solution is None:
                solution = solver.solve(*args)
            else:
                # Shift the previous plan one step and repeat its tail. The dual
                # state is reset inside solve(): q0 changes every step, so the
                # last step's multipliers describe a different problem.
                def shift(a):
                    return np.hstack([a[:, 1:], a[:, -1:]])

                solution = solver.solve(*args, shift(solution.command_trajectory),
                                        shift(solution.lambda_trajectory),
                                        shift(solution.velocity_trajectory))
            command = solution.first_command.copy()
            all_converged &= bool(solution.converged)
            solve_times.append(solution.solve_time)
            commands.append(command)
            sim.step(command, render_every=5 if viewer else 0)
            if viewer:
                sim.render()
            states.append(sim.state())
            if recorder and (step + 1) % video_stride == 0:
                recorder.capture(sim.data)
            if step % 10 == 0:
                err = float(np.linalg.norm(states[-1][:3] - target_position))
                print(f"  step {step:4d}: solve {solution.solve_time * 1000:6.1f} ms  "
                      f"pos_err {err:.4f}  comp {solution.complementarity_violation:.2e}")
    finally:
        if viewer:
            sim.close_rendering()
        if recorder:
            try:
                recorder.save()
            finally:
                recorder.close()

    if not commands:
        raise RuntimeError("no MPC step was executed")

    state = np.asarray(states).T
    control = np.asarray(commands).T
    final = state[:, -1]
    quaternion_error = float(min(np.linalg.norm(final[3:7] - target_quaternion),
                                 np.linalg.norm(final[3:7] + target_quaternion)))
    mean_solve = float(np.mean(solve_times))
    return Result(
        name="allegro", solution=solution, planner=f"mpc_{object}",
        state=state, control=control, start=state[:, 0],
        goal=np.concatenate([target_position, target_quaternion]),
        goal_error=float(np.linalg.norm(final[:3] - target_position)),
        rollup_iterations=len(commands), rollup_solve_time=float(np.sum(solve_times)),
        rollup_success=bool(all_converged and np.all(np.isfinite(final))),
        file=dict(title=f"Allegro In-Hand {object} Reorientation",
                  preamble=[("Target Yaw Degrees", [target_yaw_deg])],
                  start_label="object_pose(7), allegro_joints(16)",
                  goal_label="object_position(3), object_quaternion(4)",
                  state_label="object_pose(7), allegro_joints(16)",
                  control_label="joint_increment(16)"),
        artifacts=(pathlib.Path(video),) if video else (),
        rows=(("steps executed", str(len(commands))),
              ("mean solve", f"{mean_solve * 1000:.3f} ms"),
              ("MPC frequency", f"{1.0 / mean_solve:.2f} Hz"),
              ("final object position", str(np.round(final[:3], 5))),
              ("quaternion error", f"{quaternion_error:.3e}")))
