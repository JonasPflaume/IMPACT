"""MuJoCo simulation and contact extraction for the contact-implicit tasks.

The single-shooting LCP transcription does not model geometry symbolically. It
takes the contact set as *runtime data*: a gap vector ``phi`` and a contact
Jacobian ``J``, frozen at the current configuration and written into the
subproblem's parameter buffer each MPC step. This module is where that data comes
from.

Contact extraction (:func:`detect_contacts`) turns MuJoCo's contact list into the
``(phi, J)`` the transcription expects. Two details in it are easy to get wrong
and are load-bearing:

* **The linearised friction cone.** Each MuJoCo contact becomes *four* rows, not
  one. The contact frame is extended to ``[n, t1, t2, -t1, -t2]`` and each of the
  four tangential directions is folded into the normal row as ``J_n + mu*J_t``,
  so a nonnegative multiplier on each row is exactly a force inside the
  pyramidal friction cone. This is why ``n_lam == 4 * max_contacts`` everywhere.

* **The sign of the relative Jacobian.** ``J`` must express motion *of the object
  relative to its contact partner*, so the difference ``J2 - J1`` is negated when
  the object happens to be the first geom in the pair. Getting this backwards
  produces a solve that pushes contacts closed instead of open, and it does not
  look like a crash.

Inactive rows are padded rather than dropped, so the parameter buffer keeps a
fixed size across MPC steps and the subproblem never has to be rebuilt.

Ports of ``simulation/contactLCP.cpp``, ``simulation/fingertips_simulator.cpp``
and ``simulation/allegro_simulator.cpp``. Rendering uses MuJoCo's own passive
viewer instead of a hand-written GLFW loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ContactResult", "ContactParams", "detect_contacts", "MuJoCoSimulator",
           "AllegroSimulator", "FingertipsSimulator", "quaternion_from_rpy"]


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise ImportError(
            "MuJoCo is required for the simulation tasks. Install it with "
            "`pip install impact-solver[sim]` or `pip install mujoco`.") from exc
    return __import__("mujoco")


@dataclass
class ContactResult:
    """Frozen contact data for one MPC step."""

    phi: np.ndarray  # (4 * max_contacts,)  gap per linearised-cone row
    jac: np.ndarray  # (4 * max_contacts, n_qvel)


@dataclass
class ContactParams:
    object_geoms: Sequence[str] = ("obj",)
    mu: float = 0.5
    max_contacts: int = 8
    n_qvel: int = 15
    #: Value written into rows with no contact. Nonzero and small: a large gap
    #: makes those rows inactive but also flattens their contribution, while zero
    #: would read as touching.
    inactive_gap: float = 1.0
    #: MuJoCo orders velocity DOFs by joint; the transcription wants
    #: ``[object(6), robot(...)]``. When the model puts the object last, set this
    #: to the object's DOF count to rotate its block to the front.
    object_dofs_at_end: int = 0


def detect_contacts(model, data, params: ContactParams) -> ContactResult:
    """Extract ``(phi, J)`` for the linearised friction cone at the current state."""
    mujoco = _require_mujoco()
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)

    n_qvel = params.n_qvel
    phi = np.full(4 * params.max_contacts, params.inactive_gap)
    jac = np.zeros((4 * params.max_contacts, n_qvel))

    object_geoms = set(params.object_geoms)
    rows: List[Tuple[float, np.ndarray]] = []

    for i in range(data.ncon):
        contact = data.contact[i]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        geom1_is_object = name1 in object_geoms
        geom2_is_object = name2 in object_geoms
        if not (geom1_is_object or geom2_is_object):
            continue

        # MuJoCo reports the full penetration depth between the two geoms; each
        # side owns half of it.
        gap = float(contact.dist) * 0.5

        # contact.frame is row-major 3x3 whose *rows* are (n, t1, t2).
        frame = np.array(contact.frame, dtype=float).reshape(3, 3).T
        # [n, t1, t2, -t1, -t2]: the four signed tangent directions of the
        # pyramidal cone, sharing one normal.
        frame_pmd = np.zeros((3, 5))
        frame_pmd[:, :3] = frame
        frame_pmd[:, 3:] = -frame[:, 1:3]

        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        jacp1 = np.zeros((3, model.nv))
        jacp2 = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp1, None, contact.pos, body1)
        mujoco.mj_jac(model, data, jacp2, None, contact.pos, body2)

        con_jac1 = frame_pmd.T @ jacp1[:, :n_qvel]
        con_jac2 = frame_pmd.T @ jacp2[:, :n_qvel]
        # Motion of the object relative to its partner; the sign follows which
        # geom the object is.
        rel = -(con_jac2 - con_jac1) if geom1_is_object else (con_jac2 - con_jac1)

        normal_row = rel[0, :]
        friction_rows = rel[1:5, :]
        cone = normal_row[None, :] + params.mu * friction_rows  # (4, n_qvel)
        rows.append((gap, cone))

    for i, (gap, cone) in enumerate(rows[:params.max_contacts]):
        phi[4 * i:4 * i + 4] = gap
        jac[4 * i:4 * i + 4, :] = cone

    if params.object_dofs_at_end:
        # Rotate [robot..., object] into [object, robot...].
        k = params.object_dofs_at_end
        jac = np.concatenate([jac[:, n_qvel - k:], jac[:, :n_qvel - k]], axis=1)

    return ContactResult(phi=phi, jac=jac)


def quaternion_from_rpy(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """(w, x, y, z) quaternion from yaw/pitch/roll, matching ``rpyToQuaternion``."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     sy * cp * sr + cy * sp * cr,
                     sy * cp * cr - cy * sp * sr])


class MuJoCoSimulator:
    """A MuJoCo model driven by joint-position targets.

    ``step`` applies an action as a joint-angle *increment* and holds the
    resulting target for ``frame_skip`` physics steps, which is the position
    controller the tasks assume.

    Subclasses supply the qpos layout: MuJoCo's order is whatever the XML
    declares, while the transcription always wants ``[obj_pos(3), obj_quat(4),
    robot_qpos(...)]``.
    """

    #: Slice of MuJoCo qpos holding the robot joints.
    robot_qpos_slice: slice = slice(0, 0)
    #: Slice of MuJoCo qpos holding the object pose (3 position + 4 quaternion).
    object_qpos_slice: slice = slice(0, 0)
    #: Index of the object's first velocity DOF. Stated rather than derived from
    #: the qpos slices: a free joint contributes 7 qpos but 6 qvel, so the two
    #: layouts are not the same map and inferring one from the other is only
    #: right by coincidence.
    object_qvel_start: int = 0

    def __init__(self, model_path: str, *, frame_skip: int = 50,
                 contact_params: Optional[ContactParams] = None):
        mujoco = _require_mujoco()
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.contact_params = contact_params or ContactParams()
        self._viewer = None

    # -- state --------------------------------------------------------------
    def reset(self, robot_qpos: np.ndarray, object_qpos: np.ndarray) -> None:
        self.data.qpos[self.robot_qpos_slice] = np.asarray(robot_qpos, dtype=float)
        self.data.qpos[self.object_qpos_slice] = np.asarray(object_qpos, dtype=float)
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def set_goal_pose(self, position: np.ndarray, quaternion: np.ndarray,
                      *, body_name: str = "goal") -> None:
        """Set the fixed body's displayed goal pose.

        MuJoCo quaternions use ``(w, x, y, z)``, matching the task and solver.
        Some Allegro XMLs put the visible goal geom 0.15 m above this body so
        that it acts as an unobstructed orientation reference; this method
        preserves that model-defined offset.
        """
        position = np.asarray(position, dtype=float).ravel()
        quaternion = np.asarray(quaternion, dtype=float).ravel()
        if position.shape != (3,):
            raise ValueError(f"goal position must have shape (3,), got {position.shape}")
        if quaternion.shape != (4,):
            raise ValueError(f"goal quaternion must have shape (4,), got {quaternion.shape}")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("goal pose must contain only finite values")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= np.finfo(float).eps:
            raise ValueError("goal quaternion must have nonzero norm")

        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model has no body named {body_name!r}")
        self.model.body_pos[body_id] = position
        self.model.body_quat[body_id] = quaternion / quaternion_norm
        self._mujoco.mj_forward(self.model, self.data)

    def state(self) -> np.ndarray:
        """``[obj_pos(3), obj_quat(4), robot_qpos(...)]``, the solver's q layout."""
        return np.concatenate([np.asarray(self.data.qpos[self.object_qpos_slice], dtype=float),
                               np.asarray(self.data.qpos[self.robot_qpos_slice], dtype=float)])

    def set_state(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self.data.qpos[self.object_qpos_slice] = q[:7]
        self.data.qpos[self.robot_qpos_slice] = q[7:]
        self._mujoco.mj_forward(self.model, self.data)

    def object_velocity(self) -> np.ndarray:
        """Object linear + angular velocity (6,)."""
        s = self.object_qvel_start
        return np.asarray(self.data.qvel[s:s + 6], dtype=float)

    # -- dynamics -----------------------------------------------------------
    def step(self, action: np.ndarray, render_every: int = 0) -> None:
        """Apply a joint-angle increment and advance ``frame_skip`` physics steps."""
        action = np.asarray(action, dtype=float).ravel()
        current = np.asarray(self.data.qpos[self.robot_qpos_slice], dtype=float)
        target = current + action
        n_ctrl = min(target.size, self.model.nu)
        for i in range(self.frame_skip):
            self.data.ctrl[:n_ctrl] = target[:n_ctrl]
            self._mujoco.mj_step(self.model, self.data)
            if render_every and self._viewer is not None and (i + 1) % render_every == 0:
                self.render()

    def detect_contacts(self, q: Optional[np.ndarray] = None) -> ContactResult:
        """Contacts at ``q``, or at the current state when ``q`` is omitted.

        Evaluating at another configuration is done by moving MuJoCo there and
        putting it back afterwards, so this never disturbs the rollout.
        """
        if q is None:
            return detect_contacts(self.model, self.data, self.contact_params)
        saved_qpos = np.array(self.data.qpos, copy=True)
        saved_qvel = np.array(self.data.qvel, copy=True)
        try:
            self.set_state(q)
            return detect_contacts(self.model, self.data, self.contact_params)
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self._mujoco.mj_forward(self.model, self.data)

    # -- rendering ----------------------------------------------------------
    def init_rendering(self) -> None:
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def render(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def should_close(self) -> bool:
        return self._viewer is not None and not self._viewer.is_running()

    def close_rendering(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class AllegroSimulator(MuJoCoSimulator):
    """16-DOF Allegro hand manipulating a free object.

    MuJoCo qpos layout is ``[robot(16), object pos(3) quat(4)]``; the solver's is
    ``[obj_pos(3), obj_quat(4), robot(16)]``, hence the reordering here and the
    matching ``object_dofs_at_end`` in the contact Jacobian.
    """

    robot_qpos_slice = slice(0, 16)
    object_qpos_slice = slice(16, 23)
    object_qvel_start = 16

    def __init__(self, model_path: str, *, frame_skip: int = 50, mu: float = 0.5,
                 max_contacts: int = 20, object_geoms: Sequence[str] = ("obj",)):
        super().__init__(model_path, frame_skip=frame_skip,
                         contact_params=ContactParams(object_geoms=object_geoms, mu=mu,
                                                      max_contacts=max_contacts, n_qvel=22,
                                                      inactive_gap=0.01,
                                                      object_dofs_at_end=6))


class FingertipsSimulator(MuJoCoSimulator):
    """Three free-floating spherical fingertips manipulating a cube.

    MuJoCo qpos layout here is already ``[object pos(3) quat(4), fingertips(9)]``,
    so no reordering is needed.
    """

    robot_qpos_slice = slice(7, 16)
    object_qpos_slice = slice(0, 7)
    object_qvel_start = 0

    def __init__(self, model_path: str, *, frame_skip: int = 50, mu: float = 0.5,
                 max_contacts: int = 8, object_geoms: Sequence[str] = ("obj",)):
        super().__init__(model_path, frame_skip=frame_skip,
                         contact_params=ContactParams(object_geoms=object_geoms, mu=mu,
                                                      max_contacts=max_contacts, n_qvel=15,
                                                      inactive_gap=1.0))
