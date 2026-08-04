"""End-to-end checks: every task solves to its known answer.

These complement ``test_parity.py``, which proves the Python builders emit the
same problem as the C++ ones. This file checks the other half -- that solving
that problem still produces the right trajectory -- by asserting on outcomes
(objective, feasibility, where the object ends up) rather than on iteration
counts. Iteration counts are not a stable target: the same solver source
recompiled without ``-ffast-math`` moves them by ~1%, because these are nonconvex
solves running for hundreds of outer iterations.

Each task is imported from its own directory, which is the whole of its public
API: ``config()`` for the tuned settings and ``solve()`` for the result.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.box import task as box
from examples.cart_transporter import task as cart_transporter
from examples.push_circle import task as push_circle
from examples.push_t import task as push_t
from examples.toy import task as toy


def test_push_circle_goes_around_the_disk():
    """The pusher must orbit the disk rather than push it the wrong way.

    The scenario is built so the naive move is wrong: the disk's goal is where
    the pusher starts, so pushing from the initial contact drives the disk away.
    Reaching the goal is therefore evidence the solver escaped that minimum, and
    the pusher's angle relative to the disk has to sweep past 90 degrees to do it.
    """
    result = push_circle.solve(push_circle.config(horizon=100))

    assert result.converged
    assert result.complementarity_violation < 1e-5
    assert result.dynamics_violation < 1e-5
    assert result.goal_error < 5e-3

    # Non-penetration: the pusher stays outside the disk at every knot.
    state = result.state
    disk, pusher = state[:2, :], state[2:, :]
    assert np.linalg.norm(pusher - disk, axis=0).min() > push_circle.PushCircle().R - 1e-4

    angles = np.unwrap(np.arctan2(*(pusher - disk)[::-1, :]))
    assert np.abs(angles[-1] - angles[0]) > np.deg2rad(90), "pusher did not go around"


@pytest.mark.parametrize("mode", ["multiple", "single"])
def test_push_circle_solves_in_both_transcriptions(mode):
    result = push_circle.solve(push_circle.config(horizon=40), mode=mode)
    assert result.converged
    assert result.state.shape == (4, 41)
    assert result.planner.endswith("_single") == (mode == "single")


@pytest.mark.parametrize("module,goal,tol", [
    (box, [0.3, 0.2, 0.5], 5e-2),
    (push_t, [0.05, 0.05, 0.3], 5e-2),
    (cart_transporter, [1.0, 1.0, 0.0, 0.0], 5e-2),
])
def test_planar_tasks_reach_their_goal(module, goal, tol):
    result = module.solve(module.config(goal=goal))
    assert result.dynamics_violation < 1e-4
    assert result.goal_error < tol


def test_toy_task_reaches_the_analytic_optimum():
    result = toy.solve()
    assert result.converged
    assert result.objective_value == pytest.approx(toy.EXPECTED_OBJECTIVE, abs=1e-6)


@pytest.mark.slow
def test_allegro_mpc_step():
    """One MuJoCo contact query plus one solve, end to end."""
    mujoco = pytest.importorskip("mujoco", reason="pip install 'impact-solver[sim]'")

    from examples.allegro.sim import AllegroSimulator, quaternion_from_rpy
    from examples.allegro.task import AllegroLCPProblem, config, model_path
    from examples.common.trajectory_io import CHECKOUT_ROOT
    from impact import LCPSingleShootingSolver

    if CHECKOUT_ROOT is None:
        pytest.skip("MuJoCo models live in the repository's resources/")
    xml = model_path("cube")

    sim = AllegroSimulator(str(xml))
    sim.reset(np.array([0.125, 1.13, 1.45, 1.24, -0.02, 0.445, 1.17, 1.5,
                        -0.459, 1.54, 1.11, 1.23, 0.638, 1.85, 1.5, 1.26]),
              np.array([0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]))
    target_position = np.array([0.0, 0.0, 0.05])
    target_quaternion = quaternion_from_rpy(np.pi / 2)
    sim.set_goal_pose(target_position, target_quaternion)

    goal_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
    assert goal_id >= 0
    assert np.allclose(sim.model.body_pos[goal_id], target_position)
    assert np.allclose(sim.model.body_quat[goal_id], target_quaternion)

    contacts = sim.detect_contacts()
    # 20 contacts x 4 linearised-cone rows, against 22 velocity DOFs.
    assert contacts.phi.shape == (80,)
    assert contacts.jac.shape == (80, 22)
    assert np.all(np.isfinite(contacts.jac))

    settings = config(horizon=4)
    settings.print_level = 0
    sol = LCPSingleShootingSolver(AllegroLCPProblem()).solve(
        settings, sim.state(), contacts.phi, contacts.jac,
        target_position, target_quaternion)
    assert sol.first_command.shape == (16,)
    assert np.all(np.isfinite(sol.first_command))
    # The AuLa inequality channel holds the command inside its bound.
    assert np.max(np.abs(sol.first_command)) <= 0.1 + 1e-6
