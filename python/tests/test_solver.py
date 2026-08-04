"""The solver package on its own terms: no task imported anywhere in this file.

``impact`` is meant to be usable without the repository's examples, so these
tests build their MPCCs inline. If any of them ever needs ``examples``, something
task-specific has leaked back into the solver.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

import impact


def test_no_task_material_is_importable_from_the_solver():
    """The package exposes a solver, not a task library."""
    for gone in ("tasks", "viz", "sim", "presets", "trajectory_io", "cli", "baselines"):
        assert not hasattr(impact, gone), f"impact.{gone} should live in examples/"


def test_config_exposes_every_hyperparameter():
    """The config is the C++ struct, so its defaults cannot drift from the solver's."""
    values = impact.config_to_dict(impact.AulaConfig())
    # A representative field from each group the solver actually reads.
    for name in ("rho_max", "rho_scale", "penalty_decrease_ratio", "safeguard_factor",
                 "comp_scale", "outer_tol_comp", "inner_tol_init", "inner_tol_ramp_start",
                 "newton_tol", "newton_lambda_min",
                 "newton_max_damping_tries", "forcing_cap", "forcing_factor", "use_saddle",
                 "saddle_sigma_primal"):
        assert name in values, f"{name} is not reachable from Python"
    assert set(values) == set(impact.field_names())


def test_config_round_trips_and_rejects_typos():
    config = impact.apply_config(impact.AulaConfig(), rho_max=42.0, horizon=7,
                                 x_goal=[1.0, 2.0])
    assert config.rho_max == 42.0 and config.horizon == 7
    assert np.allclose(config.x_goal, [1.0, 2.0])
    assert impact.config_from_dict(impact.config_to_dict(config)).rho_max == 42.0

    # A silently dropped hyper-parameter still produces a plausible trajectory,
    # so a typo has to fail loudly -- and point at what was meant.
    with pytest.raises(AttributeError, match="rho_max"):
        impact.apply_config(impact.AulaConfig(), rho_maks=1.0)


def _toy_description():
    """min ||z - [0.8, 0.2, 0.25, 0.75]||^2 over two complementarity blocks."""
    target = np.array([0.8, 0.2, 0.25, 0.75])
    z = ca.SX.sym("z", 4)
    desc = impact.MPCCDescription(z=z, cost=z - ca.DM(target), cost_is_linear=True)
    desc.add_complementarity("axis_a", z[0], z[1],
                             impact.BlockOptions(scale=1.0, rho_init=1.0, tol=1e-8))
    desc.add_complementarity("axis_b", z[2], z[3],
                             impact.BlockOptions(scale=0.75, rho_init=2.0, tol=1e-8))
    return desc


def test_generic_mpcc_reaches_the_analytic_optimum():
    """Two separate complementarity groups, started from the biactive corner."""
    config = impact.AulaConfig()
    config.max_outer_iters = 300
    config.rho_scale = 1.5
    config.rho_max = 1e3
    config.outer_tol_h = config.outer_tol_comp = 1e-7
    config.print_level = 0

    built = impact.build_mpcc(_toy_description())
    r = impact.Solver().solve(built.subproblem, config, np.zeros(4))

    assert r.converged
    # Closest feasible point is [0.8, 0, 0, 0.75]; objective 0.2^2 + 0.25^2.
    assert r.objective_value == pytest.approx(0.1025, abs=1e-6)
    assert abs(r.z[0] * r.z[1]) < 1e-7
    assert abs(r.z[2] * r.z[3]) < 1e-7


def test_build_mpcc_rejects_malformed_blocks():
    z = ca.SX.sym("z", 2)
    desc = impact.MPCCDescription(z=z, cost=z, cost_is_linear=True)
    desc.add_complementarity("bad", z[0], z[0], impact.BlockOptions(scale=-1.0))
    with pytest.raises(ValueError, match="scale"):
        impact.build_mpcc(desc)


class _RestingMass(impact.MPCCProblem):
    """A unit mass resting on the ground: the smallest complementarity ODE.

    x = [height, velocity], u = [contact force]. The ground sits at height 0, so
    ``0 <= f (perp) height >= 0`` both switches the force on only in contact and,
    through ``height >= 0``, keeps the mass out of the ground.

    Nothing specifies the contact force, so recovering ``f == gravity`` is a real
    check on the complementarity rows rather than on the integrator.
    """

    state_dim = property(lambda self: 2)
    control_dim = property(lambda self: 1)
    comp_dim = property(lambda self: 1)
    time_step = property(lambda self: 0.05)

    def dynamics(self, x, u):
        return ca.vertcat(x[1], u[0] - 1.0)      # gravity pulls toward the ground

    def G(self, x, u):
        return ca.vertcat(u[0])

    def H(self, x, u):
        return ca.vertcat(x[0])


def _resting_config():
    config = impact.AulaConfig()
    config.horizon = 30
    config.x_0 = np.array([0.0, 0.0])
    config.x_goal = np.array([0.0, 0.0])
    config.final_cost_weight = 100.0
    config.stage_cost_weight = 1e-3
    config.stage_state_cost_weight = 1.0
    config.max_outer_iters = 600
    config.outer_tol_h = config.outer_tol_g = 1e-5
    config.outer_tol_comp = 1e-4
    # A toy still needs the two knobs every real task tunes. The complementarity
    # residual is the small quantity here, so it is conditioned up; and the inner
    # sweep has to be solved tightly enough that the outer loop is not measuring
    # its own stagnation tolerance. Left at library defaults this problem stalls
    # above its complementarity target -- which is the caveat the notebook makes
    # explicit, not a property of the transcriptions being compared.
    config.comp_scale = 10.0
    config.inner_tol_final = 1e-5
    config.print_level = 0
    return config


def test_both_shooting_transcriptions_agree_on_the_same_task():
    """Single and multiple shooting are interchangeable at the call site.

    They differ in what is a decision variable, not in what they describe, so on
    a problem both solve to tolerance they must recover the same physics -- and
    both must return the same ShootingSolution shape.
    """
    config = _resting_config()
    multiple = impact.MultipleShootingSolver(_RestingMass()).solve(config)
    single = impact.SingleShootingSolver(_RestingMass()).solve(config)

    for solution in (multiple, single):
        assert isinstance(solution, impact.ShootingSolution)
        assert solution.converged
        assert solution.state_trajectory.shape == (2, config.horizon + 1)
        assert solution.control_trajectory.shape == (1, config.horizon)
        assert solution.complementarity_violation < 1e-4
        # Non-penetration is the whole point of the complementarity row.
        assert solution.state_trajectory[0].min() > -2e-4
        # The contact force is an output: it has to come out as gravity. The
        # tolerance is the outer complementarity target above, not a free
        # parameter: the two are the same quantity seen from either leg.
        assert solution.control_trajectory[0] == pytest.approx(1.0, abs=5e-2)

    # Single shooting rolls the state out through the same `step` the builder
    # inlined, so its first knot is x_0 exactly rather than to fix_point tolerance.
    assert np.allclose(single.state_trajectory[:, 0], config.x_0)


def test_planner_tag_names_what_actually_ran():
    assert impact.planner_tag(impact.AulaConfig()) == "bcd_aula"


def test_stationarity_target_drags_the_inner_tolerances_with_it():
    """The trap documented in impact.report: the certificate measures the inner
    solver's own stopping test, so setting it alone would measure the default."""
    config = impact.tighten_to_stationarity(impact.AulaConfig(), 1e-8)
    assert config.check_stationarity and config.stationarity_tol == 1e-8
    assert config.newton_tol == pytest.approx(1e-9)
    assert config.inner_tol_final == pytest.approx(1e-11)
    assert config.inner_tol_init >= config.inner_tol_final
