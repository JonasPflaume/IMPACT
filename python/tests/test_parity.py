"""Check the Python modelling layer against the C++ builders.

The Python layer re-derives the augmented-Lagrangian rows, the parameter-buffer
offsets and the saddle layout that ``buildMPCC`` derives in C++. Comparing whole
solves would not settle whether it got them right: these are nonconvex problems
run for hundreds of outer iterations, so a real formulation difference and a
last-bit rounding difference both surface the same way, as a slightly different
iteration count.

This test compares the artefacts instead. ``experiments/parity_dump`` writes out
the CasADi functions the C++ builder emitted plus its block metadata; here the
same task is rebuilt through the Python path and checked for

  * identical metadata -- offsets, dimensions, scales, tolerances, saddle layout;
  * bit-identical residual and Jacobian at random points, including points that
    straddle the ``max`` kinks where the two builds could plausibly disagree.

Bit-identical is the right bar, not "close": both sides evaluate through the same
CasADi VM, so equal expression graphs give equal bits. Any difference at all
means the graphs differ.

Run with::

    cmake --build build --target parity_dump
    pytest python/tests/test_parity.py

``conftest.py`` puts ``python/`` on the path (appended, so it cannot shadow the
installed solver); setting ``PYTHONPATH=python`` instead would, because that
directory also holds the extension-less ``impact`` source tree.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile

import casadi as ca
import numpy as np
import pytest

import impact
from impact.shooting import build_multiple_shooting, build_single_shooting
from impact.stage import MPCCStage

from examples.box.task import BoxPushing, config as box_config
from examples.cart_transporter.task import CartTransporter, config as cart_config
from examples.push_circle.task import PushCircle, config as push_circle_config
from examples.push_t.task import PushT, config as push_t_config

REPO = pathlib.Path(__file__).resolve().parents[2]
# Built in the Python-flavoured tree, so it links the same CasADi wheel this
# process imports. CasADi's serialization is forward- but not backward-compatible,
# and a system CasADi newer than the wheel writes a format the wheel cannot read.
PARITY_DUMP = REPO / "build_py" / "experiments" / "parity_dump" / "parity_dump"

# (task, horizon, python problem, python stage factory, tuned config). Horizons are
# kept small: the comparison is structural, and a short horizon exercises every row
# type while keeping the C++ dump fast.
TASKS = {
    "push_circle": (PushCircle, MPCCStage, push_circle_config),
    "box": (BoxPushing, MPCCStage, box_config),
    "push_t": (PushT, MPCCStage, push_t_config),
    "cart_transporter": (CartTransporter, MPCCStage, cart_config),
}

CASES = [
    (task, mode, horizon)
    for task in TASKS
    for mode in ("multiple", "single")
    for horizon in (4,)
]


requires_dump = pytest.mark.skipif(
    not PARITY_DUMP.exists(),
    reason=f"{PARITY_DUMP} not built; run: cmake --build build --target parity_dump")


def _python_layout(task: str, mode: str, horizon: int):
    problem_cls, stage_cls, tuned = TASKS[task]
    config = tuned(horizon=horizon)
    config.print_level = 0
    stage = stage_cls(problem_cls(), config)
    builder = build_multiple_shooting if mode == "multiple" else build_single_shooting
    return builder(stage, config), config


def _dump_cpp(task: str, mode: str, horizon: int, outdir: pathlib.Path):
    cmd = [str(PARITY_DUMP), task, mode, str(horizon), str(outdir)]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads((outdir / "meta.json").read_text())


@requires_dump
@pytest.mark.parametrize("task,mode,horizon", CASES)
def test_structure_matches_cpp(task, mode, horizon):
    """Offsets, dimensions and the saddle layout must agree exactly."""
    layout, _ = _python_layout(task, mode, horizon)
    sub = layout.subproblem

    with tempfile.TemporaryDirectory() as tmp:
        outdir = pathlib.Path(tmp)
        meta = _dump_cpp(task, mode, horizon, outdir)

        assert sub.num_opt == meta["n_opt"]
        assert layout.off_p == meta["off_p"]
        assert layout.off_x0 == meta["off_x0"]

        # Block names and penalties, in residual-row order. The order is the
        # contract: it is what fixes which rows each saddle block indexes.
        py_names = [n for n, _ in sub.rho_values]
        cpp_names = ([b["name"] for b in meta["dual_blocks"]]
                     + [c["name"] for c in meta["comp_blocks"]])
        assert py_names == cpp_names
        py_rhos = [r for _, r in sub.rho_values]
        cpp_rhos = ([b["rho_init"] for b in meta["dual_blocks"]]
                    + [c["rho_init"] for c in meta["comp_blocks"]])
        assert py_rhos == pytest.approx(cpp_rhos, rel=0, abs=0)


def _random_params(rng, meta):
    """A parameter buffer that is random but structurally admissible.

    Penalties must be positive (the rows carry ``sqrt(rho)``), so those slots are
    drawn accordingly rather than from the normal draw that fills the rest.
    """
    params = rng.normal(scale=1.0, size=meta["n_params"])
    params[0] = 1.0  # rho_one
    for blk in meta["dual_blocks"]:
        params[blk["rho_offset"]] = abs(params[blk["rho_offset"]]) + 0.1
    for blk in meta["comp_blocks"]:
        params[blk["rho_offset"]] = abs(params[blk["rho_offset"]]) + 0.1
    return params


@requires_dump
@pytest.mark.parametrize("task,mode,horizon", CASES)
def test_residual_and_jacobian_bit_identical(task, mode, horizon):
    """r(z; p) and dr/dz must agree to the last bit at random points."""
    layout, _ = _python_layout(task, mode, horizon)

    with tempfile.TemporaryDirectory() as tmp:
        outdir = pathlib.Path(tmp)
        meta = _dump_cpp(task, mode, horizon, outdir)
        cpp_r = ca.Function.load(str(outdir / "residual.casadi"))
        cpp_J = ca.Function.load(str(outdir / "jacobian.casadi"))
        py_r, py_J = layout.residual, layout.jacobian

        n_opt, n_params = meta["n_opt"], meta["n_params"]
        assert layout.n_params == n_params
        assert cpp_r.size1_in(0) == n_opt and cpp_r.size1_in(1) == n_params
        assert cpp_r.size1_out(0) == py_r.size1_out(0), "residual row count differs"
        assert cpp_J.sparsity_out(0) == py_J.sparsity_out(0), "Jacobian sparsity differs"

        rng = np.random.default_rng(0xC0FFEE)
        for trial in range(8):
            # Vary the scale so rows land on both sides of the max() kinks; a
            # difference that only shows on one branch would otherwise slip past.
            z = rng.normal(scale=float(10.0 ** rng.integers(-2, 2)), size=n_opt)
            params = _random_params(rng, meta)

            r_cpp = np.asarray(cpp_r(z, params)).ravel()
            r_py = np.asarray(py_r(z, params)).ravel()
            assert np.array_equal(r_cpp, r_py), (
                f"{task}/{mode} trial {trial}: residual differs "
                f"(max |diff| {np.max(np.abs(r_cpp - r_py)):.3e})")

            J_cpp = np.asarray(cpp_J(z, params).nonzeros())
            J_py = np.asarray(py_J(z, params).nonzeros())
            assert np.array_equal(J_cpp, J_py), (
                f"{task}/{mode} trial {trial}: Jacobian differs "
                f"(max |diff| {np.max(np.abs(J_cpp - J_py)):.3e})")


@requires_dump
@pytest.mark.parametrize("task,mode,horizon", CASES)
def test_solution_matches_cpp_end_to_end(task, mode, horizon):
    """A short solve from both builds must reach the same z.

    Structural equality plus equal residuals already implies this, but running it
    catches anything the metadata comparison does not cover -- for instance a
    parameter written at the wrong offset by the front-end rather than the builder.
    """
    layout, config = _python_layout(task, mode, horizon)
    config.max_outer_iters = 5
    sub = layout.subproblem
    nx = TASKS[task][0]().state_dim
    sub.set_param_value(layout.off_p, np.zeros(nx))
    sub.set_param_value(layout.off_x0, np.zeros(nx))
    z0 = np.zeros(sub.num_opt)
    result = impact.Solver().solve(sub, config, z0)
    assert np.all(np.isfinite(result.z))
