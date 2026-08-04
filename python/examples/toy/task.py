"""A minimal 4D MPCC, built by hand.

    minimize    ||x - [0.8, 0.2, 0.25, 0.75]||^2
    subject to  0 <= x1 ⊥ x2 >= 0
                0 <= x3 ⊥ x4 >= 0

Not a trajectory problem: no horizon, no shooting, just an
:class:`~impact.MPCCDescription` passed to :func:`~impact.build_mpcc`. It uses
two *separate* complementarity groups rather than one stacked group, which is
what exercises the multi-block path -- each group carries its own slacks,
multipliers, penalty, scale and tolerance. Here the second group is deliberately
given scale 0.75 and ``rho_init`` 2.0 to show that they are independent.

The closest feasible point is ``[0.8, 0, 0, 0.75]``: the objective wants
``x2 = 0.2`` and ``x3 = 0.25``, but complementarity forbids both legs of a pair
being positive, so one of each pair is driven to zero. Objective
``0.2^2 + 0.25^2 = 0.1025``.

Port of ``experiments/toy_mpcc/toy_mpcc.cpp``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import casadi as ca
import numpy as np

from impact import AulaConfig, Solver
from impact.mpcc import BlockOptions, MPCCDescription, build_mpcc
from impact.report import planner_tag

from ..common import Result

__all__ = ["TARGET", "EXPECTED_OBJECTIVE", "build", "config", "solve"]

TARGET = np.array([0.8, 0.2, 0.25, 0.75])
EXPECTED_OBJECTIVE = 0.1025


def build():
    """Assemble the toy MPCC. Returns a :class:`~impact.mpcc.BuiltMPCC`."""
    z = ca.SX.sym("z", 4)
    desc = MPCCDescription(z=z, p=ca.SX.sym("p", 0), cost=z - ca.DM(TARGET),
                           cost_is_linear=True)
    desc.add_complementarity("axis_a", z[0], z[1], BlockOptions(scale=1.0, rho_init=1.0,
                                                                tol=1e-8))
    desc.add_complementarity("axis_b", z[2], z[3], BlockOptions(scale=0.75, rho_init=2.0,
                                                                tol=1e-8))
    return build_mpcc(desc)


def config() -> AulaConfig:
    """The tuned settings, from ``experiments/toy_mpcc/toy_mpcc.cpp``."""
    c = AulaConfig()
    c.max_outer_iters = 300
    c.max_inner_iters = 50
    c.rho_scale = 1.5
    c.rho_max = 1e3
    c.outer_tol_h = 1e-7
    c.outer_tol_comp = 1e-7
    c.newton_max_iter = 50
    # Quiet as a library call; the CLI turns the per-outer trace back on
    # (--print-level, default 1), which is what the C++ drivers print.
    c.print_level = 0
    return c


def solve(cfg: Optional[AulaConfig] = None,
          z0: Optional[Sequence[float]] = None) -> Result:
    """Solve it.

    ``z0`` defaults to the biactive corner ``(0,0,0,0)`` -- the hardest place to
    start, since every pair sits exactly on the kink and neither branch is
    preferred.
    """
    cfg = config() if cfg is None else cfg
    built = build()
    start = np.zeros(4) if z0 is None else np.asarray(z0, dtype=float).ravel()

    r = Solver().solve(built.subproblem, cfg, start)
    return Result(
        name="toy", solution=r, planner=planner_tag(cfg),
        rows=(("x*", str(np.round(r.z, 10))),
              ("expected objective", f"{EXPECTED_OBJECTIVE}"),
              ("|x1 x2|, |x3 x4|", f"{abs(r.z[0] * r.z[1]):.3e}, "
                                   f"{abs(r.z[2] * r.z[3]):.3e}")))
