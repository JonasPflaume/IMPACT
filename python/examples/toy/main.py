#!/usr/bin/env python3
"""the smallest complete example: a 4D MPCC with two complementarity blocks

No horizon, no dynamics, no shooting -- ``task.py`` hands an ``MPCCDescription``
straight to ``build_mpcc``. This is the entry point to use when your problem is
not a trajectory: an inverse problem, a static contact configuration, a bilevel
program written in complementarity form.

Run:  python python/examples/toy/main.py --visualize
      python python/examples/toy/main.py --set rho_scale=1.25   # any AulaConfig field
      python python/examples/toy/main.py --print-config         # what would run, as JSON
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path: python python/examples/toy/main.py
    import pathlib
    import sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "examples.toy"

import argparse

from ..common import cli
from . import task, viz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--z0", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0],
                        help="initial guess; the default (0,0,0,0) is the biactive corner")
    # No redraw flag below: this task writes no trajectory file, so there is
    # nothing to redraw later, and the drawing flag works off the result in memory.
    #
    # Plus the flags every example shares (examples/common/cli.py). Listed here
    # rather than left to --help, so what you can pass is visible where you are:
    #   solver  --stat-tol --tol --newton-tol --inner-tol
    #           --rho-max --max-outer --max-inner --no-saddle --jit
    #           --set FIELD=VALUE                 any other AulaConfig field, repeatable
    #   output  --output --no-save --print-config --quiet --print-level
    #           --visualize --out-dir --minimal
    cli.add_flags(parser, replay=False)
    args = parser.parse_args(argv)

    config = task.config()
    cli.apply_solver_flags(args, config)
    answered = cli.prepare(args, config, name="toy")
    if answered is not None:
        return answered

    return cli.finish(task.solve(config, z0=args.z0), args, render=viz.render)


if __name__ == "__main__":
    raise SystemExit(cli.guard(main))
