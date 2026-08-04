#!/usr/bin/env python3
"""planar box pushing; the contact point slides along the box's sides

Ten complementarity pairs choose which side is pushing and where on it, so the
switching sequence is an output of the solve rather than a mode schedule someone
wrote down.

Run:  python python/examples/box/main.py --goal 0.1 0.1 1.0 --visualize
      python python/examples/box/main.py --tol 1e-6                # tighter tolerance
      python python/examples/box/main.py --set rho_scale=1.25      # any AulaConfig field
      python python/examples/box/main.py --print-config            # what would run
      python python/examples/box/main.py --render-only             # redraw the newest
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path
    import pathlib
    import sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "examples.box"

import argparse

from ..common import cli
from . import task, viz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="initial box pose")
    parser.add_argument("--goal", nargs=3, type=float, default=[0.1, 0.1, 1.0],
                        help="goal box pose")
    parser.add_argument("--horizon", type=int, default=50, help="number of knots")
    # Plus the flags every example shares (examples/common/cli.py). Listed here
    # rather than left to --help, so what you can pass is visible where you are:
    #   solver  --stat-tol --tol --newton-tol --inner-tol
    #           --rho-max --max-outer --max-inner --no-saddle --jit
    #           --set FIELD=VALUE                 any other AulaConfig field, repeatable
    #   output  --output --no-save --print-config --quiet --print-level
    #           --visualize --out-dir --minimal --render-only [TRAJECTORY]
    cli.add_flags(parser)
    args = parser.parse_args(argv)

    config = task.config(horizon=args.horizon, start=args.start, goal=args.goal)
    cli.apply_solver_flags(args, config)
    answered = cli.prepare(args, config, name="box", render=viz.render)
    if answered is not None:
        return answered

    return cli.finish(task.solve(config), args, render=viz.render)


if __name__ == "__main__":
    raise SystemExit(cli.guard(main))
