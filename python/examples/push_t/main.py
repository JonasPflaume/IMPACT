#!/usr/bin/env python3
"""Push-T: reorient a T-shaped slider

The largest planar task here: 43 complementarity pairs per knot, seven of them
the absolute-value encoding that turns each face's signed distance into a
distance, eight tying force to contact, and 28 enforcing that at most one face
carries force at a time.

Run:  python python/examples/push_t/main.py --goal 0.05 0.05 1.5708 --visualize
      python python/examples/push_t/main.py --tol 1e-6               # tighter tolerance
      python python/examples/push_t/main.py --set rho_scale=1.25     # any AulaConfig field
      python python/examples/push_t/main.py --print-config           # what would run
      python python/examples/push_t/main.py --render-only            # redraw the newest
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path
    import pathlib
    import sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "examples.push_t"

import argparse

from ..common import cli
from . import task, viz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="initial T pose")
    parser.add_argument("--goal", nargs=3, type=float, default=[0.05, 0.05, 1.5708],
                        help="goal T pose")
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
    answered = cli.prepare(args, config, name="push_t", render=viz.render)
    if answered is not None:
        return answered

    return cli.finish(task.solve(config), args, render=viz.render)


if __name__ == "__main__":
    raise SystemExit(cli.guard(main))
