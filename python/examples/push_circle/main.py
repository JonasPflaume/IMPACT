#!/usr/bin/env python3
"""planar disk pushing; the pusher must orbit the disk to reach the goal

The scenario is arranged so the naive action is wrong: the disk's goal is *where
the pusher already stands*, so pushing from the initial contact drives the disk
the wrong way. The pusher has to travel around the disk and push from the far
side.

Nothing scripts that manoeuvre. It comes out of the complementarity
``0 <= f_n (perp) phi >= 0``, whose second leg ``phi = ||s - q|| - R >= 0`` is
non-penetration: the pusher cannot pass through the disk, so a trajectory that
reaches the goal has to route around it. The trajectory is initialised by
replicating the start (``use_constant_state_init``), which is the hard case -- a
local method started there sees only the wrong-direction push.

Run:  python python/examples/push_circle/main.py --horizon 100 --visualize
      python python/examples/push_circle/main.py --distance 1.5 --angle 180 --mode single
      python python/examples/push_circle/main.py --tol 1e-6            # tighter tolerance
      python python/examples/push_circle/main.py --print-config        # what would run
      python python/examples/push_circle/main.py --render-only         # redraw the newest
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path
    import pathlib
    import sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "examples.push_circle"

import argparse

from ..common import cli
from . import task, viz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--horizon", type=int, default=100, help="number of knots")
    parser.add_argument("--distance", type=float, default=1.5,
                        help="pusher/goal distance from the disk centre")
    parser.add_argument("--angle", type=float, default=225.0,
                        help="pusher start direction in degrees")
    parser.add_argument("--goal", nargs=2, type=float, default=None,
                        help="override the disk goal (default: the pusher's own start)")
    parser.add_argument("--mode", choices=("multiple", "single"), default="multiple",
                        help="shooting transcription")
    # Plus the flags every example shares (examples/common/cli.py). Listed here
    # rather than left to --help, so what you can pass is visible where you are:
    #   solver  --stat-tol --tol --newton-tol --inner-tol
    #           --rho-max --max-outer --max-inner --no-saddle --jit
    #           --set FIELD=VALUE                 any other AulaConfig field, repeatable
    #   output  --output --no-save --print-config --quiet --print-level
    #           --visualize --out-dir --minimal --render-only [TRAJECTORY]
    cli.add_flags(parser)
    args = parser.parse_args(argv)

    config = task.config(horizon=args.horizon, distance=args.distance,
                         angle=args.angle, goal=args.goal)
    cli.apply_solver_flags(args, config)
    answered = cli.prepare(args, config, name="push_circle", render=viz.render)
    if answered is not None:
        return answered

    return cli.finish(task.solve(config, mode=args.mode), args, render=viz.render)


if __name__ == "__main__":
    raise SystemExit(cli.guard(main))
