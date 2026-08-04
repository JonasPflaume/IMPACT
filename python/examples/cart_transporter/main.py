#!/usr/bin/env python3
"""two carts transporting cargo across a gap

Coulomb friction between cargo and cart, written as three complementarity pairs:
sliding one way pins the friction force to one end of its cone, sliding the other
way to the other, and sticking is the remaining case where both slacks vanish and
the force is free inside it. 300 knots, the longest horizon here.

Run:  python python/examples/cart_transporter/main.py --visualize
      python python/examples/cart_transporter/main.py --horizon 300 --frame-interval 10
      python python/examples/cart_transporter/main.py --max-outer 2000
      python python/examples/cart_transporter/main.py --print-config   # what would run
      python python/examples/cart_transporter/main.py --render-only    # redraw the newest
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path
    import pathlib
    import sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "examples.cart_transporter"

import argparse

from ..common import cli
from . import task, viz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0],
                        help="initial [x1, x2, x1_dot, x2_dot]")
    parser.add_argument("--goal", nargs=4, type=float, default=[1.0, 1.0, 0.0, 0.0],
                        help="goal [x1, x2, x1_dot, x2_dot]")
    parser.add_argument("--horizon", type=int, default=300, help="number of knots")
    parser.add_argument("--frame-interval", type=int, default=10,
                        help="overlay snapshot interval, when drawing")
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

    def render(trajectory, **kwargs):
        return viz.render(trajectory, frame_interval=args.frame_interval, **kwargs)

    answered = cli.prepare(args, config, name="cart_transporter", render=render)
    if answered is not None:
        return answered

    return cli.finish(task.solve(config), args, render=render)


if __name__ == "__main__":
    raise SystemExit(cli.guard(main))
