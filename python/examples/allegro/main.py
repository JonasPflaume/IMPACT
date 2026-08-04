#!/usr/bin/env python3
"""in-hand reorientation: receding-horizon MPC against MuJoCo

The only closed-loop example here. Each step queries MuJoCo for the current
contact set, solves a short single-shooting LCP horizon, applies the first
command and repeats. The subproblem is built once: ``phi`` and the contact
Jacobian are per-solve *parameters*, so nothing symbolic is rebuilt between MPC
steps -- which is what makes the loop run at a control rate at all.

Requires the ``sim`` extra and the repository's MuJoCo models:
``pip install 'impact-solver[sim]'``.

Run:  python python/examples/allegro/main.py --object cube --render
      python python/examples/allegro/main.py --video results/allegro/inhand.gif
      python python/examples/allegro/main.py --max-steps 20 --horizon 4 --mu 0.5
      python python/examples/allegro/main.py --print-config       # what would run, as JSON
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path
    import pathlib
    import sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "examples.allegro"

import argparse

from ..common import cli
from . import task


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--object", default="cube",
                        help="selects resources/xmls/env_allegro_<object>.xml")
    parser.add_argument("--model", default=None, help="explicit MuJoCo XML path")
    parser.add_argument("--horizon", type=int, default=4, help="MPC horizon")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="number of MPC steps to execute")
    parser.add_argument("--frame-skip", type=int, default=50,
                        help="simulator substeps per MPC step")
    parser.add_argument("--mu", type=float, default=0.5, help="contact friction coefficient")
    parser.add_argument("--target-yaw-deg", type=float, default=90.0,
                        help="goal yaw in degrees")
    parser.add_argument("--render", action="store_true",
                        help="open the interactive MuJoCo viewer")
    parser.add_argument("--video", default=None,
                        help="write a headless GIF of the rollout")
    parser.add_argument("--video-stride", type=int, default=1,
                        help="capture every Nth executed step")
    parser.add_argument("--camera", default="demo-cam", help="camera name for the video")
    parser.add_argument("--width", type=int, default=640, help="video width")
    parser.add_argument("--height", type=int, default=480, help="video height")
    parser.add_argument("--fps", type=int, default=20, help="video frame rate")
    # No drawing flags below: the rollout draws itself as it runs, with the two
    # flags above, and leaves no saved plan to replay afterwards.
    #
    # Plus the flags every example shares (examples/common/cli.py). Listed here
    # rather than left to --help, so what you can pass is visible where you are:
    #   solver  --stat-tol --tol --newton-tol --inner-tol
    #           --rho-max --max-outer --max-inner --no-saddle --jit
    #           --set FIELD=VALUE                 any other AulaConfig field, repeatable
    #   output  --output --no-save --print-config --quiet --print-level
    cli.add_flags(parser, viz=False)
    args = parser.parse_args(argv)

    config = task.config(horizon=args.horizon)
    cli.apply_solver_flags(args, config)
    answered = cli.prepare(args, config, name="allegro")
    if answered is not None:
        return answered

    result = task.solve(config, object=args.object, model=args.model,
                        max_steps=args.max_steps, frame_skip=args.frame_skip, mu=args.mu,
                        target_yaw_deg=args.target_yaw_deg, viewer=args.render,
                        video=args.video, video_stride=args.video_stride,
                        camera=args.camera, width=args.width, height=args.height,
                        fps=args.fps)
    return cli.finish(result, args)


if __name__ == "__main__":
    raise SystemExit(cli.guard(main))
