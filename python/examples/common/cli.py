"""The command-line plumbing every example shares.

An example's ``main.py`` declares its own task knobs and then borrows three
things from here: the solver A/B flags (``--tol``, ``--newton-tol``, ``--set
FIELD=VALUE``), the output flags (``--visualize``, ``--output``, ...), and the
tail that prints, saves and draws the result. The task-specific half stays in the
example; nothing in this module knows a task exists.

    import argparse
    from ..common import cli
    from . import task, viz

    def main(argv=None) -> int:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--horizon", type=int, default=50)
        cli.add_flags(parser)
        args = parser.parse_args(argv)

        config = task.config(horizon=args.horizon)
        cli.apply_solver_flags(args, config)
        answered = cli.prepare(args, config, name="box", render=viz.render)
        if answered is not None:
            return answered
        return cli.finish(task.solve(config), args, render=viz.render)

The flags only ever *override* a config field when actually passed, so an
invocation without them behaves exactly as the task's tuned defaults say.

Two measurement traps are guarded here rather than left to callers, and both are
documented on :func:`impact.report.tighten_to_stationarity` and
:func:`impact.report.result_line`: a stationarity target has to drag the inner
tolerances with it, and index-set classification is reported threshold-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, List, Optional

from impact import (AulaConfig, apply_config, config_to_dict, settings_line,
                    tighten_to_stationarity)

from . import trajectory_io
from .result import Result

__all__ = ["add_flags", "add_solver_flags", "add_output_flags", "solver_overrides",
           "apply_solver_flags", "prepare", "finish", "guard"]


def guard(entry: Callable, argv: Optional[List[str]] = None) -> int:
    """Call an example's ``main()``, reporting the usual mistakes as messages.

    A mistyped ``--set`` field or a missing trajectory should read as a
    command-line error, not as a traceback out of the middle of the solver. This
    wraps both ways in -- ``run.py`` and an example's own ``__main__`` -- so the
    two behave the same.
    """
    try:
        return entry(argv)
    except (AttributeError, FileNotFoundError, KeyError, TypeError, ValueError,
            RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


# ------------------------------------------------------------------- flags --


def add_flags(parser: argparse.ArgumentParser, *, viz: bool = True,
              replay: Optional[bool] = None):
    """Both shared groups at once: solver A/B, then output."""
    add_solver_flags(parser)
    add_output_flags(parser, viz=viz, replay=replay)
    return parser


def add_solver_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Port of ``experiments/impact_cli.h``: the solver A/B flags."""
    g = parser.add_argument_group("solver")
    g.add_argument("--stat-tol", type=float, default=None,
                   help="require ||grad L_A||_inf < v to converge")
    g.add_argument("--tol", type=float, default=None,
                   help="outer feasibility tolerance (h, g, comp)")
    g.add_argument("--newton-tol", type=float, default=None,
                   help="Gauss-Newton gradient tolerance")
    g.add_argument("--inner-tol", type=float, default=None,
                   help="late-outer inner stagnation tolerance")
    g.add_argument("--rho-max", type=float, default=None, help="penalty cap")
    g.add_argument("--max-outer", type=int, default=None, help="outer iteration budget")
    g.add_argument("--max-inner", type=int, default=None, help="inner iteration budget")
    g.add_argument("--no-saddle", action="store_true",
                   help="classical normal-equations X-step instead of the saddle form")
    g.add_argument("--jit", action="store_true",
                   help="compile the CasADi functions through a C compiler")
    g.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="FIELD=VALUE",
                   help="set any other AulaConfig field; repeatable")
    return parser


def add_output_flags(parser: argparse.ArgumentParser, *, viz: bool = True,
                     replay: Optional[bool] = None) -> argparse.ArgumentParser:
    """What to do with the result: save it, draw it, or just print the settings.

    ``viz=False`` for examples with nothing to draw -- the Allegro MPC rollout
    renders itself while it runs. ``replay=False`` for examples that can draw a
    result but have no trajectory file to redraw one *from*, which is what
    ``--render-only`` reads.
    """
    replay = viz if replay is None else replay
    g = parser.add_argument_group("output")
    g.add_argument("--output", default=None, help="explicit trajectory path")
    g.add_argument("--no-save", action="store_true", help="do not write a trajectory file")
    g.add_argument("--print-config", action="store_true",
                   help="print the tuned solver settings as JSON and stop")
    g.add_argument("--quiet", action="store_true", help="suppress the solver trace")
    g.add_argument("--print-level", type=int, default=1,
                   help="0 silent, 1 outer summary, 2 inner, 3 inner GN")
    if viz:
        g.add_argument("--visualize", action="store_true",
                       help="render the result after solving")
        g.add_argument("--out-dir", default=None, help="visualization directory")
        g.add_argument("--minimal", action="store_true",
                       help="hide axes and labels where the visualizer supports it")
    if replay:
        g.add_argument("--render-only", nargs="?", const="", default=None,
                       metavar="TRAJECTORY",
                       help="render a saved trajectory (newest, or the one given) and stop")
    return parser


# ------------------------------------------------------------------- apply --


def solver_overrides(args: argparse.Namespace) -> dict:
    """The AulaConfig fields these flags ask for, as a dict.

    Only flags actually passed appear, so the task's tuned defaults survive
    untouched everywhere the user stayed quiet.
    """
    out = {}
    if args.tol is not None and args.tol > 0.0:
        out["outer_tol_h"] = out["outer_tol_g"] = out["outer_tol_comp"] = args.tol
    if args.newton_tol is not None and args.newton_tol > 0.0:
        out["newton_tol"] = args.newton_tol
    if args.rho_max is not None and args.rho_max > 0.0:
        out["rho_max"] = args.rho_max
    if args.max_outer is not None and args.max_outer > 0:
        out["max_outer_iters"] = args.max_outer
    if args.max_inner is not None and args.max_inner > 0:
        out["max_inner_iters"] = args.max_inner
    if args.no_saddle:
        out["use_saddle"] = False
    if args.jit:
        out["jit"] = True
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects FIELD=VALUE, got '{item}'")
        field, _, text = item.partition("=")
        out[field.strip()] = _literal(text.strip())
    return out


def apply_solver_flags(args: argparse.Namespace, config: AulaConfig) -> AulaConfig:
    """Apply parsed flags on top of a task's tuned defaults.

    ``--stat-tol`` and ``--inner-tol`` are applied last and in that order,
    because ``--stat-tol`` moves the inner tolerances with it and an explicit
    ``--inner-tol`` has to win over that.
    """
    apply_config(config, **solver_overrides(args))
    if args.stat_tol is not None and args.stat_tol > 0.0:
        tighten_to_stationarity(config, args.stat_tol)
    if args.inner_tol is not None and args.inner_tol > 0.0:
        config.inner_tol_final = args.inner_tol
        config.inner_tol_init = max(config.inner_tol_init, args.inner_tol)
    if getattr(args, "print_level", None) is not None:
        config.print_level = 0 if args.quiet else args.print_level
    return config


def _literal(text: str):
    """Parse a ``--set`` value as bool, int, float or string, in that order."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


# ------------------------------------------------------------- the two tails --


def prepare(args: argparse.Namespace, config: AulaConfig, *, name: str,
            render: Optional[Callable] = None) -> Optional[int]:
    """The last step before solving: answer the flags that need no solve, or echo.

    Returns an exit code if the command is already answered -- ``--print-config``
    prints what *would* run and ``--render-only`` redraws a trajectory already on
    disk -- and ``None`` to carry on, having echoed the settings actually in
    force. Both live here rather than in each ``main.py`` because getting them
    wrong is silent: a config printed before the A/B flags have landed describes
    a solve that is not the one about to happen.
    """
    if args.print_config:
        print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))
        return 0

    if getattr(args, "render_only", None) is not None:
        if render is None:
            raise TypeError(f"'{name}' writes no trajectory file to render from")
        path = trajectory_io.resolve_trajectory(name, args.render_only or None)
        print(f"rendering {path}")
        for written in render(path, out_dir=args.out_dir, minimal=args.minimal):
            print(f"rendered   : {written}")
        return 0

    print(settings_line(config))
    return None


def finish(result: Result, args: argparse.Namespace,
           render: Optional[Callable] = None) -> int:
    """Print the result, save it, draw it. The last line of a ``main()``."""
    print()
    print(result.summary())
    print(result.result_line())

    if result.file is not None and not args.no_save:
        print(f"trajectory : {result.save(args.output)}")
    for artifact in result.artifacts:
        print(f"artifact   : {artifact}")

    if getattr(args, "visualize", False):
        if render is None:
            raise TypeError(f"'{result.name}' has no visualizer")
        # A renderer reads the saved trajectory, so what it gets is a path -- and
        # the file is written first if it has not been already, which is what keeps
        # the picture and the file in agreement. A task that writes no trajectory
        # (toy) gets the result itself, since that is all there is to draw from.
        subject = result if result.file is None else (result.path or result.save())
        for written in render(subject, goal=result.goal, out_dir=args.out_dir,
                              minimal=args.minimal):
            print(f"rendered   : {written}")
    return 0
