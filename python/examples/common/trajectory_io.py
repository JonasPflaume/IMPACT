"""Read and write the solver's trajectory text files.

The format is the one the C++ drivers write and the ``*_visual.py`` scripts
parse, reproduced byte-for-byte -- header lines, blank-line spacing, and the
``%.10f`` fixed-point formatting included -- so trajectories from either
front-end are interchangeable and the existing visualizers need no change.

It is a sequence of ``# Comment`` headers, each followed by its values::

    # <Task> Trajectory
    # Planner: bcd_aula
    # Task: push_circle

    # Start State (qx, qy, sx, sy)
    0.0000000000 0.0000000000 -1.0606601718 -1.0606601718

    # Goal State (qx, qy)
    ...
    # State Trajectory (rows: timesteps, cols: qx, qy, sx, sy)
    <one row per timestep>

    # Control Trajectory (rows: timesteps, cols: fn, ft, vx, vy)
    <one row per timestep>
"""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["write_trajectory", "read_trajectory", "recorded_goal", "results_dir",
           "default_output_path", "resolve_trajectory", "resolve_output_dir",
           "task_results_dir", "default_results_root", "CHECKOUT_ROOT"]

#: Repository root, or None if this file has been copied out of the checkout.
#: This module lives at `python/examples/common/`, so the root is three levels up,
#: and the `impact_solver/` sibling is what confirms it -- results have to land in
#: the repository's `results/`, not wherever the process happened to be started.
CHECKOUT_ROOT = next(
    (p for p in [pathlib.Path(__file__).resolve().parents[3]] if (p / "impact_solver").is_dir()),
    None)


def default_results_root() -> pathlib.Path:
    """``<repo>/results`` in a source checkout, else ``./results``."""
    return (CHECKOUT_ROOT / "results") if CHECKOUT_ROOT else (pathlib.Path.cwd() / "results")


def results_dir(task: str, root: Optional[pathlib.Path] = None) -> pathlib.Path:
    """``results/<task>``, created if missing."""
    base = pathlib.Path(root) if root is not None else default_results_root()
    d = base / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_output_path(task: str, planner: str, timestamp_ms: int,
                        root: Optional[pathlib.Path] = None) -> pathlib.Path:
    """``results/<task>/<planner>/trajectory_<ts>.txt``, as the drivers name it."""
    d = results_dir(task, root) / planner
    d.mkdir(parents=True, exist_ok=True)
    return d / f"trajectory_{timestamp_ms}.txt"


def _fmt_row(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.10f}" for v in values)


def write_trajectory(path, *, title: str, planner: str, task: str,
                     start_state: Sequence[float], goal_state: Sequence[float],
                     state_trajectory: np.ndarray, control_trajectory: np.ndarray,
                     iterations: int, solve_time: float, success: bool,
                     start_label: str, goal_label: str, state_label: str,
                     control_label: str,
                     preamble: Optional[List[Tuple[str, Sequence[float]]]] = None) -> pathlib.Path:
    """Write one trajectory file.

    ``state_trajectory`` is ``nx x (T+1)`` and ``control_trajectory`` ``nu x T``
    -- the solver's column-per-timestep layout. The file stores one *row* per
    timestep, so both are transposed on the way out.

    ``preamble`` holds task-specific scalar sections written before the start
    state (push_circle's ``# Disk Radius``, for instance), as
    ``[(header, values), ...]``.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out: List[str] = [f"# {title}", f"# Planner: {planner}", f"# Task: {task}", ""]
    for header, values in (preamble or []):
        out += [f"# {header}", _fmt_row(values), ""]
    out += [f"# Start State ({start_label})", _fmt_row(start_state), ""]
    out += [f"# Goal State ({goal_label})", _fmt_row(goal_state), ""]
    out += ["# Iterations", str(int(iterations)), ""]
    out += ["# Solve Time (seconds)", f"{float(solve_time):.10f}", ""]
    out += ["# Success", "1" if success else "0", ""]
    out += [f"# State Trajectory (rows: timesteps, cols: {state_label})"]
    out += [_fmt_row(col) for col in np.asarray(state_trajectory).T]
    out += ["", f"# Control Trajectory (rows: timesteps, cols: {control_label})"]
    out += [_fmt_row(col) for col in np.asarray(control_trajectory).T]

    path.write_text("\n".join(out) + "\n")
    return path


def read_trajectory(path) -> Dict[str, object]:
    """Parse a trajectory file into a dict.

    Returns ``state``/``control`` as ``(T, n)`` arrays -- rows are timesteps, the
    file's own layout -- plus ``start``, ``goal``, ``success``, ``iterations``,
    ``solve_time``, ``planner`` and any task-specific scalar sections keyed by
    their header text.
    """
    lines = pathlib.Path(path).read_text().splitlines()

    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip()
            sections.setdefault(current, [])
        elif stripped and current is not None:
            sections[current].append(stripped)

    def find(prefix: str) -> Optional[List[str]]:
        for key, values in sections.items():
            if key.startswith(prefix):
                return values
        return None

    def numbers(prefix: str) -> Optional[np.ndarray]:
        rows = find(prefix)
        if rows is None:
            return None
        return np.array([[float(v) for v in row.split()] for row in rows])

    planner = ""
    for key in sections:
        if key.startswith("Planner:"):
            planner = key.split(":", 1)[1].strip()

    state = numbers("State Trajectory")
    control = numbers("Control Trajectory")
    start = numbers("Start State")
    goal = numbers("Goal State")
    iterations = find("Iterations")
    solve_time = find("Solve Time")
    success = find("Success")

    out: Dict[str, object] = {
        "planner": planner,
        "state": state,
        "control": control,
        "start": None if start is None else start.ravel(),
        "goal": None if goal is None else goal.ravel(),
        "iterations": int(iterations[0]) if iterations else 0,
        "solve_time": float(solve_time[0]) if solve_time else 0.0,
        "success": bool(int(success[0])) if success else False,
        "sections": sections,
    }
    return out


def recorded_goal(path) -> Optional[np.ndarray]:
    """The goal state stored in a trajectory file, or ``None`` if unreadable.

    What the visualizers draw their goal marker from. Leaving the goal out is not
    the same as having no goal: the planar renderers simply omit the marker when
    it is ``None``, so a renderer that forgets to look produces a picture quietly
    missing the thing the trajectory was aiming at.
    """
    try:
        return read_trajectory(path)["goal"]
    except (OSError, ValueError, KeyError):
        return None


def resolve_trajectory(task: str, supplied_path=None,
                       root: Optional[pathlib.Path] = None) -> pathlib.Path:
    """An explicit trajectory path, or the newest one recorded for ``task``."""
    if supplied_path:
        p = pathlib.Path(supplied_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Trajectory file does not exist: {p}")
        return p
    d = results_dir(task, root)
    candidates = [p for p in d.rglob("trajectory_*.txt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory_*.txt files found under {d}. Run an experiment first or "
            "pass a trajectory path explicitly.")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def resolve_output_dir(task: str, supplied_path=None,
                       root: Optional[pathlib.Path] = None) -> pathlib.Path:
    """An explicit output directory, or ``results/<task>``. Created if missing."""
    if supplied_path:
        d = pathlib.Path(supplied_path).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d
    return results_dir(task, root)


#: Alias kept for the visualizers, which were written against ``visual_utils``.
task_results_dir = results_dir
