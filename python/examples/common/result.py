"""What a task's ``solve()`` hands back.

One small container, deliberately not a framework: a task fills it in and is done
with it. It exists because saving a trajectory and printing a convergence block
are the same job in every example, and because a caller that wants to *use* a
result (a sweep, a notebook, a test) should not have to know which front-end the
task went through to produce it.

    from examples.box.task import solve

    result = solve(horizon=50)
    print(result.summary())
    result.save()                 # results/box/bcd_aula/trajectory_<ms>.txt

Solver statistics are read straight off the underlying solution, so
``result.converged``, ``result.solve_time`` and ``result.complementarity_violation``
work regardless of whether the task used a shooting builder or built its MPCC by
hand.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from impact.report import result_line

from . import trajectory_io

__all__ = ["Result"]


@dataclass
class Result:
    """A finished solve: the numbers, plus what to do with them."""

    #: Task name; names the ``results/<name>/`` directory.
    name: str
    solution: Any
    planner: str

    state: Optional[np.ndarray] = None            # nx x (T + 1)
    control: Optional[np.ndarray] = None          # nu x T
    start: Optional[np.ndarray] = None
    goal: Optional[np.ndarray] = None
    goal_error: float = 0.0

    #: ``write_trajectory`` keywords fixed by the task (title and column labels).
    #: ``None`` means this task has no trajectory to write.
    file: Optional[Dict[str, Any]] = None
    #: Extra ``(label, text)`` rows for :meth:`summary`.
    rows: Tuple[Tuple[str, str], ...] = ()

    #: Overrides for the headline numbers, where the task counts differently from
    #: a single solve: the Allegro MPC loop reports its whole rollout, not its
    #: last step. Read them through :attr:`iterations`, :attr:`solve_time` and
    #: :attr:`converged`, which fall back to the solution when unset -- the raw
    #: fields must not be named after the statistics they override, or they would
    #: shadow the forwarding below and answer ``None`` for every other task.
    rollup_iterations: Optional[int] = None
    rollup_solve_time: Optional[float] = None
    rollup_success: Optional[bool] = None

    #: Files the task already wrote while running (an MPC video, say).
    artifacts: Tuple[Path, ...] = ()
    #: Where :meth:`save` put the trajectory, once it has.
    path: Optional[Path] = field(default=None, init=False)

    # -- the headline numbers ----------------------------------------------
    @property
    def converged(self) -> bool:
        """Did the task succeed -- across the whole rollout, where there is one."""
        return (self.solution.converged if self.rollup_success is None
                else self.rollup_success)

    @property
    def solve_time(self) -> float:
        """Seconds spent solving; the rollout's total where there is one."""
        return (self.solution.solve_time if self.rollup_solve_time is None
                else self.rollup_solve_time)

    @property
    def iterations(self) -> int:
        """Iterations, in whatever the task counts: MPC steps for a rollout."""
        return (self.solution.total_inner_iterations if self.rollup_iterations is None
                else self.rollup_iterations)

    @property
    def is_rollup(self) -> bool:
        """True when the headline numbers describe more than one solve."""
        return self.rollup_success is not None or self.rollup_solve_time is not None

    def __getattr__(self, name: str):
        # Solver statistics live on the solution; forwarding them keeps Result from
        # having to restate two dozen fields that would then have to be kept in
        # sync with AulaResult.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self.__dict__["solution"], name)
        except KeyError:  # during __init__, before `solution` is bound
            raise AttributeError(name) from None
        except AttributeError:
            raise AttributeError(
                f"neither Result nor {self.name}'s solution has '{name}'") from None

    def __repr__(self) -> str:
        return (f"<Result {self.name} planner={self.planner} converged={self.converged} "
                f"objective={self.objective_value:.6g}>")

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        """A human-readable block: convergence, feasibility, where time went.

        Where the task overrode the headline numbers -- the Allegro MPC loop
        reports its whole rollout, not its last step -- the overrides win, so this
        agrees with what :meth:`save` writes into the file rather than quietly
        describing a different thing.
        """
        s = self.solution
        rows: List[Tuple[str, str]] = [
            ("converged", "YES" if self.converged else "NO"),
            ("objective", f"{s.objective_value:.6f}"),
            ("outer / inner / GN", f"{s.outer_iterations} / {s.total_inner_iterations} / "
                                   f"{s.total_gn_iterations}"),
            ("dynamics violation", f"{s.dynamics_violation:.3e}"),
            ("complementarity", f"{s.complementarity_violation:.3e}"),
            ("solve time", f"{self.solve_time:.3f} s"),
        ]
        # The eval/factor split is measured per solve, so it is meaningless against
        # a summed rollout time -- report it only when the two describe one solve.
        if not self.is_rollup and s.solve_time > 0.0:
            rows.append(("time split", f"eval {100 * s.eval_time / s.solve_time:.0f}%, "
                                       f"factor {100 * s.factor_time / s.solve_time:.0f}%"))
        if self.goal is not None:
            rows.append(("goal error", f"{self.goal_error:.3e}"))
        rows.extend(self.rows)

        width = max(len(label) for label, _ in rows)
        head = f"--- {self.name} / {self.planner} ---"
        return "\n".join([head] + [f"{label:<{width}} : {text}" for label, text in rows])

    def result_line(self) -> str:
        """The ``RESULT key=value ...`` line the sweep scripts parse."""
        return result_line(self.planner, self.solution, self.goal_error)

    # -- outputs -----------------------------------------------------------
    def save(self, path=None) -> Path:
        """Write the trajectory file; returns where it went.

        With no path this is ``results/<task>/<planner>/trajectory_<ms>.txt``, the
        layout the C++ drivers and the visualizers already use.
        """
        if self.file is None:
            raise TypeError(f"task '{self.name}' produces no trajectory to save")
        destination = (Path(path) if path is not None else
                       trajectory_io.default_output_path(self.name, self.planner,
                                                         int(time.time() * 1000)))
        self.path = trajectory_io.write_trajectory(
            destination,
            planner=self.planner, task=self.name,
            start_state=self.start, goal_state=self.goal,
            state_trajectory=self.state, control_trajectory=self.control,
            iterations=self.iterations, solve_time=self.solve_time,
            success=bool(self.converged),
            **self.file)
        return self.path
