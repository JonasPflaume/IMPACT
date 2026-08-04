"""The little that every example shares: file format, flags, result object.

Nothing here knows about any particular task, and no example is *required* to use
any of it -- an example is just a directory with a ``main.py`` in it. These are
the pieces that would otherwise be copied seven times: the trajectory text format
the C++ drivers also write, the solver A/B flags, and the object a task's
``solve()`` hands back so that saving and reporting look the same everywhere.

    from ..common import Result, cli, trajectory_io
"""

from __future__ import annotations

from . import cli, plotting, trajectory_io
from .plotting import show_if_interactive
from .result import Result
from .trajectory_io import (CHECKOUT_ROOT, read_trajectory, resolve_output_dir,
                            resolve_trajectory, results_dir, write_trajectory)

__all__ = ["Result", "cli", "plotting", "trajectory_io", "show_if_interactive",
           "CHECKOUT_ROOT", "read_trajectory", "write_trajectory", "resolve_trajectory",
           "resolve_output_dir", "results_dir"]
