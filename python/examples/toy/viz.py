"""Draw the two complementarity crosses, the target and the solution.

The only visualizer here that does not read a trajectory file, because the task
has no trajectory: it takes the :class:`~examples.common.Result` directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

from ..common import trajectory_io
from .task import TARGET

__all__ = ["render"]


def render(result, out=None, out_dir=None, **_) -> List[Path]:
    """Plot both pairs; returns the file written."""
    import matplotlib as mpl

    if not os.environ.get("DISPLAY"):
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(result.solution.z, dtype=float)
    extent = 1.05 * max(1.0, float(np.max(np.abs(TARGET))), float(np.max(np.abs(z))))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, ax in enumerate(axes):
        ax.plot([0, extent], [0, 0], color="#2563eb", lw=4)
        ax.plot([0, 0], [0, extent], color="#2563eb", lw=4,
                label=r"$0\leq x\perp y\geq0$")
        ax.scatter(*TARGET[2 * i:2 * i + 2], marker="x", s=100, lw=2.5,
                   color="#dc2626", label="target")
        ax.scatter(*z[2 * i:2 * i + 2], marker="o", s=80, color="#16a34a",
                   label="solution")
        ax.set(xlim=(-0.05 * extent, extent), ylim=(-0.05 * extent, extent),
               xlabel=f"x{2 * i + 1}", ylabel=f"x{2 * i + 2}",
               title=f"Complementarity pair {i + 1}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()

    path = (Path(out) if out else
            trajectory_io.resolve_output_dir("toy", out_dir) / "toy_mpcc.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return [path]
