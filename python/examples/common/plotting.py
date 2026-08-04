"""Matplotlib odds and ends shared by the visualizers.

Requires the ``viz`` extra: ``pip install impact-solver[viz]``.
"""

from __future__ import annotations

__all__ = ["show_if_interactive"]

#: Backends that cannot open a window, so `show()` on them is a warning and a no-op.
_HEADLESS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})


def show_if_interactive() -> None:
    """``plt.show()``, but only where a window can actually appear.

    Every renderer here doubles as a script you run to look at something and as a
    step in a batch pipeline. Calling ``show()`` unconditionally makes the batch
    case print a UserWarning per figure, which trains people to ignore the
    warnings that matter.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() not in _HEADLESS:
        plt.show()
