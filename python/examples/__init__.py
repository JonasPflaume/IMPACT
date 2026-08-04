"""IMPACT's examples: one directory per task, each with its own model and runner.

None of this is part of the solver. :mod:`impact` is a task-free MPCC solver;
this is the material the paper's experiments are made of, kept in the repository
rather than in the wheel so it can be read, copied and edited. ``pip install .``
gives you the solver alone.

Every example is a directory holding the same three files:

    push_circle/task.py     the model, its tuned AulaConfig, and solve()
    push_circle/viz.py      how to draw a trajectory of it
    push_circle/main.py     the command line

Nothing registers them. :mod:`examples.run` finds the directories, and an example
is equally runnable on its own::

    python python/examples/run.py push_circle --horizon 100 --visualize
    python python/examples/push_circle/main.py --horizon 100 --visualize

or from Python, where a task's ``solve()`` is the whole API::

    from examples.push_circle.task import solve

    result = solve(horizon=100)
    print(result.summary())
    result.save()               # results/push_circle/bcd_aula/trajectory_<ms>.txt

Do not add ``python/`` to ``PYTHONPATH`` to reach any of this: ``python/`` also
holds the ``impact`` source tree, whose copy has no compiled extension, and
``PYTHONPATH`` precedes site-packages -- it would shadow the installed solver.
The scripts here ``sys.path.append`` instead.

Writing your own example means subclassing :class:`impact.MPCCProblem` (explicit
ODE) or :class:`impact.LCPProblem` (contact), or building an
:class:`impact.MPCCDescription` directly -- ``box/task.py``, ``allegro/task.py``
and ``toy/task.py`` are one of each -- and putting it in a new directory. There
is no third step.
"""

from __future__ import annotations

from .run import example_names, example_summary

__all__ = ["example_names", "example_summary"]
