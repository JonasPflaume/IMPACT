# IMPACT example tasks

Everything in this directory is *material*, not library. The `impact` package
next door is a task-free MPCC solver; this is the collection of models, tuned
settings, drivers and visualizers the paper's experiments are made of. It is
deliberately **not installed** by `pip install .` — it lives in the repository so
it can be read, copied and edited.

Start with [`notebooks/push_circle.ipynb`](notebooks/push_circle.ipynb): a guided
tour of the whole Python surface using the disk-pushing task.

## One example, one directory

```
push_circle/
  task.py     the model, its tuned AulaConfig, and solve()
  viz.py      how to draw a saved trajectory of it
  main.py     the command line
```

That is the entire convention. **Nothing registers an example**: `run.py` lists
the directories that contain a `main.py`, and an example is equally runnable on
its own. These two are the same run:

```bash
python python/examples/run.py push_circle --horizon 100 --visualize
python python/examples/push_circle/main.py --horizon 100 --visualize
```

Writing a new example means writing a new directory. There is no second step —
no table to append to, no `__init__` to edit, no flags to declare twice.

## Running things

```bash
python python/examples/run.py list                   # every example, with its summary
python python/examples/run.py push_t --visualize     # solve, save, render
python python/examples/run.py push_circle --distance 1.5 --angle 225 --horizon 100
python python/examples/run.py box --start 0 0 0 --goal 0.1 0.1 1.0 --tol 1e-6
python python/examples/run.py box --print-config     # the tuned settings, as JSON
python python/examples/run.py push_t --render-only   # redraw the newest trajectory
python python/examples/run.py box --help             # box's own knobs, plus the shared flags
```

`python -m examples.run ...` is the same thing wherever `python/` is on the
import path — an editable install, or running from inside `python/`.

**Do not put `python/` on `PYTHONPATH` to get there.** It holds the `impact`
*source* tree as well, and that copy has no compiled extension in it — the `.so`
is installed into `site-packages/impact/` by the wheel. `PYTHONPATH` is placed
ahead of site-packages, so it would shadow the working solver and everything
would fail with `ImportError: cannot import name '_impact_core'`. The scripts
here `sys.path.append` for exactly that reason, and the Docker image uses a
`.pth` file (whose entries `site` *appends*) instead.

From Python, a task's `solve()` is the whole API:

```python
from examples.box.task import config, solve

result = solve(config(horizon=50, goal=[0.3, 0.2, 0.5]))
print(result.summary())
result.save()                  # results/box/bcd_aula/trajectory_<ms>.txt

from examples.box.viz import render
render(result.path)            # results/box/
```

`config()` is a plain `AulaConfig`, so any solver setting is reachable by setting
the field: `cfg = config(); cfg.rho_max = 400.0`. A tuned config is quiet
(`print_level = 0`); the command line turns the per-outer trace back on, which is
what the C++ drivers print.

## The examples

| example | what it is |
|---|---|
| `toy` | 4D MPCC with two complementarity blocks; no horizon, no dynamics |
| `push_circle` | planar disk pushing; the pusher must orbit the disk to reach the goal |
| `box` | planar box pushing; 10 pairs choose which side pushes and where |
| `push_t` | Push-T; the largest planar task (43 pairs, 7 equalities per knot) |
| `cart_transporter` | two carts carrying cargo across a gap; horizon 300 |
| `allegro` | in-hand reorientation; receding-horizon MPC against MuJoCo |

One of them bends the convention where the task demands it, which is allowed —
the convention is a default, not a frame:

* `allegro` is a closed-loop MPC rollout, so there is no saved plan to replay:
  its `viz.py` records the run as it happens (`--video`, `--render`) and it has
  no `--visualize`.

## What is shared

`common/` is the small amount that would otherwise be copied into every example:

```
common/trajectory_io.py   the trajectory text format the C++ drivers also write
common/cli.py             the solver A/B flags (--tol, --set FIELD=VALUE),
                          the output flags, and the print/save/draw tail
common/result.py          Result: what a task's solve() hands back
common/plotting.py        show() only where a window can actually appear
```

Per-task tuning is *not* shared — it lives in each task's own `config()`, next to
the model it belongs to.

## Extras

The solver needs only `casadi` and `numpy`. These do not:

```bash
pip install 'impact-solver[viz]'   # matplotlib + pillow, for the visualizers
pip install 'impact-solver[sim]'   # mujoco, for the Allegro task
pip install 'impact-solver[all]'   # both, plus pytest
```

## About the tuning

Every task here runs with hand-tuned conditioning scales and tolerances rather
than library defaults, copied from `experiments/<task>/<task>_impact_multiple.cpp`.
Reproducing a published number means reproducing those, which is why they sit in
each task's `config()` rather than being derived at run time.

The penalty schedule (`rho_max` / `rho_scale`) sits beside them for the same
reason, and is likewise the driver's own.

`AulaConfig.auto_rho_init` is the alternative to tuning it by hand: it balances
each block's effective penalty against the objective at the initial point. It is
off by default *because* these tasks are tuned — switched on across all ten
drivers it still converges everywhere, but reaches a 3.2x worse objective in
geometric mean at 1.6x the wall time. Reach for it on a problem you have not
tuned, not on one you have.

## Writing your own

Copy a directory that resembles your problem and edit it:

* an explicit ODE with contacts → `box/`, which subclasses `impact.MPCCProblem`;
* a contact set supplied at run time → `allegro/`, which subclasses
  `impact.LCPProblem`;
* something that is not a trajectory at all → `toy/`, which builds an
  `impact.MPCCDescription` by hand.

Then give it a `main.py` with a one-line docstring summary and a
`main(argv=None) -> int`; `run.py` will list it and run it without being told it
exists. The notebook's last section walks through a complete one from scratch.
