"""The examples' contract: a directory is an example, and nothing registers it.

A registry used to be the single description each task had, and these tests
pinned that it stayed single. The registry is gone -- an example is a directory
with a ``main.py`` in it -- so what is worth pinning now is the convention that
replaced it: the dispatcher finds examples by looking rather than by being told,
the listing does not pay to import them, and every task's ``solve()`` hands back
the same shape whichever front-end it went through.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import re
import sys

import numpy as np
import pytest

import examples
from examples import run
from examples.common import cli, trajectory_io
from examples.common.trajectory_io import read_trajectory
from impact import AulaConfig

#: Every example that ships in this repository.
SHIPPED = {"toy", "push_circle", "box", "push_t", "cart_transporter", "allegro"}


def test_nothing_puts_python_ahead_of_site_packages():
    """No bootstrap may ``sys.path.insert(0, python/)``.

    ``python/`` holds the ``impact`` *source* tree as well as ``examples``, and the
    source copy has no compiled extension -- the ``.so`` is installed into
    ``site-packages/impact/`` by the wheel. Anything that puts this directory
    ahead of site-packages shadows the working solver with that copy, and every
    entry point dies on ``ImportError: cannot import name '_impact_core'``. Under
    an *editable* install it happens to work, which is exactly why this needs a
    test rather than a comment: the failure is invisible to the person who writes
    it. Append instead.

    ``impact/__init__.py`` is exempt, and is the only thing that can be. It does
    not prepend ``python/``: it prepends *its own parent*, which under a wheel is
    the site-packages directory it was installed into, and only when a sibling
    ``casadi/`` is sitting there -- which is true of an installed wheel and never
    of this source tree. That is the one insert that has to come first, because
    it is how the extension gets bound to the libcasadi it was compiled against
    before any other CasADi on the path can claim the SONAME.
    """
    here = pathlib.Path(__file__).resolve()
    root = here.parents[1]
    exempt = {pathlib.Path("impact/__init__.py")}
    offenders = [
        path.relative_to(root)
        for path in list(root.rglob("*.py")) + list(root.rglob("*.ipynb"))
        if path != here                                  # this file names the pattern
        and "__pycache__" not in path.parts
        and path.relative_to(root) not in exempt
        and "sys.path.insert(0" in path.read_text()
    ]
    assert not offenders, (
        f"these must use sys.path.append for python/: {offenders}")


# ------------------------------------------------------------------ discovery --


def test_every_shipped_example_is_found_and_described():
    names = set(examples.example_names())
    assert names == SHIPPED
    for name in sorted(names):
        assert examples.example_summary(name), f"{name}/main.py has no docstring summary"


def test_a_new_directory_is_an_example_with_nothing_to_register(tmp_path, monkeypatch):
    """The point of the whole layout: writing the directory is the registration."""
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "main.py").write_text(
        '"""a probe example"""\n\n\ndef main(argv=None):\n    return 0\n')
    (tmp_path / "notes").mkdir()          # no main.py, so not an example
    (tmp_path / "notes" / "task.py").write_text("x = 1\n")
    monkeypatch.setattr(run, "EXAMPLES_DIR", tmp_path)

    assert run.example_names() == ["probe"]
    assert run.example_summary("probe") == "a probe example"


def test_listing_does_not_import_the_examples(capsys):
    """Summaries are parsed, not imported.

    Importing every ``main.py`` to print a list would build each task's CasADi
    graph and import MuJoCo -- slow, and it would make ``list`` fail for anyone
    without the optional extras installed.
    """
    already = {name.split(".")[1] for name in sys.modules if name.startswith("examples.")}

    assert run.main(["list"]) == 0
    listed = capsys.readouterr().out
    for name in SHIPPED:
        assert name in listed

    imported = {name.split(".")[1] for name in sys.modules if name.startswith("examples.")}
    assert not (imported & SHIPPED) - already


#: Examples whose ``main.py`` uses the shared flag groups -- all of them here.
SHARED_FLAGS = sorted(SHIPPED)


def _listed_flags(name: str) -> set:
    """The flags an example's ``main.py`` tells a reader it accepts."""
    source = (run.EXAMPLES_DIR / name / "main.py").read_text()
    block = source.split("# Plus the flags every example shares", 1)[1]
    block = block.split("cli.add_flags(", 1)[0]
    # `--help` is argparse's own and is named in the comment's prose, not listed.
    return set(re.findall(r"--[a-z][a-z-]+", block)) - {"--help"}


def _parser_flags(name: str, capsys) -> set:
    """The shared flags argparse actually registered, read off ``--help``."""
    module = importlib.import_module(f"examples.{name}.main")
    with pytest.raises(SystemExit):
        module.main(["--help"])
    help_text = capsys.readouterr().out
    # The task's own knobs come first; the shared groups start at "solver:".
    return set(re.findall(r"--[a-z][a-z-]+", help_text.split("\nsolver:", 1)[1]))


@pytest.mark.parametrize("name", SHARED_FLAGS)
def test_each_main_lists_the_shared_flags_it_accepts(name, capsys):
    """The comment in ``main.py`` is what a reader sees; ``--help`` is the truth.

    They are written in two places on purpose -- someone reading the file should
    not have to open ``common/cli.py`` to find out what they can pass -- so this
    is what stops the two from drifting.
    """
    assert _listed_flags(name) == _parser_flags(name, capsys)


def test_an_unknown_name_lists_the_alternatives(capsys):
    assert run.main(["nope"]) == 2
    assert "push_circle" in capsys.readouterr().err


def test_dashes_and_underscores_name_the_same_example(capsys):
    assert run.main(["push-circle", "--print-config"]) == 0
    assert json.loads(capsys.readouterr().out)["horizon"] == 100


# --------------------------------------------------------------- the results --


def test_solve_forwards_solver_statistics():
    from examples.toy.task import solve

    result = solve()
    assert result.converged is result.solution.converged
    assert result.objective_value == result.solution.objective_value
    # A name on neither says so, rather than returning None.
    with pytest.raises(AttributeError, match="toy's solution"):
        result.no_such_statistic


def test_a_library_solve_is_quiet(capsys):
    """A tuned config is quiet; the trace is something the CLI asks for."""
    from examples.toy.task import config, solve

    assert config().print_level == 0
    solve()
    assert capsys.readouterr().out == ""


def test_save_writes_a_file_the_reader_understands(tmp_path):
    from examples.box.task import config, solve

    result = solve(config(horizon=10))
    path = result.save(tmp_path / "trajectory_1.txt")
    assert path.is_file() and result.path == path

    got = read_trajectory(path)
    assert got["planner"] == result.planner
    # The file stores one row per timestep, so it comes back transposed.
    assert np.allclose(got["state"], result.state.T)
    assert np.allclose(got["control"], result.control.T)
    assert got["success"] == result.converged


def test_default_save_path_follows_the_drivers_layout(tmp_path, monkeypatch):
    from examples.box.task import config, solve

    monkeypatch.setattr(trajectory_io, "CHECKOUT_ROOT", None)
    monkeypatch.chdir(tmp_path)
    result = solve(config(horizon=10))
    path = result.save()
    assert path.parent == tmp_path / "results" / "box" / result.planner
    assert path.name.startswith("trajectory_")


def test_a_task_without_a_trajectory_says_so():
    from examples.toy.task import solve

    with pytest.raises(TypeError, match="no trajectory"):
        solve().save()


def test_an_example_with_no_saved_plan_refuses_to_replay_one():
    """Allegro is a closed-loop MPC rollout, not a plan to play back."""
    args = argparse.Namespace(print_config=False, render_only="", out_dir=None,
                              minimal=False)
    with pytest.raises(TypeError, match="no trajectory file"):
        cli.prepare(args, AulaConfig(), name="allegro")


def test_trajectory_file_round_trips(tmp_path):
    state = np.arange(12, dtype=float).reshape(3, 4)
    control = np.arange(6, dtype=float).reshape(2, 3)
    path = tmp_path / "trajectory_1.txt"
    trajectory_io.write_trajectory(
        path, title="T", planner="bcd_aula", task="demo",
        start_state=[1.0, 2.0, 3.0], goal_state=[4.0, 5.0, 6.0],
        state_trajectory=state, control_trajectory=control,
        iterations=7, solve_time=1.5, success=True,
        start_label="a, b, c", goal_label="a, b, c", state_label="a, b, c",
        control_label="u, v")

    got = trajectory_io.read_trajectory(path)
    assert got["planner"] == "bcd_aula"
    assert got["success"] and got["iterations"] == 7
    assert np.allclose(got["state"], state.T)
    assert np.allclose(got["control"], control.T)
    assert np.allclose(trajectory_io.recorded_goal(path), [4.0, 5.0, 6.0])


# ------------------------------------------------------------------- the CLI --


def _timeless(text: str) -> str:
    """The same output with the wall-clock numbers taken out."""
    kept = [line for line in text.splitlines()
            if not line.startswith(("solve time", "time split"))]
    return "\n".join(re.sub(r"time=\S+", "time=", line) for line in kept)


def test_the_dispatcher_and_the_example_are_the_same_entry_point(capsys):
    """``run.py box ...`` must be exactly ``box/main.py ...``, not a parallel path."""
    from examples.box import main as box_main

    assert run.main(["box", "--horizon", "10", "--quiet", "--no-save"]) == 0
    through_dispatcher = capsys.readouterr().out
    assert box_main.main(["--horizon", "10", "--quiet", "--no-save"]) == 0
    assert _timeless(capsys.readouterr().out) == _timeless(through_dispatcher)


def test_cli_solves_and_saves(tmp_path, capsys):
    out = tmp_path / "trajectory_9.txt"
    assert run.main(["box", "--horizon", "10", "--goal", "0.02", "0.02", "0.1",
                     "--quiet", "--output", str(out)]) == 0
    assert out.is_file()
    assert "RESULT mode=bcd_aula" in capsys.readouterr().out


def test_cli_set_reaches_any_config_field(capsys):
    assert run.main(["box", "--horizon", "10", "--quiet", "--no-save",
                     "--set", "rho_scale=1.25", "--set", "use_saddle=false"]) == 0
    # Nothing to assert in the output; the point is that it parsed and ran.
    assert "RESULT" in capsys.readouterr().out


def test_cli_reports_an_unknown_config_field(capsys):
    assert run.main(["box", "--horizon", "10", "--quiet", "--no-save",
                     "--set", "rho_scail=2"]) == 1
    assert "rho_scail" in capsys.readouterr().err


def test_print_config_describes_the_solve_that_would_run(capsys):
    """After the A/B flags, not before -- otherwise it describes a different solve."""
    assert run.main(["box", "--print-config", "--rho-max", "321"]) == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings["rho_max"] == 321.0
