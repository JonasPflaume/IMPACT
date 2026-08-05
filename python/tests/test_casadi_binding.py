"""One CasADi, on both sides of the boundary.

The solver's Python half builds the residual and its Jacobian and hands them to
C++ in CasADi's own serialized form, which only works if the CasADi that wrote
those bytes and the libcasadi that reads them are one release. Nothing in the
environment enforces that. Every CasADi 3.x wheel exports the SONAME
``libcasadi.so.3.7`` -- the 3.6 releases included -- so whichever copy reaches the
process first is the one the extension gets bound to, RPATH or no RPATH, and a
second CasADi earlier on ``sys.path`` is enough to make the two halves different
releases. Import ``impact`` first and it binds the process to the pairing it was
built for; import CasADi first and that choice is already made.

These tests pin both ends of that: the environment they are running in is a
matched pair, the guard is silent when the pair matches, and it fails the import
-- immediately, naming both versions -- when it does not, rather than letting the
mismatch surface much later as an unreadable serialized function inside a solve.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import casadi as ca
import pytest

import impact
from impact import _impact_core

#: ``python/``, which the probes below append so they can reach ``examples``.
PYTHON_DIR = str(pathlib.Path(__file__).resolve().parents[1])


def _run(source: str, **env_overrides) -> subprocess.CompletedProcess:
    """A probe in a fresh interpreter, since import order is a per-process fact."""
    env = dict(os.environ, **env_overrides)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source)],
                          capture_output=True, text=True, env=env)


def _stub_casadi(directory: pathlib.Path, version: str) -> str:
    """A ``casadi`` package that is only a version number.

    Staging the real failure needs two CasADi releases installed side by side,
    which a test cannot arrange. It does not need to: the guard's decision reads
    the version string and nothing else, so a stub in the right place on the path
    takes exactly the branch a 3.6 wheel on ``PYTHONPATH`` would take.
    """
    (directory / "casadi").mkdir()
    (directory / "casadi" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    return str(directory)


# ------------------------------------------------- what this environment is --


def test_both_sides_of_the_boundary_are_the_same_casadi():
    """The property everything else in this file exists to protect."""
    assert impact._casadi_mismatch(ca.__version__,
                                   _impact_core.casadi_runtime_version,
                                   _impact_core.casadi_build_version) == ""


def test_the_extension_says_which_casadi_it_is_bound_to():
    """Compiled-against and bound-to are different questions, and both are asked.

    The second one cannot be answered at build time -- it is whatever the loader
    picked -- so the extension has to be the one to answer it, by calling into
    the library it actually got.
    """
    assert impact._casadi_series(_impact_core.casadi_build_version) is not None
    assert impact._casadi_series(_impact_core.casadi_runtime_version) is not None


# ------------------------------------------------------------ the decision --


@pytest.mark.parametrize("python_version, runtime_version, build_version", [
    ("3.6.7", "3.7.2", "3.7.2"),   # a 3.6 wheel claimed the SONAME first
    ("3.7.2", "3.6.7", "3.6.7"),   # the same collision, seen from the other side
    ("3.7.2", "3.7.2", "3.6.7"),   # built against one CasADi, running against another
])
def test_a_mismatched_trio_is_refused(python_version, runtime_version, build_version):
    why = impact._casadi_mismatch(python_version, runtime_version, build_version)
    assert why, "a release difference has to be caught, wherever it sits"
    assert "3.6.7" in why and "3.7.2" in why, f"both versions have to be named: {why}"


def test_a_patch_difference_is_not_a_mismatch():
    """3.7.0 reads what 3.7.2 wrote; refusing that would break installs that work."""
    assert impact._casadi_mismatch("3.7.0", "3.7.2", "3.7.2") == ""


def test_an_unreadable_version_accuses_nobody():
    """A version string this guard cannot parse is not evidence of anything."""
    assert impact._casadi_mismatch("", "3.7.2", "3.7.2") == ""
    assert impact._casadi_series("who knows") is None


# ------------------------------------------------------------ import order --


def test_a_foreign_casadi_imported_first_fails_the_import(tmp_path):
    """The bug this guard is for: caught at ``import impact``, not mid-solve.

    Within one release series the symbols line up and the extension loads, so a
    mismatched pair used to get all the way to the first solve before failing --
    inside ``Function::deserialize``, on a serialized function, a long way from
    the import that chose the wrong CasADi.
    """
    stub = _stub_casadi(tmp_path, "3.6.7")
    done = _run("import casadi\nimport impact\n",
                PYTHONPATH=os.pathsep.join([stub, os.environ.get("PYTHONPATH", "")]))

    assert done.returncode != 0, "a CasADi from another release must not be accepted"
    assert "ImportError" in done.stderr
    assert "3.6.7" in done.stderr and _impact_core.casadi_runtime_version in done.stderr
    assert "Import impact before CasADi" in done.stderr


def test_importing_impact_first_survives_that_same_environment(tmp_path):
    """And this is why the order is worth asking for: it is a fix, not a warning.

    Same shadowing ``PYTHONPATH`` as above. Imported first, ``impact`` puts its
    own site-packages at the front for the length of one import, so the CasADi
    the process ends up with is the one its extension was built against -- and
    the caller's later ``import casadi`` gets that one too.
    """
    stub = _stub_casadi(tmp_path, "3.6.7")
    done = _run("import impact\nimport casadi\nprint(casadi.__version__)\n",
                PYTHONPATH=os.pathsep.join([stub, os.environ.get("PYTHONPATH", "")]))

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == _impact_core.casadi_runtime_version


@pytest.mark.parametrize("first, second", [("casadi", "impact"), ("impact", "casadi")])
def test_either_order_works_in_this_environment(first, second):
    """Order is only a repair for a broken environment; here neither may fail.

    This is the test that goes red on a machine with two CasADi installs of
    different releases -- which is a fact about the machine, and one worth being
    told before a solve tells you.
    """
    done = _run(f"""
        import {first}
        import {second}
        import casadi, impact
        print(casadi.__version__, impact._impact_core.casadi_runtime_version,
              impact._impact_core.casadi_build_version)
    """)

    assert done.returncode == 0, done.stderr
    assert impact._casadi_mismatch(*done.stdout.split()) == ""


@pytest.mark.parametrize("statement", [
    "import examples.toy.main",
    "from examples import run; run.main(['toy', '--print-config'])",
])
def test_an_entry_point_reaches_the_solver_before_casadi(statement):
    """Every way in has to import ``impact`` first, and no file says so.

    ``main.py`` imports ``..common`` (which imports the solver) before ``.task``
    (which imports CasADi) only because that is the order isort produces. Swap
    those two lines and every example still runs -- on a machine with one CasADi.
    This is the test that notices on the other kind.
    """
    done = _run(f"""
        import sys
        sys.path.append({PYTHON_DIR!r})
        seen = []
        sys.addaudithook(lambda event, args: event == "import" and seen.append(args[0]))
        {statement}
        print(next(name for name in seen if name in ("casadi", "impact")), file=sys.stderr)
    """)

    assert done.returncode == 0, done.stderr
    assert done.stderr.splitlines()[-1] == "impact"
