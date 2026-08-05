"""IMPACT -- an implicit active-set augmented Lagrangian for MPCCs.

This package is the solver and nothing else: a generic MPCC assembler, the two
shooting transcriptions built on it, and the augmented-Lagrangian solve itself.
It knows no tasks. The models, tuned settings, visualizers and drivers for the
paper's experiments live outside it, in the repository's ``examples/``.

Models are written in Python with the CasADi Python API; the solve runs in C++.
The split is at the *symbolic* boundary: Python derives the residual rows and
their Jacobian and passes them over as serialized CasADi functions, so no CasADi
expression is ever built in C++ and the solver never sees a task.

    import impact                    # before casadi; see below
    import casadi as ca
    import numpy as np
    from impact import AulaConfig, BlockOptions, MPCCDescription, Solver, build_mpcc

    #   minimize ||z - [0.8, 0.2]||^2   s.t.   0 <= z1 (perp) z2 >= 0
    z = ca.SX.sym("z", 2)
    desc = MPCCDescription(z=z, cost=z - ca.DM([0.8, 0.2]), cost_is_linear=True)
    desc.add_complementarity("pair", z[0], z[1], BlockOptions(tol=1e-8))

    result = Solver().solve(build_mpcc(desc).subproblem, AulaConfig(), np.zeros(2))
    print(result.z, result.converged)

For a trajectory problem, describe one stage as an :class:`MPCCProblem` (explicit
ODE) or an :class:`LCPProblem` (contact) and hand it to a shooting front-end:

    solution = MultipleShootingSolver(MyTask()).solve(config)
    solution.state_trajectory        # nx x (horizon + 1)

``AulaConfig`` is the C++ struct itself rather than a Python mirror of it, so its
defaults are the solver's defaults by construction and cannot drift out of sync.

Import this package before CasADi. Both sides of that serialized boundary have to
be one CasADi release, and importing it first is what makes them one: the loader
binds the extension to whatever libcasadi is already in the process, and every
CasADi 3.x ships the same SONAME, so a second copy earlier on ``sys.path`` -- a
``pip install --user`` of a different release, an ``/opt`` tree on ``PYTHONPATH``
-- would be bound instead. Order does not matter when the two agree, and when
they do not this package says so at import, naming both, rather than letting it
surface as an unreadable function in the middle of a solve.
"""

from __future__ import annotations

import os as _os
import re as _re
import sys as _sys

# CasADi first, and specifically *our* CasADi: the extension is linked against
# the libcasadi that ships in the `casadi` wheel installed alongside this package
# (see the INSTALL_RPATH of `$ORIGIN/../casadi` in python/CMakeLists.txt), and
# importing the module here makes the failure mode legible (a plain ImportError
# naming casadi) if that wheel is missing.
#
# Which copy gets imported is not a detail. A second CasADi earlier on sys.path
# -- a robotpkg/openrobots tree on PYTHONPATH, a stray `pip install --user`, say
# -- is imported first, and its extension pulls in its own libcasadi under the
# SONAME `libcasadi.so.3.7`. The dynamic loader then satisfies *our* extension's
# DT_NEEDED entry from that already-loaded object and never consults our RPATH at
# all, because a SONAME already in the link map is reused rather than searched
# for. That SONAME identifies nothing: every CasADi 3.x wheel exports
# `libcasadi.so.3.7`, the 3.6 releases included, so the copy that wins may be a
# different release entirely. Putting our own site-packages at the front of
# sys.path for the duration of this one import binds the whole process to the
# pairing the extension was built for, without the caller having to curate
# PYTHONPATH.
_impact_sitedir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_impact_sibling_casadi = _os.path.join(_impact_sitedir, "casadi")
_impact_casadi_first = "casadi" in _sys.modules


def _casadi_series(version):
    """``"3.7.2"`` -> ``(3, 7)``, or ``None`` if that is not a version at all.

    The pair is the whole comparison, and both halves of that are deliberate.
    Major.minor because it is what compatibility actually follows -- and what
    ``casadi>=3.7,<3.8`` in pyproject.toml already promises -- while the SONAME,
    the obvious thing to compare, follows nothing: a CasADi 3.6 wheel also calls
    its library `libcasadi.so.3.7`. Not the patch level, because 3.7.0 and 3.7.2
    read each other's serialized functions and refusing that would break installs
    that work.
    """
    match = _re.match(r"\s*(\d+)\.(\d+)", str(version or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _casadi_mismatch(python_version, runtime_version, build_version):
    """Why these three CasADi versions cannot work together; ``""`` if they can.

    Three, because there are two joints and either can slip:

    * what ``import casadi`` gave Python, and what libcasadi the extension is
      bound to. The problem crosses into C++ as CasADi's own serialized
      functions, so these two have to be one release or the bytes are unreadable
      -- ``Serialization protocol is not compatible``, thrown from inside the
      first solve, a long way from the import that caused it.
    * what libcasadi the extension is bound to, and which CasADi it was compiled
      against. Different releases there and the extension is calling into a
      library it was not built for, which is undefined behaviour that happens to
      raise an undefined-symbol ImportError when it is lucky.

    Versions it cannot parse are not evidence of anything, so they say nothing.
    """
    python, runtime, build = (_casadi_series(v)
                              for v in (python_version, runtime_version, build_version))
    if None in (python, runtime, build):
        return ""
    if python != runtime:
        return (f"Python imported CasADi {python_version}, but the solver extension is "
                f"bound to libcasadi {runtime_version}. Problems reach the solver as "
                "CasADi's own serialized functions and one release cannot read "
                "another's, so the first solve would fail inside "
                "`casadi::Function::deserialize`.")
    if runtime != build:
        return (f"The solver extension was compiled against CasADi {build_version} but "
                f"is bound to libcasadi {runtime_version}, a different release of a "
                "library whose SONAME does not distinguish the two.")
    return ""


def _casadi_version_at(directory):
    """The version of an *unimported* CasADi wheel, read off its own config.h."""
    try:
        with open(_os.path.join(directory, "include", "casadi", "config.h")) as header:
            match = _re.search(r'CASADI_VERSION_STRING\s+"([^"]+)"', header.read())
    except OSError:
        return ""
    return match.group(1) if match else ""


def _loaded_libcasadi():
    """Every libcasadi mapped into this process, by path. Linux only, best effort.

    Worth the two lines: when the versions disagree this is the one fact that
    names the offending file rather than describing it.
    """
    try:
        with open("/proc/self/maps") as maps:
            paths = {line.split()[-1] for line in maps if "/libcasadi.so" in line}
    except OSError:
        return []
    return sorted(path for path in paths
                  if _os.path.basename(path).startswith("libcasadi.so"))


if _impact_casadi_first:
    # Someone imported CasADi before us, so the choice is already made: the
    # loader has committed to whichever libcasadi that copy pulled in and there
    # is no way back -- the extension module is already mapped, and nothing short
    # of a new process can rebind it. It may still be the right one, and usually
    # is; the two checks below are what decide, because "usually" is exactly the
    # kind of thing that stops being true after someone's `pip install`.
    _casadi = _sys.modules["casadi"]
elif _os.path.isdir(_impact_sibling_casadi):
    _sys.path.insert(0, _impact_sitedir)
    try:
        import casadi as _casadi  # noqa: F401
    finally:
        # Remove the entry we just added, not some pre-existing duplicate: this
        # is index 0 unless the import itself reordered sys.path, which nothing
        # in CasADi does.
        try:
            _sys.path.remove(_impact_sitedir)
        except ValueError:
            pass
else:
    # No sibling wheel -- an in-tree checkout, or a build against a system
    # CasADi. Whatever the environment resolves is the only candidate there is.
    import casadi as _casadi  # noqa: F401

_impact_casadi_version = getattr(_casadi, "__version__", "")
_impact_casadi_dir = _os.path.dirname(_os.path.abspath(getattr(_casadi, "__file__", "") or ""))

#: What to do about a mismatch -- not the same advice in the two ways of getting one.
_IMPACT_REMEDY = (
    "Import impact before CasADi -- `import impact` first, then `import casadi` -- "
    "or take the other CasADi off sys.path/PYTHONPATH."
    if _impact_casadi_first else
    "impact imported CasADi itself and still got a mismatched pair, so this is not "
    "an import-order problem: the `casadi` package that answers `import casadi` is "
    "not the one whose libcasadi sits next to the extension. Install both from the "
    "same place."
)

try:
    from . import _impact_core
except ImportError as _exc:  # pragma: no cover - environment-dependent
    # The overwhelmingly likely cause is the SONAME collision described above: a
    # foreign CasADi got imported first and the loader bound our DT_NEEDED to its
    # libcasadi, which is a different release exporting the same SONAME, so a
    # symbol we reference is simply not in it. The bare message is a mangled name,
    # so attach the diagnosis and the things that actually fix it.
    if _impact_casadi_first and _impact_casadi_dir != _impact_sibling_casadi:
        _impact_wanted = _casadi_version_at(_impact_sibling_casadi)
        raise ImportError(
            f"{_exc}\n\n"
            f"CasADi {_impact_casadi_version} was imported from {_impact_casadi_dir} "
            "before impact was, so the dynamic loader bound the solver extension to "
            "that copy's libcasadi: every CasADi 3.x ships one SONAME "
            "(libcasadi.so.3.7, the 3.6 releases included), whichever copy is already "
            "in the process wins, and impact's own RPATH is never consulted. The "
            "symbol error above is what a different CasADi release looks like from "
            "the inside.\n\n"
            + (f"impact needs the CasADi installed beside it at "
               f"{_impact_sibling_casadi} ({_impact_wanted}).\n\n" if _impact_wanted else "")
            + _IMPACT_REMEDY
        ) from _exc
    raise

# The extension loaded, which does not mean it loaded against the right CasADi.
# Whether a foreign libcasadi happens to be missing a symbol this extension
# references is luck; when it is not, the import above succeeds in silence and
# the mismatch surfaces much later, in the middle of a solve, as a serialized
# function the C++ side cannot read. So ask both sides what they actually are
# rather than waiting to find out.
_impact_why = _casadi_mismatch(_impact_casadi_version,
                               _impact_core.casadi_runtime_version,
                               _impact_core.casadi_build_version)
if _impact_why:  # pragma: no cover - environment-dependent
    raise ImportError(
        f"{_impact_why}\n\n"
        f"  CasADi imported by Python : {_impact_casadi_version} from {_impact_casadi_dir}\n"
        f"  libcasadi in this process : {', '.join(_loaded_libcasadi()) or 'unknown'}\n"
        f"  extension was built for   : CasADi {_impact_core.casadi_build_version}"
        + (f", installed beside it at {_impact_sibling_casadi}"
           if _os.path.isdir(_impact_sibling_casadi) else "")
        + f"\n\n{_IMPACT_REMEDY}"
    )

del _impact_sitedir, _impact_sibling_casadi, _impact_casadi_first
del _impact_casadi_version, _impact_casadi_dir, _impact_why, _IMPACT_REMEDY
from ._impact_core import (
    AulaConfig,
    AulaResult,
    ConstraintViolation,
    Solver,
    Status,
    Subproblem,
    project_complementarity,
)
from .config import (
    VECTOR_FIELDS,
    apply_config,
    config_from_dict,
    config_to_dict,
    field_names,
)
from .mpcc import (
    BlockOptions,
    BuiltMPCC,
    ConstraintKind,
    MPCCConstraint,
    MPCCDescription,
    build_mpcc,
)
from .report import planner_tag, result_line, settings_line, tighten_to_stationarity
from .shooting import (
    LCPSingleShootingSolver,
    LCPSolution,
    MultipleShootingSolver,
    ShootingLayout,
    ShootingSolution,
    SingleShootingSolver,
    TrajectorySolution,
    build_multiple_shooting,
    build_single_shooting,
)
from .stage import LCPProblem, LCPStage, MPCCProblem, MPCCStage, StageProblem

__version__ = _impact_core.__version__

__all__ = [
    # solver core
    "AulaConfig",
    "AulaResult",
    "ConstraintViolation",
    "Solver",
    "Status",
    "Subproblem",
    "project_complementarity",
    # generic MPCC assembly
    "BlockOptions",
    "BuiltMPCC",
    "ConstraintKind",
    "MPCCConstraint",
    "MPCCDescription",
    "build_mpcc",
    # task interfaces
    "LCPProblem",
    "LCPStage",
    "MPCCProblem",
    "MPCCStage",
    "StageProblem",
    # shooting transcriptions
    "LCPSingleShootingSolver",
    "LCPSolution",
    "MultipleShootingSolver",
    "ShootingLayout",
    "ShootingSolution",
    "SingleShootingSolver",
    "TrajectorySolution",
    "build_multiple_shooting",
    "build_single_shooting",
    # config as data
    "VECTOR_FIELDS",
    "apply_config",
    "config_from_dict",
    "config_to_dict",
    "field_names",
    # reporting
    "planner_tag",
    "result_line",
    "settings_line",
    "tighten_to_stationarity",
]
