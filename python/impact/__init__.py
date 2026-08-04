"""IMPACT -- an implicit active-set augmented Lagrangian for MPCCs.

This package is the solver and nothing else: a generic MPCC assembler, the two
shooting transcriptions built on it, and the augmented-Lagrangian solve itself.
It knows no tasks. The models, tuned settings, visualizers and drivers for the
paper's experiments live outside it, in the repository's ``examples/``.

Models are written in Python with the CasADi Python API; the solve runs in C++.
The split is at the *symbolic* boundary: Python derives the residual rows and
their Jacobian and passes them over as serialized CasADi functions, so no CasADi
expression is ever built in C++ and the solver never sees a task.

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
"""

from __future__ import annotations

import os as _os
import sys as _sys

# CasADi first, and specifically *our* CasADi: the extension is linked against
# the libcasadi that ships in the `casadi` wheel installed alongside this package
# (see the INSTALL_RPATH of `$ORIGIN/../casadi` in python/CMakeLists.txt), and
# importing the module here makes the failure mode legible (a plain ImportError
# naming casadi) if that wheel is missing.
#
# Which copy gets imported is not a detail. A second CasADi earlier on sys.path
# -- a robotpkg/openrobots tree on PYTHONPATH, say -- is imported first, and its
# extension pulls in its own libcasadi under the SONAME `libcasadi.so.3.7`. The
# dynamic loader then satisfies *our* extension's DT_NEEDED entry from that
# already-loaded object and never consults our RPATH at all, because a SONAME
# already in the link map is reused rather than searched for. The two libraries
# are different C++ builds -- the wheel predates the GCC 5 std::string ABI, a
# distro build does not -- so the import dies on a mangled symbol name
# (`...deserializeERKSs` vs `...deserializeERKNSt7__cxx11...`) that says nothing
# about the actual cause. Putting our own site-packages at the front of sys.path
# for the duration of this one import binds the whole process to the pairing the
# extension was built for, without the caller having to curate PYTHONPATH.
_impact_sitedir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_impact_sibling_casadi = _os.path.join(_impact_sitedir, "casadi")

if "casadi" in _sys.modules:
    # Someone imported CasADi before us, so the choice is already made: the
    # loader has committed to whichever libcasadi that copy pulled in and there
    # is no way back. It may still be the right one, or an ABI-compatible one,
    # so this is not an error yet -- see the handler around the extension import
    # below, which turns the symbol failure into an explanation if it is not.
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

try:
    from . import _impact_core
except ImportError as _exc:  # pragma: no cover - environment-dependent
    # The overwhelmingly likely cause is the SONAME collision described above:
    # a foreign CasADi got imported first and the loader bound our DT_NEEDED to
    # its libcasadi, whose std::string ABI does not match the one we were
    # compiled against. The bare message is a mangled symbol name, so attach the
    # diagnosis and the two things that actually fix it.
    _impact_loaded_dir = _os.path.dirname(
        _os.path.abspath(getattr(_casadi, "__file__", "") or "")
    )
    if "undefined symbol" in str(_exc) and _impact_loaded_dir != _impact_sibling_casadi:
        raise ImportError(
            f"{_exc}\n\n"
            "impact was built against the CasADi installed beside it at "
            f"{_impact_sibling_casadi}, but CasADi had already been imported "
            f"from {_impact_loaded_dir} by the time impact was imported, so the "
            "dynamic loader bound the extension to that copy's libcasadi. The "
            "two are different C++ builds exporting one SONAME, which is what "
            "the undefined symbol above means. Either import impact before "
            "CasADi, or take the other CasADi off sys.path/PYTHONPATH."
        ) from _exc
    raise

del _impact_sitedir, _impact_sibling_casadi
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
