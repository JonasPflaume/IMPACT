"""Generic MPCC assembly: the Python side of ``buildMPCC``.

This module derives the augmented-Lagrangian least-squares residual of an MPCC
and hands the resulting CasADi functions to the C++ solver.

    minimize    || cost(z, p) ||^2
    subject to  an ordered list of equality / inequality / complementarity blocks

It is a direct port of ``impact_solver/src/mpcc_subproblem.cpp``, and deliberately
a literal one. Row order, parameter-buffer offsets and the saddle layout are not
implementation details the solver can rediscover -- they are the contract it
reads (``SaddleLayout`` tells it where each block's penalty lives, ``DualBlock``
offsets tell ``syncParams`` where to scatter the multipliers). Any reordering
here would be silently absorbed and would change the arithmetic. Where a line
looks gratuitously specific -- ``fmax`` rather than ``if`` -- it is matching the
C++ emission, and ``tests/test_parity.py`` compares the two builds numerically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Union

import casadi as ca

from . import _impact_core as _core

__all__ = [
    "ConstraintKind",
    "BlockOptions",
    "MPCCConstraint",
    "MPCCDescription",
    "BuiltMPCC",
    "build_mpcc",
]

SXLike = Union[ca.SX, Sequence[ca.SX]]


def _vcat(expr: SXLike) -> ca.SX:
    """Accept either a single expression or a list of stage expressions."""
    if isinstance(expr, (list, tuple)):
        return ca.vertcat(*expr) if expr else ca.SX(0, 1)
    return expr


class ConstraintKind(Enum):
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    COMPLEMENTARITY = "complementarity"


@dataclass
class BlockOptions:
    """Per-block AuLa tuning.

    ``scale`` conditions the residual, ``rho_init`` seeds the safeguarded penalty
    and ``tol`` is this block's unscaled convergence tolerance.
    """

    scale: float = 1.0
    rho_init: float = 1.0
    tol: float = 1e-5


@dataclass
class MPCCConstraint:
    kind: ConstraintKind
    name: str
    c: Optional[ca.SX] = None  # equality / inequality residual
    G: Optional[ca.SX] = None  # complementarity legs
    H: Optional[ca.SX] = None
    scale: float = 1.0
    rho_init: float = 1.0
    tol: float = 1e-5


@dataclass
class MPCCDescription:
    """A generic MPCC, independent of any trajectory structure."""

    z: ca.SX
    cost: ca.SX
    p: Optional[ca.SX] = None
    cost_is_linear: bool = False
    constraints: List[MPCCConstraint] = field(default_factory=list)

    check_stationarity: bool = True
    conditioned_complementarity: bool = True
    stationarity_tol: float = 1e-5
    max_stagnation_restarts: int = 2

    # Compile the residual/Jacobian through a C compiler instead of interpreting.
    jit: bool = False
    #: Flags for the JIT compile. -O0 is the C++ default and is deliberate: SX
    #: codegen emits one huge flat C function, and gcc's optimizer scales badly on
    #: it. Whether compiled evaluation actually beats CasADi's VM is
    #: problem-dependent -- measure before enabling.
    jit_flags: Sequence[str] = ("-O0",)

    def add_equality(self, name: str, c: SXLike, opts: BlockOptions = BlockOptions()) -> None:
        self.constraints.append(
            MPCCConstraint(ConstraintKind.EQUALITY, name, c=_vcat(c), scale=opts.scale,
                           rho_init=opts.rho_init, tol=opts.tol))

    def add_inequality(self, name: str, c: SXLike, opts: BlockOptions = BlockOptions()) -> None:
        self.constraints.append(
            MPCCConstraint(ConstraintKind.INEQUALITY, name, c=_vcat(c), scale=opts.scale,
                           rho_init=opts.rho_init, tol=opts.tol))

    def add_complementarity(self, name: str, G: SXLike, H: SXLike,
                            opts: BlockOptions = BlockOptions()) -> None:
        self.constraints.append(
            MPCCConstraint(ConstraintKind.COMPLEMENTARITY, name, G=_vcat(G), H=_vcat(H),
                           scale=opts.scale, rho_init=opts.rho_init, tol=opts.tol))


@dataclass
class BuiltMPCC:
    """An assembled subproblem plus the offset of the user runtime parameter ``p``.

    Write ``p`` at ``off_p`` in the parameter buffer before solving.

    ``residual`` / ``jacobian`` / ``gh`` are the CasADi functions handed to the
    solver. They are kept here because the solver only takes their serialized
    form, which leaves nothing on this side to inspect otherwise -- and being able
    to evaluate exactly what the solver evaluates is what makes the comparison
    against the C++ builders in ``tests/test_parity.py`` meaningful.
    """

    subproblem: "_core.Subproblem"
    off_p: int
    n_opt: int
    n_params: int
    residual: Optional[ca.Function] = None
    jacobian: Optional[ca.Function] = None
    gh: Optional[ca.Function] = None


def _function_options(jit: bool, jit_flags: Sequence[str] = ("-O0",)) -> tuple:
    """CasADi options for the emitted functions; ``(fopts, jopt)``."""
    fopts = {}
    if jit:
        fopts = {
            "jit": True,
            "compiler": "shell",
            "jit_cleanup": True,
            # SX codegen emits one huge flat C function, so gcc's optimizer scales
            # badly on it: -O2/-O3 can take minutes on the contact problems.
            "jit_options": {"compiler": "gcc", "flags": list(jit_flags)},
        }
    jopt = dict(fopts)
    jopt["enable_fd"] = False
    return fopts, jopt


def build_mpcc(desc: MPCCDescription) -> BuiltMPCC:
    """Assemble ``desc`` into a solver-ready subproblem.

    Both shooting builders end here; a non-trajectory MPCC can call it directly.
    """
    z = desc.z
    if z.numel() == 0 or z.size2() != 1:
        raise ValueError("build_mpcc: z must be a non-empty column vector")
    cost = desc.cost if desc.cost is not None else ca.SX(0, 1)
    if cost.numel() != 0 and cost.size2() != 1:
        raise ValueError("build_mpcc: cost must be a column residual vector")
    if desc.check_stationarity and not (desc.stationarity_tol > 0.0 and
                                        math.isfinite(desc.stationarity_tol)):
        raise ValueError("build_mpcc: stationarity_tol must be finite and positive")
    if desc.max_stagnation_restarts < 0:
        raise ValueError("build_mpcc: max_stagnation_restarts must be nonnegative")

    for cc in desc.constraints:
        for label, v in (("scale", cc.scale), ("rho_init", cc.rho_init), ("tol", cc.tol)):
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(
                    f"build_mpcc: block '{cc.name}' {label} must be finite and positive")
        if cc.kind is ConstraintKind.COMPLEMENTARITY:
            if (cc.G is None or cc.H is None or cc.G.numel() == 0 or cc.G.size2() != 1
                    or cc.H.size2() != 1 or cc.G.size1() != cc.H.size1()):
                raise ValueError(f"build_mpcc: complementarity block '{cc.name}' needs "
                                 "equally-sized non-empty column legs")
        elif cc.c is None or cc.c.numel() == 0 or cc.c.size2() != 1:
            raise ValueError(
                f"build_mpcc: constraint block '{cc.name}' must be a non-empty column vector")

    n_opt = int(z.size1())
    p_user = desc.p if desc.p is not None else ca.SX(0, 1)
    np_user = int(p_user.size1()) if p_user.numel() else 0
    n_cost_rows = int(cost.size1())
    linear = desc.cost_is_linear

    # --- symbolic AuLa parameters, in the order the solver's syncParams expects --
    psyms: List[ca.SX] = []
    off = 0

    def add_param(sym: ca.SX) -> int:
        nonlocal off
        psyms.append(sym)
        here = off
        off += int(sym.size1())
        return here

    off_rho_one = add_param(ca.SX.sym("rho_one", 1))

    duals = []  # dicts mirroring DualBlockSpec, in residual-row order
    comps = []  # dicts mirroring CompBlockSpec
    saddle_blocks = []  # (count, rho_offset), residual-row order

    res: List[ca.SX] = [cost]
    for cc in desc.constraints:
        if cc.kind is ConstraintKind.COMPLEMENTARITY:
            n = int(cc.G.size1())
            suffix = str(len(comps))
            off_sG = add_param(ca.SX.sym("sG_" + suffix, n))
            off_sH = add_param(ca.SX.sym("sH_" + suffix, n))
            off_kappaG = add_param(ca.SX.sym("kappaG_" + suffix, n))
            off_kappaH = add_param(ca.SX.sym("kappaH_" + suffix, n))
            off_rho = add_param(ca.SX.sym("rho_comp_" + suffix, 1))
            sG, sH, kappaG, kappaH, rho = psyms[-5], psyms[-4], psyms[-3], psyms[-2], psyms[-1]

            res.append(ca.sqrt(rho) * (cc.scale * (cc.G - sG) + kappaG / rho))
            res.append(ca.sqrt(rho) * (cc.scale * (cc.H - sH) + kappaH / rho))

            comps.append(dict(name=cc.name or ("comp_" + suffix), dim=n, scale=cc.scale,
                              rho_init=cc.rho_init, tol=cc.tol, sG_offset=off_sG,
                              sH_offset=off_sH, kappaG_offset=off_kappaG,
                              kappaH_offset=off_kappaH, rho_offset=off_rho,
                              G=cc.G, H=cc.H))
            saddle_blocks.append((n, off_rho))
            saddle_blocks.append((n, off_rho))
        else:
            n = int(cc.c.size1())
            off_kappa = add_param(ca.SX.sym("kappa_" + cc.name, n))
            off_rho = add_param(ca.SX.sym("rho_" + cc.name, 1))
            kappa, rho = psyms[-2], psyms[-1]
            ineq = cc.kind is ConstraintKind.INEQUALITY
            if ineq:
                res.append(ca.sqrt(rho) * ca.fmax(cc.scale * cc.c + kappa / rho, 0.0))
            else:
                res.append(ca.sqrt(rho) * (cc.scale * cc.c + kappa / rho))
            duals.append(dict(name=cc.name, inequality=ineq, dim=n, scale=cc.scale,
                              rho_init=cc.rho_init, tol=cc.tol, kappa_offset=off_kappa,
                              rho_offset=off_rho, scaled=cc.scale * cc.c))
            saddle_blocks.append((n, off_rho))

    off_p = off
    if np_user > 0:
        psyms.append(p_user)
        off += np_user
    n_params = off
    p_full = ca.vertcat(*psyms)

    # ------------------------------------------------------------- functions --
    fopts, jopt = _function_options(desc.jit, desc.jit_flags)
    r = ca.vertcat(*res)

    residual_func = ca.Function("mpcc_residual", [z, p_full], [r], fopts)
    jacobian_func = ca.Function("mpcc_jac", [z, p_full], [ca.jacobian(r, z)], jopt)

    if comps:
        gh_func = ca.Function("GH", [z, p_full],
                              [ca.vertcat(*[c["G"] for c in comps]),
                               ca.vertcat(*[c["H"] for c in comps])], fopts)
    else:
        gh_func = ca.Function("GH", [z, p_full], [ca.SX.zeros(0, 1), ca.SX.zeros(0, 1)], fopts)

    obj_func = ca.Function("obj", [z, p_full], [ca.sumsqr(cost)], fopts)

    spec = _core.SubproblemSpec()
    spec.n_opt = n_opt
    spec.n_params = n_params
    spec.residual = residual_func.serialize()
    spec.jacobian = jacobian_func.serialize()
    spec.gh = gh_func.serialize()
    spec.obj = obj_func.serialize()

    if desc.max_stagnation_restarts > 0:
        spec.obj_grad = ca.Function("obj_grad", [z, p_full],
                                    [ca.gradient(ca.sumsqr(cost), z)], fopts).serialize()
    if desc.check_stationarity:
        spec.stationarity = ca.Function("aug_stationarity", [z, p_full],
                                        [ca.gradient(ca.sumsqr(r), z)], fopts).serialize()

    # -------------------------------------------------------- saddle layout ---
    # Linear cost rows are constant and fold into the constant block; nonlinear
    # cost rows are a penalty block at rho = 1.
    layout = _core.SaddleLayout()
    layout.n_z = n_opt
    layout.n_cost = n_cost_rows if linear else 0
    blocks = []
    row = layout.n_cost
    if not linear and n_cost_rows > 0:
        blocks.append(_core.SaddleBlock(row, n_cost_rows, off_rho_one))
        row += n_cost_rows
    for count, rho_off in saddle_blocks:
        if count > 0:
            blocks.append(_core.SaddleBlock(row, count, rho_off))
            row += count
    layout.blocks = blocks
    layout.n_dual = row - layout.n_cost
    spec.layout = layout

    # ------------------------------------------------------------- channels ---
    dual_specs = []
    for d in duals:
        b = _core.DualBlockSpec()
        b.name = d["name"]
        b.inequality = d["inequality"]
        b.dim = d["dim"]
        b.scale = d["scale"]
        b.rho_init = d["rho_init"]
        b.tol = d["tol"]
        b.kappa_offset = d["kappa_offset"]
        b.rho_offset = d["rho_offset"]
        b.eval_scaled = ca.Function("eval_" + d["name"], [z, p_full], [d["scaled"]]).serialize()
        dual_specs.append(b)
    spec.dual_blocks = dual_specs

    comp_specs = []
    for c in comps:
        b = _core.CompBlockSpec()
        b.name = c["name"]
        b.dim = c["dim"]
        b.scale = c["scale"]
        b.rho_init = c["rho_init"]
        b.tol = c["tol"]
        b.sG_offset = c["sG_offset"]
        b.sH_offset = c["sH_offset"]
        b.kappaG_offset = c["kappaG_offset"]
        b.kappaH_offset = c["kappaH_offset"]
        b.rho_offset = c["rho_offset"]
        comp_specs.append(b)
    spec.comp_blocks = comp_specs

    spec.check_stationarity = desc.check_stationarity
    spec.conditioned_complementarity = desc.conditioned_complementarity
    spec.stationarity_tol = desc.stationarity_tol
    spec.max_stagnation_restarts = desc.max_stagnation_restarts
    spec.param_values = [(off_rho_one, [1.0])]

    return BuiltMPCC(subproblem=_core.build_subproblem(spec), off_p=off_p, n_opt=n_opt,
                     n_params=n_params, residual=residual_func, jacobian=jacobian_func,
                     gh=gh_func)
