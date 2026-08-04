"""Working with :class:`~impact.AulaConfig` as data.

The config is the C++ struct itself rather than a Python mirror of it, so its
defaults are the solver's defaults by construction and cannot drift out of sync.
The cost of that is that it is not a dataclass: it has no keyword constructor and
no ``asdict``. This module supplies both, by reflecting over the struct rather
than by restating its ~60 fields anywhere.

    from impact import AulaConfig, apply_config, config_to_dict

    config = apply_config(AulaConfig(), horizon=100, max_outer_iters=500)
    json.dump(config_to_dict(config), open("run.json", "w"))
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ._impact_core import AulaConfig

__all__ = ["field_names", "config_to_dict", "config_from_dict", "apply_config",
           "VECTOR_FIELDS"]

#: Fields that take a vector rather than a scalar.
VECTOR_FIELDS = frozenset({"x_0", "x_goal", "cmd_lower", "cmd_upper"})


def field_names() -> Tuple[str, ...]:
    """Every settable hyper-parameter, sorted."""
    probe = AulaConfig()
    return tuple(sorted(name for name in dir(probe)
                        if not name.startswith("_") and not callable(getattr(probe, name))))


def config_to_dict(config: AulaConfig) -> Dict[str, Any]:
    """Every solver hyper-parameter as a plain dict.

    Useful for logging what a run actually used, and for round-tripping a config
    through JSON. Vector-valued fields come back as lists.
    """
    out: Dict[str, Any] = {}
    for name in field_names():
        value = getattr(config, name)
        out[name] = value.ravel().tolist() if isinstance(value, np.ndarray) else value
    return out


def apply_config(config: AulaConfig, **overrides) -> AulaConfig:
    """Set fields on an existing config, in place; returns it for chaining.

    An unknown name is an error naming the closest matches rather than a silently
    dropped setting -- a dropped ``rho_max`` still produces a perfectly plausible
    trajectory, which is the worst way for a typo to fail.
    """
    for name, value in overrides.items():
        if not hasattr(config, name):
            import difflib

            close = difflib.get_close_matches(name, field_names(), n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise AttributeError(f"AulaConfig has no hyper-parameter '{name}'.{hint}")
        setattr(config, name,
                np.asarray(value, dtype=float) if name in VECTOR_FIELDS else value)
    return config


def config_from_dict(values: Dict[str, Any], config: AulaConfig = None) -> AulaConfig:
    """Apply a dict of hyper-parameters onto a config (a fresh one by default)."""
    return apply_config(config if config is not None else AulaConfig(), **values)
