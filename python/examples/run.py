#!/usr/bin/env python3
"""Run any example. Nothing is registered anywhere; the directories *are* the list.

    python python/examples/run.py list
    python python/examples/run.py push_t --visualize
    python python/examples/run.py push_circle --distance 1.5 --angle 225 --horizon 100
    python python/examples/run.py box --start 0 0 0 --goal 0.1 0.1 1.0 --tol 1e-6
    python python/examples/run.py box --print-config
    python python/examples/run.py push_t --render-only        # newest trajectory -> GIF

An example is a directory next to this file containing a ``main.py`` that defines
``main(argv=None) -> int``. That is the entire contract. This file finds those
directories, hands everything after the name to the example's own parser, and has
no opinion about what is inside -- so writing an example is writing a directory,
with nothing here to edit and no table to append to.

Which also means the dispatcher is never the only way in. These are the same run:

    python python/examples/run.py push_t --visualize
    python python/examples/push_t/main.py --visualize

``python -m examples.run ...`` works too, wherever ``python/`` is on the import
path (an editable install, or running from inside ``python/``).
"""

from __future__ import annotations

if __package__ in (None, ""):  # invoked by path: python python/examples/run.py ...
    import pathlib
    import sys as _sys

    # Append, never insert at 0: `python/` holds the `impact` source tree too, and
    # putting it ahead of site-packages would shadow the *installed* solver with a
    # copy that has no compiled extension -- an ImportError for `_impact_core`.
    _sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "examples"

import ast
import importlib
import pathlib
import sys
from typing import List, Optional

__all__ = ["main", "example_names", "example_summary"]

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent

#: Directories that sit here but are not examples.
_NOT_EXAMPLES = frozenset({"common", "notebooks"})


def example_names() -> List[str]:
    """Every example, found by looking rather than by asking a registry."""
    return sorted(path.parent.name for path in EXAMPLES_DIR.glob("*/main.py")
                  if path.parent.name not in _NOT_EXAMPLES
                  and not path.parent.name.startswith(("_", ".")))


def example_summary(name: str) -> str:
    """The first line of an example's ``main.py`` docstring.

    Parsed, not imported: importing every ``main.py`` to print a list would build
    each task's CasADi graph and import MuJoCo, which is a slow and fragile way to
    answer a question about text.
    """
    try:
        tree = ast.parse((EXAMPLES_DIR / name / "main.py").read_text())
    except (OSError, SyntaxError, ValueError):
        return ""
    doc = ast.get_docstring(tree) or ""
    return doc.splitlines()[0] if doc else ""


def _usage(names: List[str]) -> int:
    print(__doc__.strip())
    print("\nexamples:")
    _print_list(names)
    print("\nRun `<example> --help` for that example's own flags.")
    return 0


def _print_list(names: List[str]) -> None:
    width = max((len(name) for name in names), default=0)
    for name in names:
        print(f"  {name:<{width}}  {example_summary(name)}")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    names = example_names()

    if not argv or argv[0] in ("-h", "--help", "help"):
        return _usage(names)
    if argv[0] in ("list", "ls"):
        _print_list(names)
        return 0

    name = argv[0].replace("-", "_")
    if name not in names:
        print(f"error: no example named '{argv[0]}'; have {', '.join(names)}",
              file=sys.stderr)
        return 2

    # Imported here, not at module scope: `list` should not pay for importing the
    # solver, and this is the first point at which we know we are going to.
    from .common.cli import guard

    module = importlib.import_module(f".{name}.main", __package__)
    return guard(module.main, argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
