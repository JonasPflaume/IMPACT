"""Put ``python/`` on the path so the tests can import both packages.

``impact`` is normally installed, but ``examples`` deliberately is not -- it is
repository material, not part of the wheel -- so the tests reach it the same way
a user in a checkout does.

Appended rather than inserted at position 0: ``python/`` also holds the ``impact``
*source* tree, which has no compiled extension in it (the ``.so`` is installed
into ``site-packages/impact/`` by the wheel). Putting this directory ahead of
site-packages would shadow the installed solver with that extension-less copy and
every test would die on ``ImportError: cannot import name '_impact_core'``.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Import the solver before anything else can import CasADi. ``impact`` binds the
# process to the libcasadi its extension was compiled against, which it can only
# do while CasADi is still unimported -- see the import preamble in
# ``impact/__init__.py``. conftest runs before any test module, so doing it here
# makes the whole session safe regardless of collection order, and regardless of
# whether the environment happens to put another CasADi (a robotpkg tree on
# PYTHONPATH, say) ahead of the one installed beside the wheel.
import impact  # noqa: E402,F401
