"""Ensure ``better_robot`` is importable when running tests_v2 directly.

If the package has been installed via ``uv pip install -e .`` this conftest
is a no-op; otherwise it prepends ``<repo>/src`` to ``sys.path`` so the
tests still work.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
