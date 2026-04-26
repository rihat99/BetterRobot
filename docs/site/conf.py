"""Sphinx configuration for the BetterRobot user docs.

Targets Sphinx 7+ with MyST + Furo + autodoc. Source roots are added
to ``sys.path`` so autodoc can import ``better_robot`` without an
editable install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ── Project metadata ───────────────────────────────────────────────────
project = "BetterRobot"
author = "BetterRobot contributors"
copyright = "BetterRobot contributors"

# Resolve the version from the package itself when available.
try:
    from better_robot import __version__ as _release  # type: ignore
except Exception:
    _release = "0.0.0"
release = _release
version = ".".join(_release.split(".")[:2])

# ── Extensions ─────────────────────────────────────────────────────────
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# Markdown is the primary source format; restructured text still works
# for the auto-generated API stubs.
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# ── MyST options ───────────────────────────────────────────────────────
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "substitution",
]
myst_heading_anchors = 3

# ── Theme ──────────────────────────────────────────────────────────────
html_theme = "furo"
html_title = f"BetterRobot {version}"
html_static_path: list[str] = []

# ── Autodoc ────────────────────────────────────────────────────────────
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# ── Intersphinx ────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# ── Napoleon (NumPy docstrings) ────────────────────────────────────────
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# ── Suppress imports of optional deps when autodoc can't satisfy them ──
autodoc_mock_imports = [
    "viser",
    "trimesh",
    "yourdfpy",
    "robot_descriptions",
    "scipy",
    "pinocchio",
    "pin",
]

# CI builds enable this with -W for strict warnings.
nitpicky = bool(int(os.environ.get("BR_DOCS_NITPICKY", "0")))
