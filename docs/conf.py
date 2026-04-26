"""Sphinx configuration for the BetterRobot user docs.

The Sphinx source root is ``docs/``. The published site combines:

- Diátaxis-shaped user docs (``getting_started/``, ``tutorials/``,
  ``how_to/``, ``concepts/``, ``contributing/``).
- Normative specs (``design/``, ``conventions/``).
- Auto-generated API reference (``reference/api/``, built by
  ``sphinx-autodoc2``).
- Reference material (``reference/{changelog,roadmap,glossary}.md``).

Audit-trail / historical files in ``docs/legacy/`` and the in-repo
``CHANGELOG.md`` / ``UPDATE_PHASES.md`` / ``status/`` are excluded from
the build (see ``exclude_patterns``).

Build locally::

    make -C docs html
    # then open docs/_build/html/index.html

Strict mode (warnings → errors) is enabled with ``BR_DOCS_NITPICKY=1``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── Project metadata ───────────────────────────────────────────────────
project = "BetterRobot"
author = "BetterRobot contributors"
copyright = "BetterRobot contributors"

try:
    from better_robot import __version__ as _release  # type: ignore
except Exception:
    _release = "0.0.0"
release = _release
version = ".".join(_release.split(".")[:2])

# ── Extensions ─────────────────────────────────────────────────────────
extensions = [
    # Markdown source. ``myst_nb`` is in [docs] for future executable-
    # notebook tutorials; it is not loaded here because we have no
    # ``.ipynb`` files yet and double-loading with ``myst_parser`` errors
    # on duplicate config registration.
    "myst_parser",
    # Auto-API
    "autodoc2",
    # Theme niceties
    "sphinx_design",
    "sphinx_copybutton",
    # Standard
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
]

# ── Source file types ──────────────────────────────────────────────────
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"

# Patterns excluded from the build. Anything that should not appear in
# the published site goes here.
exclude_patterns = [
    "_build",
    "_static",
    "Thumbs.db",
    ".DS_Store",
    # In-repo audit trail / historical material (kept on disk for forensics).
    "legacy/**",
    # In-repo project files that ship with the source tree but are not user docs.
    "CHANGELOG.md",
    "UPDATE_PHASES.md",
    "status/**",
    "README.md",
]

# ── MyST options ───────────────────────────────────────────────────────
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "substitution",
    "attrs_inline",
    "attrs_block",
    "smartquotes",
]
myst_heading_anchors = 3
myst_url_schemes = ["http", "https", "mailto", "ftp"]

# ── Autodoc2 (auto API generator) ──────────────────────────────────────
autodoc2_packages = [
    {
        "path": "../src/better_robot",
        "auto_mode": True,
    },
]
# Render API pages as MyST so cross-refs match the rest of the site.
autodoc2_render_plugin = "myst"
# Where the generated .md tree lands (relative to source root).
autodoc2_output_dir = "reference/api"
autodoc2_index_template = None  # use the autodoc2-generated index
# Skip optional deps that aren't installed in the docs build env.
autodoc2_replace_annotations = []
autodoc2_replace_bases = []
autodoc2_skip_module_regexes = [
    r"better_robot\..*\._.*",  # private submodules
]
autodoc2_hidden_objects = ["dunder", "private", "inherited"]
autodoc2_class_docstring = "both"
# Intentionally no `autodoc2_module_all_regexes`: with that regex set,
# the top-level package toctree only listed submodules whose names
# happen to match a symbol in ``__all__``, so most of the library was
# missing from the rendered API index.

# ── Theme ──────────────────────────────────────────────────────────────
html_theme = "sphinx_book_theme"
html_title = "BetterRobot"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sphinx = False
html_show_sourcelink = False
html_last_updated_fmt = ""

# sphinx-book-theme options. Kept deliberately close to IsaacLab's
# config so the rendered look matches.
html_theme_options = {
    "path_to_docs": "docs/",
    "repository_url": "https://github.com/rihat99/BetterRobot",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "use_sidenotes": True,
    "navigation_with_keys": True,
    "logo": {
        "text": "BetterRobot",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rihat99/BetterRobot",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/better-robot/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
}

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ],
}

# ── Autodoc / Napoleon ─────────────────────────────────────────────────
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True

# ── Intersphinx ────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "trimesh": ("https://trimesh.org/", None),
}

# ── Copy button ────────────────────────────────────────────────────────
copybutton_prompt_text = r">>> |\$ |# "
copybutton_prompt_is_regexp = True

# ── Misc ───────────────────────────────────────────────────────────────
todo_include_todos = True
add_module_names = False

# CI builds enable strict mode with -W. Set BR_DOCS_NITPICKY=1 locally to
# match.
nitpicky = bool(int(os.environ.get("BR_DOCS_NITPICKY", "0")))
nitpick_ignore = [
    # Optional deps that aren't always installed.
    ("py:class", "viser.ViserServer"),
    ("py:class", "trimesh.Trimesh"),
    ("py:class", "yourdfpy.URDF"),
    ("py:class", "mujoco.MjModel"),
]
