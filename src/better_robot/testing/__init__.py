"""Checks shared by BetterRobot's and BetterHuman's test suites (decision records 0013 and 0035).

``imported_packages`` walks every import of a package for its edge-table test; ``dead_citations`` resolves every
repository path cited in tracked text (principle 7.8); ``check_joint_model`` checks a joint kind against the
`JointModel` contract; ``FakeServer`` stands in for viser's server under the viewer; ``public_objects`` walks the
``__all__`` tree a contract sweep checks and ``public_api`` describes it for the API snapshot; ``glossary`` reads
BetterRobot's glossary page, which ships in the package, and spells a name with its words; ``snippets`` runs a
Markdown page's Python fences and compares their output fences; ``no_skips`` is the pytest plugin that fails every
skip and xfail. Importing this package imports neither pytest nor an optional
dependency: ``no_skips`` loads on first access.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from . import glossary, snippets
from .api import public_api, public_objects
from .citations import dead_citations
from .fake_server import FakeServer
from .import_edges import imported_packages
from .joints import check_joint_model

__all__ = [
    "FakeServer",
    "check_joint_model",
    "dead_citations",
    "glossary",
    "imported_packages",
    "no_skips",
    "public_api",
    "public_objects",
    "snippets",
]


def __getattr__(name: str) -> ModuleType:
    # The plugin imports pytest, which a library user need not have; pytest itself imports it by module path.
    if name == "no_skips":
        return importlib.import_module(f"{__name__}.no_skips")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
