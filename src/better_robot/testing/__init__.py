"""Checks shared by BetterRobot's and BetterHuman's test suites, benchmark runners and tools.

``imported_packages`` walks every import of a package for its edge-table test; ``dead_citations`` resolves every
repository path cited in tracked text (principle 7.8); ``check_joint_model`` checks a joint kind against the
`JointModel` contract and ``check_robust_kernel`` a robust kernel against the `RobustKernel` one; ``FakeServer``
stands in for viser's server under the viewer; ``public_objects`` walks the ``__all__`` tree a contract sweep checks,
``public_api`` describes it, ``snapshot_differences`` compares it with its snapshot (``python -m
better_robot.testing.api_snapshot`` writes and checks that file) and ``naming_problems`` checks its names; ``glossary``
reads BetterRobot's glossary page, which ships in the package, and spells names with its words; ``repository``
checks a repository's layout and agent files; ``bench`` measures the counters and wall time of a benchmark runner;
``snippets`` runs a Markdown page's Python fences and compares their output fences; ``no_skips`` is the pytest plugin
that fails every skip and xfail. Importing this package imports neither pytest nor an optional dependency:
``no_skips`` loads on first access.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from . import bench, glossary, repository, snippets
from .api import naming_problems, public_api, public_objects, snapshot_differences
from .citations import dead_citations
from .fake_server import FakeServer
from .import_edges import imported_packages
from .joints import check_joint_model
from .kernels import check_robust_kernel

__all__ = [
    "FakeServer",
    "bench",
    "check_joint_model",
    "check_robust_kernel",
    "dead_citations",
    "glossary",
    "imported_packages",
    "naming_problems",
    "no_skips",
    "public_api",
    "public_objects",
    "repository",
    "snapshot_differences",
    "snippets",
]


def __getattr__(name: str) -> ModuleType:
    # The plugin imports pytest, which a library user need not have; pytest itself imports it by module path.
    if name == "no_skips":
        return importlib.import_module(f"{__name__}.no_skips")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
