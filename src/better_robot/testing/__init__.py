"""Checks shared by BetterRobot's and BetterHuman's test suites (decision records 0013 and 0035).

``imported_packages`` walks every import of a package for its edge-table test; ``dead_citations`` resolves every
repository path cited in tracked text (principle 7.8); ``check_joint_model`` checks a joint kind against the
`JointModel` contract; ``FakeServer`` stands in for viser's server under the viewer; ``public_objects`` walks the
``__all__`` tree a contract sweep checks and ``public_api`` describes it for the API snapshot. None imports pytest or
an optional dependency.
"""

from __future__ import annotations

from .api import public_api, public_objects
from .citations import dead_citations
from .fake_server import FakeServer
from .import_edges import imported_packages
from .joints import check_joint_model

__all__ = [
    "FakeServer",
    "check_joint_model",
    "dead_citations",
    "imported_packages",
    "public_api",
    "public_objects",
]
