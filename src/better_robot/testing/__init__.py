"""Repository checks shared by BetterRobot's and BetterHuman's test suites (decision record 0035).

``imported_packages`` walks every import of a package for its edge-table test; ``dead_citations`` resolves every
repository path cited in tracked text (principle 7.8). Both use the standard library only, so importing this module
pulls in neither pytest nor an optional dependency.
"""

from __future__ import annotations

from .citations import dead_citations
from .import_edges import imported_packages

__all__ = ["dead_citations", "imported_packages"]
