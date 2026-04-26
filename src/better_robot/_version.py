"""Single source of truth for the package version.

Kept in step with ``pyproject.toml::project.version`` — bumped together
during the release procedure (see ``RELEASING.md``).
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
