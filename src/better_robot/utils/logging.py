"""Library-wide logger.

See ``docs/01_ARCHITECTURE.md``.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("better_robot")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced child logger under ``better_robot``."""
    return LOGGER.getChild(name) if name else LOGGER
