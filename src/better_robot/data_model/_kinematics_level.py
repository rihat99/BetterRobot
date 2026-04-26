"""Standalone module for :class:`KinematicsLevel` to avoid circular imports
between ``data_model.__init__`` and ``data_model.data``.
"""

from __future__ import annotations

from enum import Enum


class KinematicsLevel(int, Enum):
    """How far the kinematic recursion has been advanced on a ``Data``.

    Each level is a strict superset of those below it; see
    ``docs/design/02_DATA_MODEL.md §3.1``.
    """

    NONE = 0
    PLACEMENTS = 1
    VELOCITIES = 2
    ACCELERATIONS = 3


__all__ = ["KinematicsLevel"]
