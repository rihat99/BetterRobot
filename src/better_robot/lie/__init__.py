"""``better_robot.lie`` — SE3/SO3 group operations and tangent algebra.

The pure-PyTorch implementation lives directly in ``lie.se3`` and
``lie.so3``. Tangent algebra is in ``lie.tangents`` and typed value classes
are in ``lie.types``.

See ``docs/concepts/lie_and_spatial.md``.
"""

from __future__ import annotations

from . import se3, so3, tangents
from .alignment import umeyama
from .types import SE3, SO3, Pose

__all__ = ["se3", "so3", "tangents", "umeyama", "SE3", "SO3", "Pose"]
