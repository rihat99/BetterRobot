"""``better_robot.lie`` — SE3/SO3 group operations and tangent algebra.

The pure-PyTorch implementation lives in ``lie._torch_native_backend``.
The rest of the codebase uses the functional facades in ``lie.se3``,
``lie.so3``, and ``lie.tangents``, plus the typed value classes in
``lie.types``.

See ``docs/concepts/lie_and_spatial.md``.
"""

from __future__ import annotations

from . import se3, so3, tangents
from .types import SE3, SO3, Pose

__all__ = ["se3", "so3", "tangents", "SE3", "SO3", "Pose"]
