"""``better_robot.lie`` — SE3/SO3 group operations and tangent algebra.

All pypose imports in the project live exclusively under
``better_robot.lie._pypose_backend``. The rest of the codebase uses the
functional facades in ``lie.se3``, ``lie.so3``, and ``lie.tangents``.

See ``docs/03_LIE_AND_SPATIAL.md``.
"""

from __future__ import annotations

from . import se3, so3, tangents

__all__ = ["se3", "so3", "tangents"]
