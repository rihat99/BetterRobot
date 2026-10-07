"""SE(3) and SO(3) group operations, tangent algebra and point-set alignment, as plain torch functions.

- Pose ``(..., 7)`` = ``[tx, ty, tz, qx, qy, qz, qw]``; rotation ``(..., 4)`` = ``[qx, qy, qz, qw]``. The quaternion
  scalar is last everywhere.
- SE(3) tangent (twist) ``(..., 6)`` = ``[vx, vy, vz, wx, wy, wz]``: linear first, angular second. SO(3) tangent
  ``(..., 3)`` = ``[wx, wy, wz]``.
- ``compose(a, b)`` is the matrix product ``a @ b`` (apply ``b`` first); ``exp``/``log`` are the true group
  exponential and logarithm, not a decoupled translation + rotation map.
- Public: the function modules ``se3``, ``so3``, ``tangents``; the value classes ``SE3``, ``SO3``, ``Pose``;
  ``umeyama``.
"""

from __future__ import annotations

from . import se3, so3, tangents
from .alignment import umeyama
from .types import SE3, SO3, Pose

__all__ = ["se3", "so3", "tangents", "umeyama", "SE3", "SO3", "Pose"]
