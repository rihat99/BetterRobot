"""``better_robot.spatial`` — 6D spatial algebra value types.

Exposes ``Motion`` (twist), ``Force`` (wrench), ``Inertia``, and
``Symmetric3`` — thin dataclasses around ``torch.Tensor`` with named methods.
Heavy math routes through ``better_robot.lie``.

See ``docs/concepts/lie_and_spatial.md``.
"""

from __future__ import annotations

from ..lie.types import SE3, SO3, Pose
from .force import Force
from .inertia import Inertia
from .motion import Motion
from .symmetric3 import Symmetric3

__all__ = ["Motion", "Force", "Inertia", "Symmetric3", "SE3", "SO3", "Pose"]
