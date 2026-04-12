"""``better_robot.spatial`` — 6D spatial algebra value types.

Exposes ``Motion`` (twist), ``Force`` (wrench), ``Inertia``, and
``Symmetric3`` — thin dataclasses around ``torch.Tensor`` with named methods.
Heavy math routes through ``better_robot.lie``.

See ``docs/03_LIE_AND_SPATIAL.md``.
"""

from __future__ import annotations

from .force import Force
from .inertia import Inertia
from .motion import Motion
from .symmetric3 import Symmetric3

__all__ = ["Motion", "Force", "Inertia", "Symmetric3"]
