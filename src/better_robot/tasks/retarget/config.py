"""Motion retargeting configuration (stub)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetargetConfig:
    """Configuration for motion retargeting.

    Not yet implemented — placeholder for future development.
    """

    pos_weight: float = 1.0
    """Weight on position matching."""

    ori_weight: float = 0.1
    """Weight on orientation matching."""

    rest_weight: float = 0.01
    """Pull toward model._q_default."""
