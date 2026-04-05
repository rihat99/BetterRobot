"""Trajectory optimization configuration (stub)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajOptConfig:
    """Configuration for trajectory optimization.

    Not yet implemented — placeholder for future development.
    """

    horizon: int = 50
    """Number of time steps in the trajectory."""

    dt: float = 0.05
    """Time step duration (seconds)."""

    pos_weight: float = 1.0
    """Weight on end-effector position cost."""

    ori_weight: float = 0.1
    """Weight on end-effector orientation cost."""

    velocity_weight: float = 0.1
    """Weight on joint velocity regularization."""

    acceleration_weight: float = 0.01
    """Weight on joint acceleration regularization."""

    jerk_weight: float = 0.001
    """Weight on joint jerk regularization."""
