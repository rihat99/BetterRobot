"""``retarget`` — motion retargeting facade.

Reduces to ``solve_trajopt`` with ``keyframes`` built from per-step source
frame poses. Skeleton only in v1.

See ``docs/concepts/tasks.md §4``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data_model.model import Model
from .ik import OptimizerConfig
from .trajectory import Trajectory
from .trajopt import TrajOptResult


@dataclass
class RetargetCostConfig:
    """Weights for the retargeting cost stack."""

    pose_weight: float = 1.0
    smoothness_weight: float = 0.01
    limit_weight: float = 0.1


def retarget(
    source_model: Model,
    target_model: Model,
    source_trajectory: Trajectory,
    *,
    frame_map: dict[str, str],
    cost_cfg: RetargetCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
) -> TrajOptResult:
    """Find a target-model trajectory that tracks source-model frame poses.

    See docs/concepts/tasks.md §4.
    """
    raise NotImplementedError("see docs/concepts/tasks.md §4")
