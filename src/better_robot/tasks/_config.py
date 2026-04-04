"""IK solver configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class IKConfig:
    """Configuration for all IK solvers.

    All weights are non-negative floats. Larger values increase
    the influence of that cost term relative to others.
    """

    pos_weight: float = 1.0    # position component of pose error
    ori_weight: float = 0.1    # orientation component of pose error
    pose_weight: float = 1.0   # overall pose cost scale
    limit_weight: float = 0.1  # soft joint limit penalty
    rest_weight: float = 0.01  # pull toward robot._default_cfg
    jacobian: Literal["autodiff", "analytic"] = "autodiff"
    """Jacobian computation mode.

    "autodiff": torch.func.jacrev (default, works for all cost terms).
    "analytic": geometric Jacobian (faster; fixed-base and floating-base supported).
    """
