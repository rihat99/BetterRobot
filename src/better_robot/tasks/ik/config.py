"""IK solver configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class IKConfig:
    """Configuration for all IK solvers.

    All weights are non-negative floats. Larger values increase
    the influence of that cost term relative to others.
    """

    pos_weight: float = 1.0
    """Weight on position component of pose error."""

    ori_weight: float = 0.1
    """Weight on orientation component of pose error.

    Intentionally low: the Panda's default config has ~173° orientation from
    identity, which puts the SO3 log map near its singularity. High ori_weight
    causes oscillation.
    """

    pose_weight: float = 1.0
    """Overall pose cost scale."""

    limit_weight: float = 0.1
    """Soft joint limit penalty weight."""

    rest_weight: float = 0.01
    """Pull toward robot._default_cfg."""

    base_pos_weight: float = 2.0
    """Pull base position toward reference (floating-base only)."""

    base_ori_weight: float = 0.5
    """Pull base orientation toward reference (floating-base only)."""

    jacobian: Literal["autodiff", "analytic"] = "autodiff"
    """Jacobian computation mode.

    "autodiff": torch.func.jacrev (default, works for all cost terms).
    "analytic": geometric Jacobian (faster; fixed-base and floating-base supported).
    """

    solver: str = "lm"
    """Name of the solver to use from the SOLVERS registry. Default "lm".

    Available: "lm" (Levenberg-Marquardt), "lm_pypose" (PyPose LM).
    Floating-base IK always uses the PyPose LM path and ignores this field.
    """

    solver_params: dict = field(default_factory=dict)
    """Keyword arguments forwarded to the solver constructor.

    Example: IKConfig(solver="lm", solver_params={"damping": 1e-3, "reject": 8})
    """
