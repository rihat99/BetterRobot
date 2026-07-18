"""``JacobianStrategy`` — enum controlling how residual Jacobians are computed.

See ``docs/concepts/kinematics.md §3``.
"""

from __future__ import annotations

from enum import Enum


class JacobianStrategy(str, Enum):
    """How to compute the Jacobian of a residual.

    ANALYTIC    — require ``residual.jacobian(state)`` to return a tensor.
    FINITE_DIFF — central finite differences through ``model.integrate``.
    AUTO        — prefer analytic, fall back to finite differences.

    This enum belongs to the legacy ``CostStack`` lane. The named-block
    :class:`better_robot.optim.blocks.Problem` lane separately supports
    ``jacrev`` and ``jacfwd`` strategies through ``torch.func``.
    """

    ANALYTIC = "analytic"
    FINITE_DIFF = "finite_diff"
    AUTO = "auto"
