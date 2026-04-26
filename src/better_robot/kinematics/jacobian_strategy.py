"""``JacobianStrategy`` — enum controlling how residual Jacobians are computed.

See ``docs/concepts/kinematics.md §3``.
"""

from __future__ import annotations

from enum import Enum


class JacobianStrategy(str, Enum):
    """How to compute the Jacobian of a residual.

    ANALYTIC    — require ``residual.jacobian(state)`` to return a tensor.
    AUTODIFF    — ``torch.func.jacrev`` over ``residual.__call__``.
    FUNCTIONAL  — ``torch.func.jacfwd`` (useful when outputs << inputs).
    FINITE_DIFF — central finite differences through ``model.integrate``.
    AUTO        — prefer analytic, fall back to autodiff per-residual.
    """

    ANALYTIC = "analytic"
    AUTODIFF = "autodiff"
    FUNCTIONAL = "functional"
    FINITE_DIFF = "finite_diff"
    AUTO = "auto"
