"""``Cost.factory(residual_fn)`` — convenience wrapper for ad-hoc residuals.

Lets users pass a plain function as a residual by wrapping it in a
throwaway ``Residual`` class (autodiff Jacobian only).

See ``docs/concepts/residuals_and_costs.md``.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..residuals.base import Residual, ResidualState


def factory(
    residual_fn: Callable[[ResidualState], torch.Tensor],
    *,
    dim: int,
    name: str = "adhoc",
) -> Residual:
    """Wrap a plain function ``state → residual`` as a ``Residual`` object.

    See docs/concepts/residuals_and_costs.md.
    """
    raise NotImplementedError("see docs/concepts/residuals_and_costs.md")
