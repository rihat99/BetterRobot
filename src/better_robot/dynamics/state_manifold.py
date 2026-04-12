"""``StateMultibody`` — state manifold of a rigid-body system (Crocoddyl pattern).

Wraps ``model.integrate`` / ``model.difference`` into an abstract state
manifold that the future optimal-control solver consumes. Skeleton only.

See ``docs/06_DYNAMICS.md §7``.
"""

from __future__ import annotations

import torch

from ..data_model.model import Model


class StateMultibody:
    """State manifold ``x = (q, v)`` of a multibody system.

    ``nx = nq + nv`` (representation dim), ``ndx = 2 * nv`` (tangent dim).
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self.nx: int = model.nq + model.nv
        self.ndx: int = 2 * model.nv

    def zero(self) -> torch.Tensor:
        """Return the zero state ``(q_neutral, 0)``.

        See docs/06_DYNAMICS.md §7.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §7")

    def integrate(self, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """``x ⊕ dx`` on the multibody manifold.

        See docs/06_DYNAMICS.md §7.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §7")

    def diff(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """``x1 ⊖ x0`` on the multibody manifold.

        See docs/06_DYNAMICS.md §7.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §7")

    def jacobian_integrate(
        self, x: torch.Tensor, dx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``∂integrate/∂x``, ``∂integrate/∂dx``.

        See docs/06_DYNAMICS.md §7.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §7")

    def jacobian_diff(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``∂diff/∂x0``, ``∂diff/∂x1``.

        See docs/06_DYNAMICS.md §7.
        """
        raise NotImplementedError("see docs/06_DYNAMICS.md §7")
