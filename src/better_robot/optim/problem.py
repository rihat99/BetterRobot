"""``LeastSquaresProblem`` — the glue between cost stack and solver.

Holds a ``CostStack``, a ``state_factory`` that wraps raw ``x`` into a
``ResidualState``, initial ``x0``, and optional box bounds. The solvers in
``optim/optimizers/`` own the iteration strategy.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §4``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ..costs.stack import CostStack
from ..kinematics.jacobian_strategy import JacobianStrategy
from ..residuals.base import ResidualState


@dataclass
class LeastSquaresProblem:
    """Least-squares problem over a manifold-valued variable ``x ∈ (nq,)``."""

    cost_stack: CostStack
    state_factory: Callable[[torch.Tensor], ResidualState]
    x0: torch.Tensor
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO
    # nv: velocity / tangent space dimension. If None, assumed equal to x0.shape[-1].
    nv: int | None = None
    # Optional manifold retraction: retract(x, delta_v) -> x_new.
    # If None, uses Euclidean update x + delta_v.
    retract: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    @property
    def _nv(self) -> int:
        return self.nv if self.nv is not None else self.x0.shape[-1]

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual vector at ``x``.

        Returns ``(dim,)`` for unbatched ``x``, ``(B..., dim)`` otherwise.

        See docs/design/07_RESIDUALS_COSTS_SOLVERS.md §4.
        """
        state = self.state_factory(x)
        return self.cost_stack.residual(state)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Jacobian of the residual at ``x``.

        Returns ``(dim, nv)`` for unbatched ``x``.

        See docs/design/07_RESIDUALS_COSTS_SOLVERS.md §4.
        """
        state = self.state_factory(x)
        return self.cost_stack.jacobian(state, strategy=self.jacobian_strategy)

    def step(self, x: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor:
        """Apply a velocity-space update ``delta_v ∈ R^{nv}`` to ``x``.

        Uses the supplied ``retract`` if available, otherwise Euclidean ``+``.
        """
        if self.retract is not None:
            return self.retract(x, delta_v)
        # Euclidean fallback (valid when nq == nv)
        return x + delta_v

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-free gradient of ``0.5 · ‖r(x)‖²`` w.r.t. tangent variables.

        Iterates over active items in the cost stack; each contributes
        ``w² · J_iᵀ r_i``. Residuals that override ``apply_jac_transpose``
        skip the dense Jacobian; everything else falls back to
        ``J_iᵀ @ r_i`` through ``residual_jacobian``.

        Returns ``(nv,)`` for unbatched ``x``. See
        ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §8``.
        """
        state = self.state_factory(x)
        return self.cost_stack.gradient(state)

    def jacobian_blocks(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-item Jacobian dictionary for block-sparse trajopt solvers.

        Returns ``{name: J_i}`` where each ``J_i`` already includes the
        cost-stack item weight. Inactive items are omitted. Solvers that
        ignore this method continue to work via the dense
        :meth:`jacobian` path.

        See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §8``.
        """
        from ..kinematics.jacobian import residual_jacobian

        state = self.state_factory(x)
        blocks: dict[str, torch.Tensor] = {}
        for name, item in self.cost_stack.items.items():
            if not item.active:
                continue
            J_i = residual_jacobian(item.residual, state, strategy=self.jacobian_strategy)
            blocks[name] = J_i * item.weight
        return blocks
