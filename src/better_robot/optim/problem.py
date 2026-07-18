"""``LeastSquaresProblem`` — the glue between cost stack and solver.

Holds a ``CostStack``, a ``state_factory`` that wraps raw ``x`` into a
``ResidualState``, initial ``x0``, and optional box bounds. The solvers in
``optim/optimizers/`` own the iteration strategy.

The legacy compatibility lane memoizes one graph-free ``ResidualState`` by
tensor identity and mutation version, allowing residual and Jacobian calls
at the same iterate to share FK. Named-block ``Problem`` evaluation instead
uses an evaluation-local provider context.

See ``docs/concepts/solver_stack.md §4``.
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
    _cached_x: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)
    _cached_x_version: int | None = field(default=None, init=False, repr=False, compare=False)
    _cached_grad_enabled: bool | None = field(default=None, init=False, repr=False, compare=False)
    _cached_state: ResidualState | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def _nv(self) -> int:
        return self.nv if self.nv is not None else self.x0.shape[-1]

    @staticmethod
    def _tensor_version(x: torch.Tensor) -> int | None:
        """Return the in-place mutation counter, or ``None`` if unavailable."""
        try:
            return x._version
        except RuntimeError:
            # Inference tensors do not expose a version counter. Reusing a
            # state for one would make in-place mutation invisible, so those
            # evaluations intentionally take the uncached path.
            return None

    @staticmethod
    def _tracks_autograd(state: ResidualState) -> bool:
        """Whether retaining ``state`` would retain a live autograd graph."""
        if state.variables.requires_grad:
            return True
        data_values = vars(state.data).values()
        return any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in data_values
        )

    def _clear_state_cache(self) -> None:
        self._cached_x = None
        self._cached_x_version = None
        self._cached_grad_enabled = None
        self._cached_state = None

    def _state_at(self, x: torch.Tensor) -> ResidualState:
        """Build or reuse the evaluation state for this exact tensor iterate.

        This legacy one-entry memo lets solver loops pass an accepted
        trial tensor back as the next iterate, so its residual and Jacobian
        can share FK. Identity plus PyTorch's version counter prevents stale
        reuse after in-place mutation. States carrying autograd graphs are
        never retained, which keeps independent backward passes independent.

        New integrations should use named-block ``Problem`` and its
        evaluation-local provider context instead of extending this cache.
        """
        version = self._tensor_version(x)
        grad_enabled = torch.is_grad_enabled()
        if (
            version is not None
            and not x.requires_grad
            and self._cached_x is x
            and self._cached_x_version == version
            and self._cached_grad_enabled == grad_enabled
            and self._cached_state is not None
        ):
            return self._cached_state

        state = self.state_factory(x)
        if version is None or self._tracks_autograd(state):
            self._clear_state_cache()
        else:
            self._cached_x = x
            self._cached_x_version = version
            self._cached_grad_enabled = grad_enabled
            self._cached_state = state
        return state

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual vector at ``x``.

        Returns ``(dim,)`` for unbatched ``x``, ``(B..., dim)`` otherwise.

        See docs/concepts/solver_stack.md §4.
        """
        state = self._state_at(x)
        return self.cost_stack.residual(state)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Jacobian of the residual at ``x``.

        Returns ``(dim, nv)`` for unbatched ``x``.

        See docs/concepts/solver_stack.md §4.
        """
        state = self._state_at(x)
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
        ``docs/concepts/solver_stack.md §8``.
        """
        state = self._state_at(x)
        return self.cost_stack.gradient(state)

    def jacobian_blocks(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-item Jacobian dictionary for block-sparse trajopt solvers.

        Returns ``{name: J_i}`` where each ``J_i`` already includes the
        cost-stack item weight. Inactive items are omitted. Solvers that
        ignore this method continue to work via the dense
        :meth:`jacobian` path.

        See ``docs/concepts/solver_stack.md §8``.
        """
        from ..kinematics.jacobian import residual_jacobian

        state = self._state_at(x)
        blocks: dict[str, torch.Tensor] = {}
        for name, item in self.cost_stack.items.items():
            if not item.active:
                continue
            J_i = residual_jacobian(item.residual, state, strategy=self.jacobian_strategy)
            blocks[name] = J_i * item.weight
        return blocks
