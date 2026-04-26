"""``Optimizer`` protocol.

Every optimiser takes the same keyword components (``linear_solver``,
``kernel``, ``strategy``, ``scheduler``) — replacing one swaps one knob
without touching the optimisation loop. Every optimiser returns a
:class:`~better_robot.optim.state.SolverState` (the old
``OptimizationResult`` name is kept as a deprecated alias).

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5, §9`` and
``docs/conventions/15_EXTENSION.md §3``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..problem import LeastSquaresProblem
from ..state import SolverState

# Deprecated alias — remove in v1.1. Kept so that user code written
# against the old ``OptimizationResult`` name imports continue to work.
OptimizationResult = SolverState


@runtime_checkable
class Optimizer(Protocol):
    """Optimiser protocol. Concrete classes live under ``optim/optimizers/``.

    Marked ``@runtime_checkable`` so the extension-seam docs can advertise
    ``isinstance(obj, Optimizer)`` as a valid contract check
    (``docs/conventions/15_EXTENSION.md §3``).
    """

    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int,
        linear_solver,
        kernel,
        strategy,
        scheduler=None,
    ) -> SolverState:
        ...
