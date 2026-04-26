"""``better_robot.optim`` — problem + pluggable solver components.

``LeastSquaresProblem`` is the only interface the optimiser sees. Every
optimiser takes a problem, a linear solver, a robust kernel, a damping
strategy, and an optional stop scheduler. Replacing one knob swaps one
class.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md``.
"""

from __future__ import annotations

from .jacobian_spec import ResidualSpec
from .optimizers.base import OptimizationResult, Optimizer
from .problem import LeastSquaresProblem
from .state import SolverState


def solve(
    problem: LeastSquaresProblem,
    *,
    optimizer: Optimizer | None = None,
    max_iter: int = 50,
    linear_solver=None,
    kernel=None,
    strategy=None,
    scheduler=None,
) -> SolverState:
    """One-shot convenience wrapper — build the default LM optimiser and run it.

    See docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5.
    """
    from .optimizers.levenberg_marquardt import LevenbergMarquardt

    opt = optimizer if optimizer is not None else LevenbergMarquardt()
    return opt.minimize(
        problem,
        max_iter=max_iter,
        linear_solver=linear_solver,
        kernel=kernel,
        strategy=strategy,
        scheduler=scheduler,
    )


__all__ = [
    "LeastSquaresProblem",
    "Optimizer",
    "OptimizationResult",  # deprecated alias for SolverState
    "SolverState",
    "ResidualSpec",
    "solve",
]
