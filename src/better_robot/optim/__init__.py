"""``better_robot.optim`` — optimization problems and solver components.

The legacy solver stack consumes :class:`LeastSquaresProblem`.  M2a's
named-block :class:`Problem` is an evaluation and dense-assembly interface;
GN/LM integration follows in M2b.  Keeping both names explicit prevents a
block problem from being passed to a solver that cannot yet honor its scalar
objectives, masks, or provider DAG.

See ``docs/concepts/solver_stack.md``.
"""

from __future__ import annotations

from .blocks import (
    Bounds,
    Euclidean,
    ObjectiveItem,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    SE3Manifold,
    SO3Manifold,
    Values,
    VarSpec,
    detach_values,
)
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
    """Run the legacy :class:`LeastSquaresProblem` LM solver stack.

    Named-block :class:`Problem` instances are evaluation-only until M2b and
    must not be passed to this convenience wrapper.

    See docs/concepts/solver_stack.md §5.
    """
    if isinstance(problem, Problem):
        problem.require_least_squares()
        raise TypeError(
            "better_robot.optim.solve accepts LeastSquaresProblem only; named-block "
            "Problem solving lands in M2b. Use Problem.gradient/Problem.retract in a "
            "first-order loop for now."
        )
    from .optimizers.levenberg_marquardt import LevenbergMarquardt  # noqa: PLC0415

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
    # M2a named-block evaluation API.  Deliberately qualified under ``optim``;
    # the package root keeps its existing Lie ``SE3`` identity.
    "Bounds",
    "Euclidean",
    "SO3Manifold",
    "SE3Manifold",
    "RobotConfig",
    "Values",
    "VarSpec",
    "Problem",
    "ResidualItem",
    "ObjectiveItem",
    "RobotStateProvider",
    "detach_values",
]
