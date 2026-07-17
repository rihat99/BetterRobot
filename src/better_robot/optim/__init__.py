"""``better_robot.optim`` — optimization problems and solver components.

The legacy ``solve`` wrapper consumes :class:`LeastSquaresProblem`. Named-block
:class:`Problem` values use the batched :class:`LevenbergMarquardt` or
:class:`GaussNewton` step API directly. Keeping both entry points explicit
until M2c prevents accidental dispatch into the deprecated legacy stack.

See ``docs/concepts/solver_stack.md``.
"""

from __future__ import annotations

from .blocks import (
    Bounds,
    Euclidean,
    GaussNewton,
    LevenbergMarquardt,
    LMState,
    LMStatus,
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

    Named-block :class:`Problem` instances use
    ``LevenbergMarquardt().run(values, problem)`` and must not be passed to
    this legacy convenience wrapper.

    See docs/concepts/solver_stack.md §5.
    """
    if isinstance(problem, Problem):
        problem.require_least_squares()
        raise TypeError(
            "better_robot.optim.solve accepts LeastSquaresProblem only; it is the "
            "deprecated wrapper. Use LevenbergMarquardt().run(values, problem) or "
            "GaussNewton().run(values, problem) for a named-block Problem."
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
    # Named-block evaluation and solver API. Deliberately qualified under ``optim``;
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
    "LevenbergMarquardt",
    "GaussNewton",
    "LMState",
    "LMStatus",
]
