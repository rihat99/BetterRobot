"""Whole-body inverse kinematics as a named-block optimization preset.

The task facade owns no solver loop. It assembles one ``q`` variable on the
robot-configuration manifold, installs the built-in kinematic residuals and a
lazy robot-state provider, then runs a public named-block solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics
from ..optim import (
    GaussNewton,
    JacobianStrategy,
    LevenbergMarquardt,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    VarSpec,
    run_first_order,
)
from ..optim import Cauchy, Cholesky, Huber, L2, Tukey
from ..residuals.limits import JointPositionLimit
from ..residuals.pose import PoseResidual
from ..residuals.regularization import RestResidual

if TYPE_CHECKING:
    from ..data_model.data import Data


_OptimizerName = Literal[
    "lm",
    "gn",
    "adam",
    "lbfgs",
    "lm_then_adam",
    "lm_then_lbfgs",
]

_OPTIMIZER_NAMES = frozenset({"lm", "gn", "adam", "lbfgs", "lm_then_adam", "lm_then_lbfgs"})
_LINEAR_SOLVER_NAMES = frozenset({"cholesky"})
_KERNEL_NAMES = frozenset({"l2", "huber", "cauchy", "tukey"})
_DAMPING_NAMES = frozenset({"constant", "adaptive"})


@dataclass
class IKCostConfig:
    """Weights for the built-in IK residuals."""

    pos_weight: float = 1.0
    ori_weight: float = 1.0
    pose_weight: float = 1.0
    limit_weight: float = 0.1
    rest_weight: float = 0.01
    q_rest: torch.Tensor | None = None


@dataclass
class OptimizerConfig:
    """Named-block solver selection and hyperparameters.

    ``lm_then_adam`` runs LM followed by the ``torch.optim`` adapter. Batched
    L-BFGS is deliberately not exposed by the named-block
    stack yet; the retained ``lbfgs`` spellings fail with an actionable error.
    Linear-solver and Jacobian settings apply to LM/GN phases; damping applies
    only to configurations with an LM phase.
    """

    optimizer: _OptimizerName = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = "auto"
    linear_solver: Literal["cholesky"] = "cholesky"
    kernel: Literal["l2", "huber", "cauchy", "tukey"] = "l2"
    damping: Literal["constant", "adaptive"] = "adaptive"
    tol: float = 1e-6
    refine_disabled_items: tuple[str, ...] = ()


@dataclass
class IKResult:
    """Result of an unbatched or independently batched IK solve."""

    q: torch.Tensor
    residual: torch.Tensor
    iters: int | torch.Tensor
    converged: bool | torch.Tensor
    model: Model

    def fk(self) -> "Data":
        """Return FK data at the solution, preserving leading batch axes."""
        return forward_kinematics(self.model, self.q, compute_frames=True)

    def frame_pose(self, name: str) -> torch.Tensor:
        """Return one named frame pose at the solution."""
        data = self.fk()
        frame_id = self.model.frame_id(name)
        return data.frame_pose_world[..., frame_id, :]


def _validate_optimizer_config(config: OptimizerConfig) -> None:
    """Validate facade policy before building residuals or solver state."""
    if not isinstance(config.optimizer, str) or config.optimizer not in _OPTIMIZER_NAMES:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}; expected one of {sorted(_OPTIMIZER_NAMES)}")
    if not isinstance(config.linear_solver, str) or config.linear_solver not in _LINEAR_SOLVER_NAMES:
        raise ValueError(
            f"Unknown linear_solver {config.linear_solver!r}; expected one of {sorted(_LINEAR_SOLVER_NAMES)}"
        )
    if not isinstance(config.kernel, str) or config.kernel not in _KERNEL_NAMES:
        raise ValueError(f"Unknown kernel {config.kernel!r}; expected one of {sorted(_KERNEL_NAMES)}")
    if not isinstance(config.damping, str) or config.damping not in _DAMPING_NAMES:
        raise ValueError(f"Unknown damping {config.damping!r}; expected one of {sorted(_DAMPING_NAMES)}")
    if config.jacobian_strategy not in {"auto", "analytic", "jacrev", "jacfwd", "finite_difference"}:
        raise ValueError(
            f"Unknown jacobian_strategy {config.jacobian_strategy!r}; expected auto, analytic, "
            "jacrev, jacfwd, or finite_difference"
        )

    if config.optimizer == "adam":
        unused: list[str] = []
        if config.linear_solver != "cholesky":
            unused.append("linear_solver")
        if config.jacobian_strategy != "auto":
            unused.append("jacobian_strategy")
        if config.damping != "adaptive":
            unused.append("damping")
        if unused:
            fields = ", ".join(unused)
            raise ValueError(f"optimizer='adam' does not use {fields}; leave these fields at their defaults")
    if config.optimizer == "gn" and config.damping != "adaptive":
        raise ValueError("optimizer='gn' does not use damping; leave this field at its default")
    if config.refine_disabled_items and config.optimizer != "lm_then_adam":
        raise ValueError(
            "refine_disabled_items is only used by optimizer='lm_then_adam'; leave it empty for other optimizers"
        )


def _make_linear_solver(name: str):
    """Return a fresh named-block linear solver."""
    table = {"cholesky": Cholesky}
    if name not in table:
        raise ValueError(f"Unknown linear_solver {name!r}; expected one of {sorted(table)}")
    return table[name]()


def _make_robust_kernel(name: str):
    """Return a fresh robust kernel, including the explicit L2 kernel."""
    table = {"l2": L2, "huber": Huber, "cauchy": Cauchy, "tukey": Tukey}
    if name not in table:
        raise ValueError(f"Unknown kernel {name!r}; expected one of {sorted(table)}")
    return table[name]()


def _broadcast_initial_configuration(
    model: Model,
    initial_q: torch.Tensor,
    targets: dict[str, torch.Tensor],
    q_rest: torch.Tensor | None,
) -> torch.Tensor:
    if not isinstance(initial_q, torch.Tensor) or not initial_q.is_floating_point():
        raise TypeError("initial_q must be a floating torch.Tensor")
    if initial_q.ndim < 1 or initial_q.shape[-1] != model.nq:
        raise ValueError(f"initial_q must end in model.nq={model.nq}, got {tuple(initial_q.shape)}")
    batch_shapes: list[tuple[int, ...]] = [tuple(initial_q.shape[:-1])]
    for frame_name, target in targets.items():
        if not isinstance(target, torch.Tensor) or not target.is_floating_point():
            raise TypeError(f"target for frame {frame_name!r} must be a floating tensor")
        if target.ndim < 1 or target.shape[-1] != 7:
            raise ValueError(f"target for frame {frame_name!r} must end in SE3 shape (7,), got {tuple(target.shape)}")
        batch_shapes.append(tuple(target.shape[:-1]))
    if q_rest is not None:
        if not isinstance(q_rest, torch.Tensor) or not q_rest.is_floating_point():
            raise TypeError("IKCostConfig.q_rest must be a floating torch.Tensor")
        if q_rest.ndim < 1 or q_rest.shape[-1] != model.nq:
            raise ValueError(f"IKCostConfig.q_rest must end in model.nq={model.nq}, got {tuple(q_rest.shape)}")
        batch_shapes.append(tuple(q_rest.shape[:-1]))
    try:
        batch_shape = torch.broadcast_shapes(*batch_shapes)
    except RuntimeError as exc:
        raise ValueError(
            f"initial_q, target, and q_rest leading batch shapes must broadcast; got {batch_shapes}"
        ) from exc
    return torch.broadcast_to(initial_q, (*batch_shape, model.nq)).clone()


def _state_iterations(state: Any) -> torch.Tensor:
    if hasattr(state, "iterations"):
        return state.iterations
    if hasattr(state, "step"):
        return state.step
    raise TypeError(f"solver state {type(state).__name__} has no iteration tensor")


def _public_diagnostics(
    states: tuple[Any, ...],
    exemplar: torch.Tensor,
) -> tuple[int | torch.Tensor, bool | torch.Tensor]:
    nonempty = tuple(state for state in states if state is not None)
    if not nonempty:
        batch_shape = exemplar.shape[:-1]
        iterations = torch.zeros(batch_shape, dtype=torch.int64, device=exemplar.device)
        converged = torch.zeros(batch_shape, dtype=torch.bool, device=exemplar.device)
    else:
        iterations = sum(
            (_state_iterations(state) for state in nonempty),
            torch.zeros_like(_state_iterations(nonempty[0])),
        )
        converged = nonempty[-1].converged
    if iterations.ndim == 0:
        return int(iterations), bool(converged)
    return iterations, converged


def _refinement_problem(problem: Problem, disabled_items: tuple[str, ...]) -> Problem:
    known = {item.name for item in problem.residuals}
    unknown = set(disabled_items) - known
    if unknown:
        raise ValueError(f"refine_disabled_items contains unknown residual names {sorted(unknown)}")
    return Problem(
        vars=problem.vars,
        residuals=tuple(
            replace(item, weight=0.0) if item.name in disabled_items else item for item in problem.residuals
        ),
        providers=problem.providers,
        parameters=problem.parameters,
        differentiable_parameters=tuple(problem.parameter_gradients),
    )


def solve_ik(  # noqa: PLR0915 - explicit preset assembly keeps task policy visible
    model: Model,
    targets: dict[str, torch.Tensor],
    *,
    initial_q: torch.Tensor | None = None,
    cost_cfg: IKCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
) -> IKResult:
    """Solve one or an arbitrary leading batch of frame-target IK problems.

    Targets map frame names to scalar-last SE(3) poses shaped ``(B..., 7)``.
    ``initial_q`` may be ``(nq,)`` or ``(B..., nq)`` and is broadcast with all
    target batches. Per-element convergence and iteration tensors are returned
    for batched calls; unbatched diagnostics remain Python scalars.
    """
    cost_cfg = cost_cfg if cost_cfg is not None else IKCostConfig()
    optimizer_cfg = optimizer_cfg if optimizer_cfg is not None else OptimizerConfig()
    _validate_optimizer_config(optimizer_cfg)
    if not isinstance(targets, dict):
        raise TypeError("targets must be a dict mapping frame names to SE3 tensors")
    if optimizer_cfg.optimizer in {"lbfgs", "lm_then_lbfgs"}:
        raise NotImplementedError(
            "Batched named-block L-BFGS is deferred because per-element line "
            "search/history reset semantics are not implemented. Use 'adam' "
            "or 'lm_then_adam'."
        )

    start = initial_q.clone().detach() if initial_q is not None else model.q_neutral.clone()
    active_q_rest = cost_cfg.q_rest if cost_cfg.rest_weight > 0.0 else None
    start = _broadcast_initial_configuration(model, start, targets, active_q_rest)
    manifold = RobotConfig(model)
    bounds = manifold.joint_bounds()
    # The task preset starts inside its declared joint box; several shipped
    # neutral configurations lie outside it (notably Panda joint 4).
    start = manifold.project(start, bounds)

    kernel = _make_robust_kernel(optimizer_cfg.kernel)
    residuals: list[ResidualItem] = []
    parameters: dict[str, torch.Tensor] = {}
    differentiable_parameters: list[str] = []
    for target_index, (frame_name, target) in enumerate(targets.items()):
        frame_id = model.frame_id(frame_name)
        item_name = f"pose_{frame_name}"
        target_name = f"target_pose_{target_index}"
        parameters[target_name] = target
        differentiable_parameters.append(target_name)
        residuals.append(
            ResidualItem(
                item_name,
                PoseResidual(
                    frame_id=frame_id,
                    target=target,
                    pos_weight=cost_cfg.pos_weight,
                    ori_weight=cost_cfg.ori_weight,
                    model=model,
                    name=item_name,
                    target_name=target_name,
                ),
                weight=cost_cfg.pose_weight,
                kernel=kernel,
            )
        )
    if cost_cfg.limit_weight > 0.0:
        residuals.append(
            ResidualItem(
                "limits",
                JointPositionLimit(model, name="limits"),
                weight=cost_cfg.limit_weight,
                kernel=kernel,
            )
        )
    q_rest = cost_cfg.q_rest if cost_cfg.q_rest is not None else model.q_neutral
    if cost_cfg.rest_weight > 0.0:
        rest_target_name = "target_rest"
        parameters[rest_target_name] = q_rest
        differentiable_parameters.append(rest_target_name)
        residuals.append(
            ResidualItem(
                "rest",
                RestResidual(
                    model,
                    q_rest,
                    name="rest",
                    target_name=rest_target_name,
                ),
                weight=cost_cfg.rest_weight,
                kernel=kernel,
            )
        )
    problem = Problem(
        vars=(
            VarSpec(
                "q",
                (model.nq,),
                manifold=manifold,
                bounds=bounds,
            ),
        ),
        residuals=tuple(residuals),
        providers=(RobotStateProvider(model),),
        parameters=parameters,
        differentiable_parameters=tuple(differentiable_parameters),
    )
    values = {"q": start}

    if optimizer_cfg.optimizer == "adam":
        values, state = run_first_order(
            values,
            problem,
            lambda params: torch.optim.Adam(params, lr=1e-2),
            max_iter=optimizer_cfg.max_iter,
            tolerance=optimizer_cfg.tol,
        )
        states: tuple[Any, ...] = (state,)
    else:
        jacobian_strategy = optimizer_cfg.jacobian_strategy
        common = {
            "gtol": optimizer_cfg.tol,
            "linear_solver": _make_linear_solver(optimizer_cfg.linear_solver),
            "kernel": kernel,
            "jacobian_strategy": jacobian_strategy,
        }
        if optimizer_cfg.optimizer == "lm":
            solver = LevenbergMarquardt(
                max_iter=optimizer_cfg.max_iter,
                **common,
                fixed_damping=optimizer_cfg.damping == "constant",
            )
            values, state = solver.run(values, problem)
            states = (state,)
        elif optimizer_cfg.optimizer == "gn":
            solver = GaussNewton(max_iter=optimizer_cfg.max_iter, **common)
            values, state = solver.run(values, problem)
            states = (state,)
        elif optimizer_cfg.optimizer == "lm_then_adam":
            coarse_iters = optimizer_cfg.max_iter // 2
            refine_iters = optimizer_cfg.max_iter - coarse_iters
            values, coarse_state = LevenbergMarquardt(
                max_iter=coarse_iters,
                **common,
                fixed_damping=optimizer_cfg.damping == "constant",
            ).run(values, problem)
            refinement = _refinement_problem(problem, optimizer_cfg.refine_disabled_items)
            values, refine_state = run_first_order(
                values,
                refinement,
                lambda params: torch.optim.Adam(params, lr=1e-2),
                max_iter=refine_iters,
                tolerance=optimizer_cfg.tol,
            )
            states = (coarse_state, refine_state)
        else:  # validated Literal plus runtime protection for untyped callers
            raise ValueError(f"Unknown optimizer {optimizer_cfg.optimizer!r}")

    iterations, converged = _public_diagnostics(states, values["q"])
    return IKResult(
        q=values["q"],
        residual=problem.residual(values),
        iters=iterations,
        converged=converged,
        model=model,
    )
