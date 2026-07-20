"""Whole-body inverse kinematics as an object-referenced optimization preset.

The task facade owns no solver loop. It assembles one ``q`` variable on the
robot-configuration manifold, installs the built-in kinematic residuals, and
runs a public optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from .._validation import check_tensor
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics
from ..optim import (
    GaussNewton,
    JacobianStrategy,
    LevenbergMarquardt,
    Problem,
    Residual,
    RobotVariable,
    TorchOptimizer,
    Variable,
)
from ..optim import Cauchy, Cholesky, Huber, L2, Tukey
from ..residuals.limits import JointPositionLimit
from ..residuals.pose import PoseResidual
from ..residuals.regularization import RestResidual
from .utils import _public_diagnostics

if TYPE_CHECKING:
    from ..data_model.data import Data


_OptimizerName = Literal["lm", "gn", "adam", "lm_then_adam"]

_OPTIMIZER_NAMES = frozenset({"lm", "gn", "adam", "lm_then_adam"})
_LINEAR_SOLVERS = {"cholesky": Cholesky}
_KERNELS = {"l2": L2, "huber": Huber, "cauchy": Cauchy, "tukey": Tukey}
_LINEAR_SOLVER_NAMES = frozenset(_LINEAR_SOLVERS)
_KERNEL_NAMES = frozenset(_KERNELS)
_DAMPING_NAMES = frozenset({"constant", "adaptive"})
_JACOBIAN_STRATEGIES = frozenset({"auto", "analytic", "jacrev", "jacfwd", "finite_difference"})
_CONFIG_CHOICES = (
    ("optimizer", _OPTIMIZER_NAMES),
    ("linear_solver", _LINEAR_SOLVER_NAMES),
    ("kernel", _KERNEL_NAMES),
    ("damping", _DAMPING_NAMES),
    ("jacobian_strategy", _JACOBIAN_STRATEGIES),
)


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
    """Optimizer selection and hyperparameters.

    ``lm_then_adam`` runs LM followed by the ``torch.optim`` adapter.
    Linear-solver and Jacobian settings apply to LM/GN phases; damping applies
    only to configurations with an LM phase. Presets ignore fields that do
    not apply to their selected optimizer.
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


def _validate_choices(config: OptimizerConfig) -> None:
    for name, allowed in _CONFIG_CHOICES:
        value = getattr(config, name)
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"Unknown {name} {value!r}; expected one of {sorted(allowed)}")


def _make_linear_solver(name: str):
    """Return a fresh linear solver."""
    if name not in _LINEAR_SOLVERS:
        raise ValueError(f"Unknown linear_solver {name!r}; expected one of {sorted(_LINEAR_SOLVERS)}")
    return _LINEAR_SOLVERS[name]()


def _make_robust_kernel(name: str):
    """Return a fresh robust kernel, including the explicit L2 kernel."""
    if name not in _KERNELS:
        raise ValueError(f"Unknown kernel {name!r}; expected one of {sorted(_KERNELS)}")
    return _KERNELS[name]()


def _broadcast_initial_configuration(
    model: Model,
    initial_q: torch.Tensor,
    targets: dict[str, torch.Tensor],
    q_rest: torch.Tensor | None,
) -> torch.Tensor:
    tensors = [check_tensor("initial_q", initial_q, shape=(model.nq,), floating=True)]
    tensors.extend(
        check_tensor(f"target for frame {name!r}", target, shape=(7,), floating=True)
        for name, target in targets.items()
    )
    if q_rest is not None:
        tensors.append(check_tensor("IKCostConfig.q_rest", q_rest, shape=(model.nq,), floating=True))

    batch_shape = _ik_batch_shape(tensors)
    return torch.broadcast_to(initial_q, (*batch_shape, model.nq)).clone()


def _ik_batch_shape(tensors: list[torch.Tensor]) -> torch.Size:
    batch_shapes = [tuple(tensor.shape[:-1]) for tensor in tensors]
    try:
        return torch.broadcast_shapes(*batch_shapes)
    except RuntimeError as exc:
        raise ValueError(
            f"initial_q, target, and q_rest leading batch shapes must broadcast; got {batch_shapes}"
        ) from exc


def _refinement_residuals(problem: Problem, disabled_items: tuple[str, ...]) -> tuple[Residual, ...]:
    known = {residual.name: residual for residual in problem.residuals}
    unknown = set(disabled_items) - set(known)
    if unknown:
        raise ValueError(f"refine_disabled_items contains unknown residual names {sorted(unknown)}")
    return tuple(known[name] for name in disabled_items)


def solve_ik(  # noqa: PLR0912, PLR0915 - explicit preset assembly keeps task policy visible
    model: Model,
    targets: dict[str, torch.Tensor],
    *,
    initial_q: torch.Tensor | None = None,
    cost_cfg: IKCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    differentiable: bool = False,
) -> IKResult:
    """Solve one or an arbitrary leading batch of frame-target IK problems.

    Targets map frame names to scalar-last SE(3) poses shaped ``(B..., 7)``.
    ``initial_q`` may be ``(nq,)`` or ``(B..., nq)`` and is broadcast with all
    target batches. Per-element convergence and iteration tensors are returned
    for batched calls; unbatched diagnostics remain Python scalars.

    ``differentiable=True`` requires the LM optimizer and attaches a first-order
    implicit backward from ``result.q`` to graph-carrying targets. The backward
    assumes a converged, locally smooth optimum with a stable active set and
    raises the implicit solver's eligibility error when those conditions fail.
    """
    if not isinstance(differentiable, bool):
        raise TypeError("differentiable must be a bool")
    cost_cfg = cost_cfg if cost_cfg is not None else IKCostConfig()
    optimizer_cfg = optimizer_cfg if optimizer_cfg is not None else OptimizerConfig()
    _validate_choices(optimizer_cfg)
    if not isinstance(targets, dict):
        raise TypeError("targets must be a dict mapping frame names to SE3 tensors")
    if differentiable and optimizer_cfg.optimizer != "lm":
        raise ValueError("differentiable=True requires optimizer_cfg.optimizer='lm'")
    start = initial_q.clone().detach() if initial_q is not None else model.q_neutral.clone()
    active_q_rest = cost_cfg.q_rest if cost_cfg.rest_weight > 0.0 else None
    start = _broadcast_initial_configuration(model, start, targets, active_q_rest)
    q_variable = RobotVariable(model, start, name="q", bounds=True)
    # The task preset starts inside its declared joint box; several shipped
    # neutral configurations lie outside it (notably Panda joint 4). A zero
    # retraction also normalizes any unit-coordinate representatives.
    q_variable.tensor = q_variable.retract(
        q_variable.tensor.new_zeros(*q_variable.batch_shape, q_variable.free_dim),
    )

    kernel = _make_robust_kernel(optimizer_cfg.kernel)
    residuals: list[Residual] = []
    for target_index, (frame_name, target) in enumerate(targets.items()):
        frame_id = model.frame_id(frame_name)
        item_name = f"pose_{frame_name}"
        target_name = f"target_pose_{target_index}"
        target_variable = Variable(
            target,
            name=target_name,
            trainable=False,
            batch_ndim=target.ndim - 1,
        )
        residuals.append(
            PoseResidual(
                q_variable,
                frame_id=frame_id,
                target=target_variable,
                pos_weight=cost_cfg.pos_weight,
                ori_weight=cost_cfg.ori_weight,
                weight=cost_cfg.pose_weight,
                kernel=kernel,
                name=item_name,
            )
        )
    if cost_cfg.limit_weight > 0.0:
        residuals.append(
            JointPositionLimit(
                q_variable,
                weight=cost_cfg.limit_weight,
                kernel=kernel,
                name="limits",
            )
        )
    q_rest = cost_cfg.q_rest if cost_cfg.q_rest is not None else model.q_neutral
    if cost_cfg.rest_weight > 0.0:
        rest_variable = Variable(
            q_rest,
            name="target_rest",
            trainable=False,
            batch_ndim=q_rest.ndim - 1,
        )
        residuals.append(
            RestResidual(
                q_variable,
                rest_variable,
                weight=cost_cfg.rest_weight,
                kernel=kernel,
                name="rest",
            )
        )
    problem = Problem(residuals)

    if optimizer_cfg.optimizer == "adam":
        info = TorchOptimizer(
            problem,
            torch.optim.Adam,
            lr=1e-2,
            max_iterations=optimizer_cfg.max_iter,
            tolerance=optimizer_cfg.tol,
        ).optimize()
        infos = (info,)
    else:
        common = {
            "tolerance": optimizer_cfg.tol,
            "solver": _make_linear_solver(optimizer_cfg.linear_solver),
            "jacobian_strategy": optimizer_cfg.jacobian_strategy,
        }
        if optimizer_cfg.optimizer == "lm":
            solver = LevenbergMarquardt(
                problem,
                max_iterations=optimizer_cfg.max_iter,
                **common,
                fixed_damping=optimizer_cfg.damping == "constant",
            )
            info = solver.optimize(differentiate="implicit" if differentiable else None)
            infos = (info,)
        elif optimizer_cfg.optimizer == "gn":
            info = GaussNewton(
                problem,
                max_iterations=optimizer_cfg.max_iter,
                **common,
            ).optimize()
            infos = (info,)
        else:  # only lm_then_adam remains after boundary validation
            coarse_iters = optimizer_cfg.max_iter // 2
            refine_iters = optimizer_cfg.max_iter - coarse_iters
            disabled = _refinement_residuals(problem, optimizer_cfg.refine_disabled_items)
            coarse_info = LevenbergMarquardt(
                problem,
                max_iterations=coarse_iters,
                **common,
                fixed_damping=optimizer_cfg.damping == "constant",
            ).optimize()
            original_weights = tuple(residual.weight for residual in disabled)
            try:
                for residual in disabled:
                    residual.weight = 0.0
                refine_info = TorchOptimizer(
                    problem,
                    torch.optim.Adam,
                    lr=1e-2,
                    max_iterations=refine_iters,
                    tolerance=optimizer_cfg.tol,
                ).optimize()
            finally:
                for residual, weight in zip(disabled, original_weights, strict=True):
                    residual.weight = weight
            infos = (coarse_info, refine_info)
    iterations, converged, _ = _public_diagnostics(infos)
    return IKResult(
        q=q_variable.tensor,
        residual=problem.error(),
        iters=iterations,
        converged=converged,
        model=model,
    )
