"""``solve_ik`` — whole-body inverse kinematics facade (≤120 LOC target).

Single code path. Floating-base robots are those whose ``joint_models[1]``
is ``JointFreeFlyer`` — the solver does not need to know. All seven
quaternion-xyz values of the free-flyer live inside ``q``.

See ``docs/design/08_TASKS.md §1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from ..costs.stack import CostStack
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics
from ..kinematics.jacobian_strategy import JacobianStrategy
from ..optim.optimizers.adam import Adam
from ..optim.optimizers.gauss_newton import GaussNewton
from ..optim.optimizers.lbfgs import LBFGS
from ..optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from ..optim.optimizers.lm_then_lbfgs import LMThenLBFGS
from ..optim.problem import LeastSquaresProblem
from ..residuals.base import ResidualState
from ..residuals.limits import JointPositionLimit
from ..residuals.pose import PoseResidual
from ..residuals.regularization import RestResidual

if TYPE_CHECKING:
    from ..collision.robot_collision import RobotCollision
    from ..data_model.data import Data


@dataclass
class IKCostConfig:
    """Weights for the built-in IK cost stack."""

    pos_weight: float = 1.0
    ori_weight: float = 1.0
    pose_weight: float = 1.0
    limit_weight: float = 0.1
    rest_weight: float = 0.01
    collision_margin: float = 0.02
    collision_weight: float = 1.0
    q_rest: torch.Tensor | None = None


@dataclass
class OptimizerConfig:
    """Optimizer selection + hyperparameters.

    The ``lm_then_lbfgs`` option (from docs/design/08_TASKS.md) runs Levenberg-
    Marquardt for a coarse solve, then L-BFGS for final refinement.
    ``refine_disabled_items`` names cost-stack entries to drop in stage 2
    — the typical use is ``("collision",)`` once LM has the configuration
    inside free space.
    """

    optimizer: Literal["lm", "gn", "adam", "lbfgs", "lm_then_lbfgs"] = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO
    linear_solver: Literal["cholesky", "lstsq", "cg"] = "cholesky"
    kernel: Literal["l2", "huber", "cauchy", "tukey"] = "l2"
    damping: Literal["constant", "adaptive", "trust_region"] = "adaptive"
    tol: float = 1e-6
    refine_disabled_items: tuple[str, ...] = ()


@dataclass
class IKResult:
    """Return type of ``solve_ik``."""

    q: torch.Tensor
    residual: torch.Tensor
    iters: int
    converged: bool
    model: Model

    def fk(self) -> "Data":
        """Return the FK ``Data`` at the solution.

        See docs/design/08_TASKS.md §1.
        """
        return forward_kinematics(self.model, self.q, compute_frames=True)

    def frame_pose(self, name: str) -> torch.Tensor:
        """Look up a frame pose by name on the FK result.

        See docs/design/08_TASKS.md §1.
        """
        data = self.fk()
        frame_id = self.model.frame_id(name)
        return data.frame_pose_world[..., frame_id, :]

    def q_only(self) -> torch.Tensor:
        return self.q


def _make_linear_solver(name: str):
    """Return a fresh ``LinearSolver`` instance for the named string."""
    from ..optim.solvers.cg import CG
    from ..optim.solvers.cholesky import Cholesky
    from ..optim.solvers.lstsq import LSTSQ

    table = {"cholesky": Cholesky, "lstsq": LSTSQ, "cg": CG}
    if name not in table:
        raise ValueError(
            f"Unknown linear_solver {name!r}; expected one of {sorted(table)}"
        )
    return table[name]()


def _make_robust_kernel(name: str):
    """Return a fresh ``RobustKernel`` instance for the named string.

    ``"l2"`` returns ``None`` so the optimiser can take the fast path
    (no per-row reweighting work).
    """
    if name == "l2":
        return None
    from ..optim.kernels.cauchy import Cauchy
    from ..optim.kernels.huber import Huber
    from ..optim.kernels.tukey import Tukey

    table = {"huber": Huber, "cauchy": Cauchy, "tukey": Tukey}
    if name not in table:
        raise ValueError(
            f"Unknown kernel {name!r}; expected 'l2' or one of {sorted(table)}"
        )
    return table[name]()


def _make_damping_strategy(name: str):
    """Return a fresh ``DampingStrategy`` instance for the named string."""
    from ..optim.strategies.adaptive import Adaptive
    from ..optim.strategies.constant import Constant
    from ..optim.strategies.trust_region import TrustRegion

    table = {"adaptive": Adaptive, "constant": Constant, "trust_region": TrustRegion}
    if name not in table:
        raise ValueError(
            f"Unknown damping {name!r}; expected one of {sorted(table)}"
        )
    return table[name]()


def solve_ik(
    model: Model,
    targets: dict[str, torch.Tensor],
    *,
    initial_q: torch.Tensor | None = None,
    cost_cfg: IKCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    robot_collision: "RobotCollision | None" = None,
) -> IKResult:
    """Whole-body inverse kinematics for one or more frame targets.

    Single code path — floating-base robots are handled transparently by
    having ``joint_models[1] = JointFreeFlyer``.

    Parameters
    ----------
    model : Model
    targets : dict mapping frame name → target SE3 pose ``(7,)``
    initial_q : optional starting configuration ``(nq,)``
    cost_cfg : IK cost weights
    optimizer_cfg : optimizer settings
    robot_collision : optional collision model (unused in this version)

    Returns
    -------
    IKResult

    See docs/design/08_TASKS.md §1.
    """
    if cost_cfg is None:
        cost_cfg = IKCostConfig()
    if optimizer_cfg is None:
        optimizer_cfg = OptimizerConfig()

    # ── initial configuration ──────────────────────────────────────────
    if initial_q is not None:
        x0 = initial_q.clone().detach().float()
    else:
        x0 = model.q_neutral.clone().float()

    # ── build cost stack ───────────────────────────────────────────────
    stack = CostStack()

    pw = cost_cfg.pos_weight * cost_cfg.pose_weight
    ow = cost_cfg.ori_weight * cost_cfg.pose_weight

    for name, target_pose in targets.items():
        frame_id = model.frame_id(name)
        stack.add(
            f"pose_{name}",
            PoseResidual(frame_id=frame_id, target=target_pose, pos_weight=pw, ori_weight=ow),
            weight=1.0,
        )

    if cost_cfg.limit_weight > 0.0:
        stack.add("limits", JointPositionLimit(model), weight=cost_cfg.limit_weight)

    q_rest = cost_cfg.q_rest if cost_cfg.q_rest is not None else model.q_neutral
    if cost_cfg.rest_weight > 0.0:
        stack.add("rest", RestResidual(model, q_rest), weight=cost_cfg.rest_weight)

    # ── state factory ──────────────────────────────────────────────────
    def _state_factory(x: torch.Tensor) -> ResidualState:
        data = forward_kinematics(model, x, compute_frames=True)
        return ResidualState(model=model, data=data, variables=x)

    # ── build problem ──────────────────────────────────────────────────
    problem = LeastSquaresProblem(
        cost_stack=stack,
        state_factory=_state_factory,
        x0=x0,
        lower=model.lower_pos_limit.float(),
        upper=model.upper_pos_limit.float(),
        jacobian_strategy=optimizer_cfg.jacobian_strategy,
        nv=model.nv,
        retract=lambda q, dv: model.integrate(q, dv),
    )

    # ── run optimizer ──────────────────────────────────────────────────
    opt_name = optimizer_cfg.optimizer
    if opt_name == "lm":
        opt = LevenbergMarquardt(tol=optimizer_cfg.tol)
    elif opt_name == "gn":
        opt = GaussNewton(tol=optimizer_cfg.tol)
    elif opt_name == "adam":
        opt = Adam(tol=optimizer_cfg.tol)
    elif opt_name == "lbfgs":
        opt = LBFGS(tol=optimizer_cfg.tol)
    elif opt_name == "lm_then_lbfgs":
        opt = LMThenLBFGS(
            stage1_max_iter=optimizer_cfg.max_iter // 2,
            stage2_max_iter=optimizer_cfg.max_iter - optimizer_cfg.max_iter // 2,
            stage2_disabled_items=optimizer_cfg.refine_disabled_items,
            tol=optimizer_cfg.tol,
        )
    else:
        raise ValueError(f"Unknown optimizer {opt_name!r}")

    linear_solver = _make_linear_solver(optimizer_cfg.linear_solver)
    kernel = _make_robust_kernel(optimizer_cfg.kernel)
    strategy = _make_damping_strategy(optimizer_cfg.damping)

    result = opt.minimize(
        problem,
        max_iter=optimizer_cfg.max_iter,
        linear_solver=linear_solver,
        kernel=kernel,
        strategy=strategy,
    )

    return IKResult(
        q=result.x,
        residual=result.residual,
        iters=result.iters,
        converged=result.converged,
        model=model,
    )
