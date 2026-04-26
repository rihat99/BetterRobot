"""Contract tests: every extension-seam ``Protocol`` is ``@runtime_checkable``
and every built-in implementation passes the corresponding ``isinstance`` check.

See ``docs/conventions/extension.md`` for the seam inventory and
``docs/concepts/residuals_and_costs.md §5`` for the concrete Protocols.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.data_model.joint_models.base import JointModel
from better_robot.data_model.joint_models import (
    JointFixed,
    JointFreeFlyer,
    JointPlanar,
    JointPX,
    JointRX,
    JointRZ,
    JointSpherical,
    JointUniverse,
)
from better_robot.optim.kernels import Cauchy, Huber, L2
from better_robot.optim.kernels.base import RobustKernel
from better_robot.optim.optimizers.adam import Adam
from better_robot.optim.optimizers.base import Optimizer
from better_robot.optim.optimizers.gauss_newton import GaussNewton
from better_robot.optim.optimizers.lbfgs import LBFGS
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.optim.solvers import CG, LSTSQ, Cholesky, SparseCholesky
from better_robot.optim.solvers.base import LinearSolver
from better_robot.optim.strategies import Adaptive, Constant, TrustRegion
from better_robot.optim.strategies.base import DampingStrategy
from better_robot.residuals.base import Residual
from better_robot.residuals.limits import JointPositionLimit
from better_robot.residuals.pose import OrientationResidual, PoseResidual, PositionResidual
from better_robot.viewer.render_modes.base import RenderMode
from better_robot.viewer.render_modes.skeleton import SkeletonMode


def _is_runtime_checkable(proto: type) -> bool:
    return getattr(proto, "_is_runtime_protocol", False)


@pytest.mark.parametrize(
    "proto",
    [Residual, Optimizer, LinearSolver, DampingStrategy, RobustKernel, JointModel, RenderMode],
)
def test_protocol_is_runtime_checkable(proto: type) -> None:
    """Every extension-seam Protocol must support ``isinstance``."""
    assert _is_runtime_checkable(proto), (
        f"{proto.__name__} is a Protocol but not runtime_checkable; "
        f"docs/conventions/extension.md requires @runtime_checkable."
    )


# ── Residuals ─────────────────────────────────────────────────────────────────

def _dummy_pose() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_residual_instances_satisfy_protocol() -> None:
    p = PoseResidual(frame_id=0, target=_dummy_pose())
    assert isinstance(p, Residual)
    pos = PositionResidual(frame_id=0, target=_dummy_pose())
    assert isinstance(pos, Residual)
    ori = OrientationResidual(frame_id=0, target=_dummy_pose())
    assert isinstance(ori, Residual)


# ── Optimizers ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [LevenbergMarquardt, GaussNewton, Adam, LBFGS])
def test_optimizer_instances_satisfy_protocol(cls: type) -> None:
    assert isinstance(cls(), Optimizer), cls.__name__


# ── Linear solvers ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [Cholesky, LSTSQ, CG, SparseCholesky])
def test_linear_solver_instances_satisfy_protocol(cls: type) -> None:
    assert isinstance(cls(), LinearSolver), cls.__name__


# ── Damping strategies ───────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [Adaptive, Constant, TrustRegion])
def test_damping_strategy_instances_satisfy_protocol(cls: type) -> None:
    assert isinstance(cls(), DampingStrategy), cls.__name__


# ── Robust kernels ────────────────────────────────────────────────────────────

def test_robust_kernel_instances_satisfy_protocol() -> None:
    assert isinstance(L2(), RobustKernel)
    assert isinstance(Huber(delta=1.0), RobustKernel)
    assert isinstance(Cauchy(c=1.0), RobustKernel)


# ── Joint models ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "jm",
    [
        JointUniverse(),
        JointFixed(),
        JointRX(),
        JointRZ(),
        JointPX(),
        JointSpherical(),
        JointFreeFlyer(),
        JointPlanar(),
    ],
)
def test_joint_model_instances_satisfy_protocol(jm) -> None:
    assert isinstance(jm, JointModel), type(jm).__name__


# ── Render modes ──────────────────────────────────────────────────────────────

def test_render_mode_instance_satisfies_protocol() -> None:
    assert isinstance(SkeletonMode(), RenderMode)
