"""Contract tests: every extension-seam ``Protocol`` is ``@runtime_checkable``
and every built-in implementation passes the corresponding ``isinstance`` check.

See ``docs/conventions/extension.md`` for the seam inventory and
``docs/concepts/residuals_costs_and_solvers.md`` for the concrete protocols.
"""

from __future__ import annotations

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
from better_robot.optim import Cauchy, Cholesky, Huber, L2, LinearSolver, RobustKernel
from better_robot.residuals.base import Residual
from better_robot.residuals.pose import OrientationResidual, PoseResidual, PositionResidual
from better_robot.viewer.render_modes.base import RenderMode
from better_robot.viewer.render_modes.skeleton import SkeletonMode


def _is_runtime_checkable(proto: type) -> bool:
    return getattr(proto, "_is_runtime_protocol", False)


@pytest.mark.parametrize(
    "proto",
    [Residual, LinearSolver, RobustKernel, JointModel, RenderMode],
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


# ── Linear solvers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [Cholesky])
def test_linear_solver_instances_satisfy_protocol(cls: type) -> None:
    assert isinstance(cls(), LinearSolver), cls.__name__


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
