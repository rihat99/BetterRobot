"""Contract tests for extension Protocols and the residual abstract base class.

See ``docs/conventions/extension.md`` for the seam inventory and
``docs/concepts/residuals_costs_and_solvers.md`` for the concrete protocols.
"""

from __future__ import annotations

import inspect

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
from better_robot.io import ModelBuilder, build_model
from better_robot.optim import (
    BandedCholesky,
    Cauchy,
    Cholesky,
    GemanMcClure,
    Huber,
    InformativeLinearSolver,
    L2,
    LinearSolver,
    LU,
    RobotVariable,
    RobustKernel,
    Tukey,
)
from better_robot.residuals.base import Residual
from better_robot.residuals.pose import OrientationResidual, PoseResidual, PositionResidual
from better_robot.viewer.render_modes.base import RenderMode
from better_robot.viewer.render_modes.skeleton import SkeletonMode


def _is_runtime_checkable(proto: type) -> bool:
    return getattr(proto, "_is_runtime_protocol", False)


@pytest.mark.parametrize(
    "proto",
    [LinearSolver, InformativeLinearSolver, RobustKernel, JointModel, RenderMode],
)
def test_protocol_is_runtime_checkable(proto: type) -> None:
    """Every extension-seam Protocol must support ``isinstance``."""
    assert _is_runtime_checkable(proto), (
        f"{proto.__name__} is a Protocol but not runtime_checkable; "
        f"docs/conventions/extension.md requires @runtime_checkable."
    )


# ── Residuals ─────────────────────────────────────────────────────────────────


def _dummy_pose() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)


@pytest.fixture(scope="module")
def robot_variable() -> RobotVariable:
    identity = _dummy_pose()
    builder = ModelBuilder("protocol_residual")
    builder.add_body("base", mass=1.0)
    builder.add_frame("tip", parent_body="base", placement=identity)
    model = build_model(builder.finalize(), dtype=torch.float64)
    return RobotVariable(model, name="q")


def test_residual_is_an_abstract_base_class() -> None:
    assert inspect.isabstract(Residual)


@pytest.mark.parametrize("cls", [PoseResidual, PositionResidual, OrientationResidual])
def test_residual_instances_satisfy_abc(cls: type[Residual], robot_variable: RobotVariable) -> None:
    item = cls(robot_variable, frame="tip", target=_dummy_pose())
    assert isinstance(item, Residual)


# ── Linear solvers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [Cholesky, LU, BandedCholesky])
def test_linear_solver_instances_satisfy_protocol(cls: type) -> None:
    assert isinstance(cls(), LinearSolver), cls.__name__
    assert isinstance(cls(), InformativeLinearSolver), cls.__name__


# ── Robust kernels ────────────────────────────────────────────────────────────


def test_robust_kernel_instances_satisfy_protocol() -> None:
    assert isinstance(L2(), RobustKernel)
    assert isinstance(Huber(delta=1.0), RobustKernel)
    assert isinstance(Cauchy(c=1.0), RobustKernel)
    assert isinstance(Tukey(c=1.0), RobustKernel)
    assert isinstance(GemanMcClure(c=1.0), RobustKernel)


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
