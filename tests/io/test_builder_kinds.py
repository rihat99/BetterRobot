"""Every joint kind, built in or written outside the package, enters ``ModelBuilder.add_joint`` and builds a model that
runs FK; a kind that is not a ``JointModel`` instance fails at the call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, final

import pytest
import torch

from better_robot.exceptions import ArgumentError
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics
from better_robot.lie import se3
from better_robot.model.joints import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointModel,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
)
from better_robot.model.manifolds import EUCLIDEAN, JointManifold


def _assert_fk_runs(model) -> None:
    state = forward_kinematics(model, model.q_neutral)
    assert state.joint_poses.shape == (model.njoints, 7)
    assert torch.isfinite(state.joint_poses).all()
    assert model.lower_position_limit.shape == (model.nq,)
    assert model.upper_position_limit.shape == (model.nq,)
    assert model.velocity_limit.shape == (model.nv,)
    assert model.effort_limit.shape == (model.nv,)


@final
@dataclass(frozen=True)
class _Slider(JointModel):
    """An out-of-tree kind: a translation along ``-z`` (a kind subclasses ``JointModel``)."""

    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[JointManifold] = EUCLIDEAN
    axis: ClassVar[None] = None

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return se3.from_translation(torch.cat((q.new_zeros(*q.shape[:-1], 2), -q), dim=-1))

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return q.new_tensor([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]).expand(*q.shape[:-1], 6).unsqueeze(-1)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)

    def random(self, generator: torch.Generator | None, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(1, generator=generator)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(JointFixed, id="fixed"),
        pytest.param(lambda: JointRevolute(axis=(0.6, 0.8, 0.0)), id="revolute"),
        pytest.param(lambda: JointRevoluteUnbounded(axis=(0.0, 0.0, 1.0)), id="continuous"),
        pytest.param(lambda: JointPrismatic(axis=(0.6, 0.8, 0.0)), id="prismatic"),
        pytest.param(JointSpherical, id="spherical"),
        pytest.param(JointFreeFlyer, id="free_flyer"),
        pytest.param(JointPlanar, id="planar"),
        pytest.param(JointTranslation, id="translation"),
        pytest.param(lambda: JointHelical(axis=(0.0, 0.0, 1.0), pitch=0.25), id="helical"),
        pytest.param(_Slider, id="custom"),
    ],
)
def test_joint_model_instances_round_trip_and_run_fk(factory: Callable[[], JointModel]):
    joint_model = factory()
    builder = ModelBuilder(type(joint_model).__name__)
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_joint("joint", kind=joint_model, parent=base, child=tip)

    model = build_model(builder.finalize())

    assert model.joints[2] is joint_model
    _assert_fk_runs(model)


def test_add_joint_takes_a_kind_instance_only():
    builder = ModelBuilder("string_kind")
    base, tip = builder.add_body("base"), builder.add_body("tip")
    with pytest.raises(ArgumentError, match="JointModel"):
        builder.add_joint("joint", kind="revolute", parent=base, child=tip)


def test_custom_kind_runs_through_fk():
    builder = ModelBuilder("slider")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_joint("joint", kind=_Slider(), parent=base, child=tip)
    model = build_model(builder.finalize())

    state = forward_kinematics(model, torch.tensor([0.4]))

    assert state.joint_poses[2, 2].item() == pytest.approx(-0.4)
