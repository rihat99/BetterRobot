"""Every programmatic joint-builder path builds a model that runs FK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, final

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics_state
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
from better_robot.model.manifolds import EUCLIDEAN, Manifold


def _assert_fk_runs(model) -> None:
    state = forward_kinematics_state(model, model.q_neutral)
    assert state.joint_pose.shape == (model.njoints, 7)
    assert torch.isfinite(state.joint_pose).all()
    assert model.lower_position_limit.shape == (model.nq,)
    assert model.upper_position_limit.shape == (model.nq,)
    assert model.velocity_limit.shape == (model.nv,)
    assert model.effort_limit.shape == (model.nv,)


def _axis(*xyz: float) -> tuple[float, float, float]:
    """The axis as the builder stores it: a float32 tensor read back as Python floats."""
    x, y, z = torch.tensor(xyz).tolist()
    return (x, y, z)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        pytest.param(
            "add_revolute",
            {"axis": torch.tensor([0.7071068, 0.7071068, 0.0])},
            JointRevolute(axis=_axis(0.7071068, 0.7071068, 0.0)),
            id="revolute",
        ),
        pytest.param(
            "add_revolute",
            {"axis": torch.tensor([0.0, 0.0, 1.0]), "unbounded": True},
            JointRevoluteUnbounded(axis=(0.0, 0.0, 1.0)),
            id="continuous",
        ),
        pytest.param("add_revolute_x", {}, JointRevolute(axis=(1.0, 0.0, 0.0)), id="revolute_x"),
        pytest.param("add_revolute_y", {}, JointRevolute(axis=(0.0, 1.0, 0.0)), id="revolute_y"),
        pytest.param("add_revolute_z", {}, JointRevolute(axis=(0.0, 0.0, 1.0)), id="revolute_z"),
        pytest.param(
            "add_prismatic",
            {"axis": torch.tensor([0.7071068, 0.7071068, 0.0])},
            JointPrismatic(axis=_axis(0.7071068, 0.7071068, 0.0)),
            id="prismatic",
        ),
        pytest.param("add_prismatic_x", {}, JointPrismatic(axis=(1.0, 0.0, 0.0)), id="prismatic_x"),
        pytest.param("add_prismatic_y", {}, JointPrismatic(axis=(0.0, 1.0, 0.0)), id="prismatic_y"),
        pytest.param("add_prismatic_z", {}, JointPrismatic(axis=(0.0, 0.0, 1.0)), id="prismatic_z"),
        pytest.param("add_spherical", {}, JointSpherical(), id="spherical"),
        pytest.param("add_planar", {}, JointPlanar(), id="planar"),
        pytest.param(
            "add_helical",
            {"axis": torch.tensor([0.0, 0.0, 1.0]), "pitch": 0.25},
            JointHelical(axis=(0.0, 0.0, 1.0), pitch=0.25),
            id="helical",
        ),
        pytest.param("add_fixed", {}, JointFixed(), id="fixed"),
    ],
)
def test_named_joint_methods_build_and_run_fk(method, kwargs, expected: JointModel):
    builder = ModelBuilder(method)
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    getattr(builder, method)("joint", parent=base, child=tip, **kwargs)

    model = build_model(builder.finalize())

    assert model.joints[2] == expected
    _assert_fk_runs(model)


def test_add_free_flyer_root_builds_and_runs_fk():
    builder = ModelBuilder("free_flyer")
    base = builder.add_body("base")
    builder.add_free_flyer_root(child=base)

    model = build_model(builder.finalize())

    assert model.joints[1] == JointFreeFlyer()
    _assert_fk_runs(model)


@final
@dataclass(frozen=True)
class _Slider(JointModel):
    """An out-of-tree kind: a translation along ``-z`` (record 0025: a kind subclasses ``JointModel``)."""

    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = EUCLIDEAN
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
        pytest.param(lambda: JointRevolute(axis=(1.0, 1.0, 0.0)), id="revolute"),
        pytest.param(lambda: JointRevoluteUnbounded(axis=(0.0, 0.0, 1.0)), id="continuous"),
        pytest.param(lambda: JointPrismatic(axis=(1.0, 1.0, 0.0)), id="prismatic"),
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


def test_custom_kind_runs_through_fk():
    builder = ModelBuilder("slider")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_joint("joint", kind=_Slider(), parent=base, child=tip)
    model = build_model(builder.finalize())

    state = forward_kinematics_state(model, torch.tensor([0.4]))

    assert state.joint_pose[2, 2].item() == pytest.approx(-0.4)


def test_named_helical_preserves_pitch_in_fk():
    builder = ModelBuilder("helical_pitch")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_helical(
        "joint",
        parent=base,
        child=tip,
        axis=torch.tensor([0.0, 0.0, 1.0]),
        pitch=0.25,
    )
    model = build_model(builder.finalize())

    q = torch.tensor([0.4])
    state = forward_kinematics_state(model, q)

    assert model.joints[2] == JointHelical(axis=(0.0, 0.0, 1.0), pitch=0.25)
    assert state.joint_pose[2, 2].item() == pytest.approx(0.1)
