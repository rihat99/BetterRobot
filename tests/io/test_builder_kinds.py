"""Every programmatic joint-builder path builds a model that runs FK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from better_robot.data_model.joint_models import (
    JointComposite,
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointMimic,
    JointPlanar,
    JointPrismaticUnaligned,
    JointPX,
    JointPY,
    JointPZ,
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
    JointRX,
    JointRY,
    JointRZ,
    JointSpherical,
    JointTranslation,
    JointUniverse,
)
from better_robot.data_model.joint_models.base import JointModel
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics


_IDENTITY = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _assert_fk_runs(model) -> None:
    data = forward_kinematics(model, model.q_neutral)
    assert data.joint_pose_world is not None
    assert data.joint_pose_world.shape == (model.njoints, 7)
    assert torch.isfinite(data.joint_pose_world).all()
    assert model.lower_pos_limit.shape == (model.nq,)
    assert model.upper_pos_limit.shape == (model.nq,)
    assert model.velocity_limit.shape == (model.nv,)
    assert model.effort_limit.shape == (model.nv,)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_kind"),
    [
        pytest.param(
            "add_revolute",
            {"axis": torch.tensor([0.7071068, 0.7071068, 0.0])},
            "revolute_unaligned",
            id="revolute",
        ),
        pytest.param(
            "add_revolute",
            {"axis": torch.tensor([0.0, 0.0, 1.0]), "unbounded": True},
            "revolute_unbounded",
            id="continuous",
        ),
        pytest.param("add_revolute_x", {}, "revolute_rx", id="revolute_x"),
        pytest.param("add_revolute_y", {}, "revolute_ry", id="revolute_y"),
        pytest.param("add_revolute_z", {}, "revolute_rz", id="revolute_z"),
        pytest.param(
            "add_prismatic",
            {"axis": torch.tensor([0.7071068, 0.7071068, 0.0])},
            "prismatic_unaligned",
            id="prismatic",
        ),
        pytest.param("add_prismatic_x", {}, "prismatic_px", id="prismatic_x"),
        pytest.param("add_prismatic_y", {}, "prismatic_py", id="prismatic_y"),
        pytest.param("add_prismatic_z", {}, "prismatic_pz", id="prismatic_z"),
        pytest.param("add_spherical", {}, "spherical", id="spherical"),
        pytest.param("add_planar", {}, "planar", id="planar"),
        pytest.param(
            "add_helical",
            {"axis": torch.tensor([0.0, 0.0, 1.0]), "pitch": 0.25},
            "helical",
            id="helical",
        ),
        pytest.param("add_fixed", {}, "fixed", id="fixed"),
    ],
)
def test_named_joint_methods_build_and_run_fk(method, kwargs, expected_kind):
    builder = ModelBuilder(method)
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    getattr(builder, method)("joint", parent=base, child=tip, **kwargs)

    model = build_model(builder.finalize())

    assert model.joint_models[2].kind == expected_kind
    _assert_fk_runs(model)


def test_add_free_flyer_root_builds_and_runs_fk():
    builder = ModelBuilder("free_flyer")
    base = builder.add_body("base")
    builder.add_free_flyer_root(child=base)

    model = build_model(builder.finalize())

    assert model.joint_models[1].kind == "free_flyer"
    _assert_fk_runs(model)


_JointFactory = Callable[[], JointModel]


@dataclass(frozen=True)
class _CustomRevolute(JointRX):
    """Minimal out-of-tree-style JointModel with a non-built-in kind."""

    kind: str = "custom_revolute"


@pytest.mark.parametrize(
    ("factory", "expected_kind"),
    [
        pytest.param(JointUniverse, "universe", id="universe"),
        pytest.param(JointFixed, "fixed", id="fixed"),
        pytest.param(JointRX, "revolute_rx", id="revolute_rx"),
        pytest.param(JointRY, "revolute_ry", id="revolute_ry"),
        pytest.param(JointRZ, "revolute_rz", id="revolute_rz"),
        pytest.param(
            lambda: JointRevoluteUnaligned(torch.tensor([1.0, 1.0, 0.0])),
            "revolute_unaligned",
            id="revolute_unaligned",
        ),
        pytest.param(JointRevoluteUnbounded, "revolute_unbounded", id="continuous"),
        pytest.param(JointPX, "prismatic_px", id="prismatic_px"),
        pytest.param(JointPY, "prismatic_py", id="prismatic_py"),
        pytest.param(JointPZ, "prismatic_pz", id="prismatic_pz"),
        pytest.param(
            lambda: JointPrismaticUnaligned(torch.tensor([1.0, 1.0, 0.0])),
            "prismatic_unaligned",
            id="prismatic_unaligned",
        ),
        pytest.param(JointSpherical, "spherical", id="spherical"),
        pytest.param(JointFreeFlyer, "free_flyer", id="free_flyer"),
        pytest.param(JointPlanar, "planar", id="planar"),
        pytest.param(JointTranslation, "translation", id="translation"),
        pytest.param(
            lambda: JointHelical(torch.tensor([0.0, 0.0, 1.0]), pitch=0.25),
            "helical",
            id="helical",
        ),
        pytest.param(
            lambda: JointComposite((JointRX(), JointPX())),
            "composite",
            id="composite",
        ),
        pytest.param(_CustomRevolute, "custom_revolute", id="custom"),
    ],
)
def test_joint_model_instances_round_trip_and_run_fk(
    factory: _JointFactory,
    expected_kind: str,
):
    joint_model = factory()
    builder = ModelBuilder(expected_kind)
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_joint(
        "joint",
        kind=joint_model,
        parent=base,
        child=tip,
    )

    model = build_model(builder.finalize())

    assert model.joint_models[2] is joint_model
    assert model.joint_models[2].kind == expected_kind
    _assert_fk_runs(model)


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
    data = forward_kinematics(model, q)

    assert isinstance(model.joint_models[2], JointHelical)
    assert model.joint_models[2].pitch == pytest.approx(0.25)
    assert data.joint_pose_world is not None
    assert data.joint_pose_world[2, 2].item() == pytest.approx(0.1)


def test_composite_derives_dimensions_and_runs_fk():
    composite = JointComposite((JointRX(), JointPX()))
    assert composite.nq == 2
    assert composite.nv == 2

    builder = ModelBuilder("composite")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_joint(
        "joint",
        kind=composite,
        parent=base,
        child=tip,
        origin=_IDENTITY,
    )
    model = build_model(builder.finalize())

    data = forward_kinematics(model, torch.tensor([0.2, 0.3]))
    assert data.joint_pose_world is not None
    assert torch.isfinite(data.joint_pose_world).all()


def test_direct_joint_mimic_requires_a_concrete_target_kind():
    builder = ModelBuilder("mimic_placeholder")
    base = builder.add_body("base")
    source = builder.add_body("source")
    target = builder.add_body("target")
    builder.add_revolute_x("source_joint", parent=base, child=source)
    builder.add_joint(
        "target_joint",
        kind=JointMimic(),
        parent=base,
        child=target,
        mimic_source="source_joint",
    )

    with pytest.raises(NotImplementedError, match="concrete"):
        build_model(builder.finalize())
