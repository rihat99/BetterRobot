"""Tests for ``build_model``: an IR becomes a frozen ``Model`` (``docs/concepts/io.md``)."""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import ClassVar, final

import pytest
import torch

from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.io.ir import IRBody, IRFrame, IRJoint, IRModel
from better_robot.kinematics import subtree
from better_robot.exceptions import ModelError, UnknownNameError
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
from better_robot.model.manifolds import ROTATION, JointManifold

_INF = math.inf


def _id7() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _simple_arm_ir():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute(
        "j1",
        axis=(0.0, 0.0, 1.0),
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
        velocity_limit=2.0,
        effort_limit=100.0,
    )
    return b.finalize()


def test_frozen():
    model = build_model(_simple_arm_ir())
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        model.nq = 99  # type: ignore[misc]


def test_universe_joint_zero():
    model = build_model(_simple_arm_ir())
    assert model.joint_names[0] == "universe"
    assert model.joints[0] == JointFixed()
    assert model.parent_joint_ids[0] == -1


def test_root_joint_is_fixed_by_default():
    model = build_model(_simple_arm_ir())
    assert model.joint_names[1] == "root_joint"
    assert model.joints[1] == JointFixed()
    assert model.parent_joint_ids[1] == 0


def test_limits():
    model = build_model(_simple_arm_ir())
    assert abs(float(model.lower_position_limit[0]) - (-math.pi)) < 1e-5
    assert abs(float(model.upper_position_limit[0]) - math.pi) < 1e-5
    assert abs(float(model.velocity_limit[0]) - 2.0) < 1e-5
    assert abs(float(model.effort_limit[0]) - 100.0) < 1e-5


def _limits(model) -> list[list[float]]:
    return [t.tolist() for t in (model.lower_position_limit, model.upper_position_limit, model.velocity_limit)] + [
        model.effort_limit.tolist()
    ]


@pytest.mark.parametrize(
    ("root_joint", "lower", "upper"),
    [
        pytest.param(JointFreeFlyer(), [-_INF] * 7, [_INF] * 7, id="free_flyer"),
        pytest.param(JointSpherical(), [-1.0] * 4, [1.0] * 4, id="spherical"),
        pytest.param(JointPlanar(), [-_INF, -_INF, -1.0, -1.0], [_INF, _INF, 1.0, 1.0], id="planar"),
        pytest.param(JointTranslation(), [-_INF] * 3, [_INF] * 3, id="translation"),
        pytest.param(JointRevoluteUnbounded(axis=(0.0, 0.0, 1.0)), [-_INF] * 2, [_INF] * 2, id="continuous"),
        pytest.param(JointRevolute(axis=(0.0, 0.0, 1.0)), [-_INF], [_INF], id="revolute"),
        pytest.param(JointPrismatic(axis=(1.0, 0.0, 0.0)), [-_INF], [_INF], id="prismatic"),
        pytest.param(JointHelical(axis=(0.0, 0.0, 1.0), pitch=0.1), [-_INF], [_INF], id="helical"),
    ],
)
def test_a_synthetic_root_moves_unlimited_inside_its_manifold_bounds(root_joint, lower, upper):
    """A root no description wrote has no velocity or effort limit (zero would pin it); the arm keeps its own."""
    model = build_model(_simple_arm_ir(), root_joint=root_joint, dtype=torch.float64)
    nv = root_joint.nv
    assert _limits(model) == [lower + [-math.pi], upper + [math.pi], [_INF] * nv + [2.0], [_INF] * nv + [100.0]]


def test_a_root_joint_that_replaces_the_world_joint_keeps_only_its_placement():
    builder = ModelBuilder("mounted")
    builder.add_body("base")
    mount = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    limits = {"lower": -0.5, "upper": 0.5, "velocity_limit": 1.0, "effort_limit": 1.0}
    builder.add_revolute("mount", axis=(0.0, 0.0, 1.0), parent="world", child="base", origin=mount, **limits)

    model = build_model(builder.finalize(), root_joint=JointSpherical())

    assert torch.equal(model.joint_placements[1], mount)
    assert _limits(model) == [[-1.0] * 4, [1.0] * 4, [_INF] * 3, [_INF] * 3]


@final
@dataclass(frozen=True)
class _Ball(JointModel):
    """An out-of-tree ball joint: a kind on the ROTATION manifold that ``io`` does not know (record 0025)."""

    nq: ClassVar[int] = 4
    nv: ClassVar[int] = 3
    manifold: ClassVar[JointManifold] = ROTATION
    axis: ClassVar[None] = None

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return JointSpherical().placement(q)

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return JointSpherical().motion_subspace(q)

    def neutral(self) -> torch.Tensor:
        return JointSpherical().neutral()

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return JointSpherical().random(generator, lower=lower, upper=upper)


@pytest.mark.parametrize("kind", [pytest.param(JointSpherical(), id="built_in"), pytest.param(_Ball(), id="custom")])
def test_the_position_bounds_follow_the_joint_manifold(kind):
    """A quaternion lies in [-1, 1] whatever the kind; the IR joint's angle bounds do not apply to it."""
    builder = ModelBuilder("ball")
    builder.add_body("base")
    builder.add_body("tip")
    builder.add_joint("ball", kind=kind, parent="base", child="tip", lower=-0.5, upper=0.5, velocity_limit=3.0)

    model = build_model(builder.finalize())

    assert _limits(model) == [[-1.0] * 4, [1.0] * 4, [3.0] * 3, [0.0] * 3]


@pytest.mark.parametrize(
    "body_index, mass",
    [
        pytest.param(1, 0.5, id="root_base"),
        pytest.param(2, 1.0, id="child_link1"),
    ],
)
def test_body_inertia_mass(body_index, mass):
    model = build_model(_simple_arm_ir())
    assert abs(float(model.body_inertias[body_index, 0]) - mass) < 1e-5


def _model_with_ir_frame():
    b = ModelBuilder("f")
    b.add_body("root")
    b.add_body("child")
    b.add_fixed("j1", parent="root", child="child", origin=_id7())
    b.add_frame("tip", parent="child", placement=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]))
    return build_model(b.finalize())


@pytest.mark.parametrize(
    "make_model, expected_frames",
    [
        pytest.param(lambda: build_model(_simple_arm_ir()), ("universe", "base", "link1"), id="body_frames"),
        pytest.param(_model_with_ir_frame, ("tip",), id="ir_frame"),
    ],
)
def test_frames_created(make_model, expected_frames):
    model = make_model()
    for name in expected_frames:
        assert name in set(model.frame_names)
        assert name in model.frame_names


def test_name_to_id_dicts():
    model = build_model(_simple_arm_ir())
    assert model.joint_id("universe") == 0
    assert model.joint_id("root_joint") == 1
    assert model.joint_id("j1") == 2
    assert model.body_id("base") == 1
    assert model.body_id("link1") == 2


def test_parents_precede_children():
    model = build_model(_simple_arm_ir())
    assert model.parent_joint_ids[0] == -1
    assert all(parent < joint for joint, parent in enumerate(model.parent_joint_ids) if joint)


def _branching_tree_ir():
    """Y-shape with two-level depth on each arm.

    Tree (joints declared left-branch first, both levels):
        root → left_j1 → left_j2
        root → right_j1 → right_j2

    The joints are declared in a topological order (the left arm, then the right), which the
    build keeps: joint 3 is ``left_j2``, not the breadth-first ``right_j1``.
    """
    b = ModelBuilder("y_tree")
    b.add_body("root")
    b.add_body("left_link1")
    b.add_body("left_link2")
    b.add_body("right_link1")
    b.add_body("right_link2")
    b.add_revolute(
        "left_j1", axis=(0.0, 0.0, 1.0), parent="root", child="left_link1", origin=_id7(), lower=-1.0, upper=1.0
    )
    b.add_revolute(
        "left_j2", axis=(0.0, 0.0, 1.0), parent="left_link1", child="left_link2", origin=_id7(), lower=-1.0, upper=1.0
    )
    b.add_revolute(
        "right_j1", axis=(0.0, 0.0, 1.0), parent="root", child="right_link1", origin=_id7(), lower=-1.0, upper=1.0
    )
    b.add_revolute(
        "right_j2",
        axis=(0.0, 0.0, 1.0),
        parent="right_link1",
        child="right_link2",
        origin=_id7(),
        lower=-1.0,
        upper=1.0,
    )
    return b.finalize()


def test_idx_qs_follow_the_declared_order_on_a_branching_tree():
    """``joint_names``/``idx_qs`` on a Y-shape keep the declared (topological) order:
    ``(universe, root, left_j1, left_j2, right_j1, right_j2)``, not the breadth-first
    ``(..., left_j1, right_j1, left_j2, right_j2)``.
    """
    model = build_model(_branching_tree_ir())
    assert model.joint_names == (
        "universe",
        "root_joint",
        "left_j1",
        "left_j2",
        "right_j1",
        "right_j2",
    )
    # Four revolute joints → idx_qs increments 1 per joint after the fixed root
    assert model.idx_qs == (0, 0, 0, 1, 2, 3)
    assert model.idx_vs == (0, 0, 0, 1, 2, 3)
    # each subtree is a contiguous range of joint ids
    assert subtree(model, joint=2) == (2, 3)  # left_j1 subtree = {left_j1, left_j2}
    assert subtree(model, joint=4) == (4, 5)  # right_j1 subtree = {right_j1, right_j2}


def _unaligned_model():
    b = ModelBuilder("ua")
    b.add_body("base")
    b.add_body("child")
    b.add_revolute(
        "j1", parent="base", child="child", axis=torch.tensor([0.6, 0.8, 0.0]), origin=_id7(), lower=-1.0, upper=1.0
    )
    return build_model(b.finalize())


def _continuous_model():
    b = ModelBuilder("cont")
    b.add_body("base")
    b.add_body("wheel")
    b.add_revolute(
        "wheel_spin",
        parent="base",
        child="wheel",
        axis=torch.tensor([0.0, 0.0, 1.0]),
        origin=_id7(),
        unbounded=True,
    )
    return build_model(b.finalize())


def _spherical_model():
    b = ModelBuilder("ball")
    b.add_body("base")
    b.add_body("child")
    b.add_spherical("j1", parent="base", child="child", origin=_id7())
    return build_model(b.finalize())


@pytest.mark.parametrize(
    "make_model, joint_index, cls, nq, nv",
    [
        # joint index 2 = j1 (revolute around Z)
        pytest.param(lambda: build_model(_simple_arm_ir()), 2, JointRevolute, 1, 1, id="revolute_z"),
        pytest.param(_unaligned_model, 2, JointRevolute, 1, 1, id="revolute_unaligned"),
        pytest.param(_continuous_model, 2, JointRevoluteUnbounded, 2, 1, id="continuous"),  # nq = cos,sin
        pytest.param(_spherical_model, 2, JointSpherical, 4, 3, id="spherical"),
        # nq = 7 (free flyer) + 1 (revolute) = 8
        pytest.param(
            lambda: build_model(_simple_arm_ir(), root_joint=JointFreeFlyer()),
            1,
            JointFreeFlyer,
            8,
            7,
            id="free_flyer_root",
        ),
    ],
)
def test_joint_kind_mapping(make_model, joint_index, cls, nq, nv):
    model = make_model()
    assert isinstance(model.joints[joint_index], cls)
    assert model.nq == nq
    assert model.nv == nv


def test_multi_joint_chain():
    b = ModelBuilder("chain")
    b.add_body("root")
    b.add_body("l1")
    b.add_body("l2")
    b.add_body("l3")
    b.add_revolute("j1", axis=(0.0, 0.0, 1.0), parent="root", child="l1", origin=_id7(), lower=-1.0, upper=1.0)
    b.add_revolute("j2", axis=(0.0, 1.0, 0.0), parent="l1", child="l2", origin=_id7(), lower=-1.0, upper=1.0)
    b.add_prismatic("j3", axis=(1.0, 0.0, 0.0), parent="l2", child="l3", origin=_id7(), lower=0.0, upper=0.5)
    ir = b.finalize()
    model = build_model(ir)
    assert model.nq == 3
    assert model.nv == 3
    assert model.njoints == 5  # universe + root + 3 ir joints


def _mimic_ir(*, multiplier: float = 1.0, offset: float = 0.0):
    b = ModelBuilder("mimic")
    b.add_body("root")
    b.add_body("finger1")
    b.add_body("finger2")
    b.add_revolute("main_j", axis=(0.0, 0.0, 1.0), parent="root", child="finger1", origin=_id7(), lower=-1.0, upper=1.0)
    b.add_revolute(
        "mimic_j",
        axis=(0.0, 0.0, 1.0),
        parent="root",
        child="finger2",
        origin=_id7(),
        lower=-1.0,
        upper=1.0,
        mimic_source="main_j",
        mimic_multiplier=multiplier,
        mimic_offset=offset,
    )
    return b.finalize()


@pytest.mark.parametrize(("multiplier", "offset"), [(0.5, 0.0), (1.0, 0.1)])
def test_non_identity_mimic_joint_builds_reduced_map(multiplier, offset):
    model = build_model(_mimic_ir(multiplier=multiplier, offset=offset))
    target = model.joint_id("mimic_j")

    assert model.nq == model.nv == 1
    assert model.nq_full == model.nv_full == 2
    assert model.mimic_source_ids[target] == model.joint_id("main_j")
    assert model.mimic_multipliers[target] == pytest.approx(multiplier)
    assert model.mimic_offsets[target] == pytest.approx(offset)


def test_identity_mimic_joint_preserves_metadata_and_reduces_dofs():
    model = build_model(_mimic_ir())
    main_j_midx = model.joint_id("main_j")
    mimic_j_midx = model.joint_id("mimic_j")

    assert model.mimic_source_ids[mimic_j_midx] == main_j_midx
    assert model.mimic_multipliers[mimic_j_midx] == pytest.approx(1.0)
    assert model.mimic_offsets[mimic_j_midx] == pytest.approx(0.0)
    assert model.nqs[main_j_midx] == model.nvs[main_j_midx] == 1
    assert model.nqs[mimic_j_midx] == model.nvs[mimic_j_midx] == 0
    assert model.nqs_full[main_j_midx] == model.nqs_full[mimic_j_midx] == 1
    assert model.nvs_full[main_j_midx] == model.nvs_full[mimic_j_midx] == 1


def test_mimic_bad_source_raises():
    b = ModelBuilder("m")
    b.add_body("root")
    b.add_body("child")
    b.add_revolute(
        "j1",
        axis=(0.0, 0.0, 1.0),
        parent="root",
        child="child",
        origin=_id7(),
        lower=-1.0,
        upper=1.0,
        mimic_source="nonexistent_joint",
    )
    ir = b.finalize()
    with pytest.raises(ModelError, match="Mimic source"):
        build_model(ir)


def _named_ir(joints: list[tuple[str, str, str]], bodies: list[str], frames: tuple[str, ...] = ()) -> IRModel:
    """An IR of revolute joints ``(name, parent, child)`` between ``bodies``, plus frames on the first body."""
    axis = (0.0, 0.0, 1.0)
    return IRModel(
        name="names",
        bodies=[IRBody(name=body) for body in bodies],
        joints=[IRJoint(name, parent, child, JointRevolute(axis=axis)) for name, parent, child in joints],
        frames=[IRFrame(name, parent_body=bodies[0], placement=_id7()) for name in frames],
    )


@pytest.mark.parametrize(
    ("ir", "message"),
    [
        pytest.param(
            _named_ir([("j", "b0", "b1"), ("j", "b1", "b2")], ["b0", "b1", "b2"]),
            "two joints are named 'j'; rename one",
            id="two_joints",
        ),
        pytest.param(
            _named_ir([("root_joint", "b0", "b1")], ["b0", "b1"]),
            "two joints are named 'root_joint' (build_model names",
            id="joint_named_like_the_added_root",
        ),
        pytest.param(
            _named_ir([("universe", "b0", "b1")], ["b0", "b1"]),
            "two joints are named 'universe' (build_model names",
            id="joint_named_universe",
        ),
        pytest.param(
            _named_ir([("j", "universe", "b1")], ["universe", "b1"]),
            "two bodies are named 'universe' (build_model names",
            id="body_named_universe",
        ),
        pytest.param(
            _named_ir([("j", "b0", "b1")], ["b0", "b1", "b1"]),
            "two bodies are named 'b1'; rename one",
            id="two_bodies",
        ),
        pytest.param(
            _named_ir([("j", "b0", "b1")], ["b0", "b1"], frames=("b1",)),
            "frame 'b1' is already the name of a body or of another frame; rename one",
            id="frame_named_like_a_body",
        ),
    ],
)
def test_a_name_used_twice_raises_and_says_which(ir, message):
    """Names address joints, bodies and frames, so a description that repeats one cannot be built."""
    with pytest.raises(ModelError, match=re.escape(message)):
        build_model(ir)


@pytest.mark.parametrize(("parent", "child"), [("base", "nowhere"), ("nowhere", "tip")])
def test_a_builder_joint_names_bodies_added_before(parent, child):
    builder = ModelBuilder("r")
    builder.add_body("base")
    builder.add_body("tip")
    with pytest.raises(UnknownNameError, match=re.escape("unknown body 'nowhere'")):
        builder.add_revolute("j", parent=parent, child=child, axis=(0.0, 0.0, 1.0))


def test_an_ir_joint_to_a_missing_body_says_what_to_add():
    ir = _named_ir([("j", "b0", "b1")], ["b0"])
    with pytest.raises(
        ModelError, match=re.escape("body 'b1' is moved by a joint but is not among the IR bodies; add")
    ):
        build_model(ir)


def test_an_ir_neutral_is_the_start_clamped_into_the_limits_and_must_fit_the_joint():
    ir = _named_ir([("j", "b0", "b1")], ["b0", "b1"])
    ir.joints[0].lower, ir.joints[0].upper = -1.0, 1.0
    ir.joints[0].neutral = torch.tensor([0.5])
    assert build_model(ir).q_neutral.tolist() == [0.5]
    ir.joints[0].neutral = torch.tensor([3.0])
    assert build_model(ir).q_neutral.tolist() == [1.0]
    ir.joints[0].neutral = torch.zeros(2)
    with pytest.raises(ModelError, match=re.escape("joint 'j': neutral has shape (2,), expected (1,)")):
        build_model(ir)


def test_device_dtype_kwarg():
    ir = _simple_arm_ir()
    model = build_model(ir, dtype=torch.float64)
    assert model.joint_placements.dtype == torch.float64
    assert model.lower_position_limit.dtype == torch.float64
