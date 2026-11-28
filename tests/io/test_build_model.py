"""Tests for ``build_model``: an IR becomes a frozen ``Model``."""

from __future__ import annotations

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
from better_robot.exceptions import DtypeMismatchError, ModelError, UnknownNameError
from better_robot.model.joints import (
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
from better_robot.lie import se3
from better_robot.model.manifolds import ROTATION, JointManifold

from tests.support.robots import one_revolute_arm_ir

_INF = math.inf


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.complex64, "float32"], ids=str)
def test_a_dtype_other_than_float32_or_float64_is_refused(dtype):
    with pytest.raises(DtypeMismatchError, match="is not float32 or float64; pass torch.float32 or torch.float64"):
        build_model(one_revolute_arm_ir(), dtype=dtype)


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
    model = build_model(one_revolute_arm_ir(), root_joint=root_joint, dtype=torch.float64)
    nv = root_joint.nv
    assert _limits(model) == [lower + [-math.pi], upper + [math.pi], [_INF] * nv + [2.0], [_INF] * nv + [100.0]]


def test_a_root_joint_that_replaces_the_world_joint_keeps_only_its_placement():
    builder = ModelBuilder("mounted")
    builder.add_body("base")
    mount = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    limits = {"lower": -0.5, "upper": 0.5, "velocity_limit": 1.0, "effort_limit": 1.0}
    builder.add_joint(
        "mount", kind=JointRevolute(axis=(0.0, 0.0, 1.0)), parent="world", child="base", origin=mount, **limits
    )

    model = build_model(builder.finalize(), root_joint=JointSpherical())

    assert torch.equal(model.joint_placements[1], mount)
    assert _limits(model) == [[-1.0] * 4, [1.0] * 4, [_INF] * 3, [_INF] * 3]


@final
@dataclass(frozen=True)
class _Ball(JointModel):
    """An out-of-tree ball joint: a kind on the ROTATION manifold that ``io`` does not know."""

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
    b.add_joint(
        "left_j1",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="root",
        child="left_link1",
        origin=se3.identity(),
        lower=-1.0,
        upper=1.0,
    )
    b.add_joint(
        "left_j2",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="left_link1",
        child="left_link2",
        origin=se3.identity(),
        lower=-1.0,
        upper=1.0,
    )
    b.add_joint(
        "right_j1",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="root",
        child="right_link1",
        origin=se3.identity(),
        lower=-1.0,
        upper=1.0,
    )
    b.add_joint(
        "right_j2",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="right_link1",
        child="right_link2",
        origin=se3.identity(),
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


def test_mimic_bad_source_raises():
    b = ModelBuilder("m")
    b.add_body("root")
    b.add_body("child")
    b.add_joint(
        "j1",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="root",
        child="child",
        origin=se3.identity(),
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
        joints=[
            IRJoint(name=name, parent_body=parent, child_body=child, joint=JointRevolute(axis=axis))
            for name, parent, child in joints
        ],
        frames=[IRFrame(name=name, parent_body=bodies[0], placement=se3.identity()) for name in frames],
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
        builder.add_joint("j", kind=JointRevolute(axis=(0.0, 0.0, 1.0)), parent=parent, child=child)


def test_an_ir_joint_to_a_missing_body_says_what_to_add():
    ir = _named_ir([("j", "b0", "b1")], ["b0"])
    with pytest.raises(
        ModelError, match=re.escape("body 'b1' is moved by a joint but is not among the IR bodies; add")
    ):
        build_model(ir)


@pytest.mark.parametrize(
    ("ir", "message"),
    [
        pytest.param(
            _named_ir([("j", "b0", "b1")], ["b0", "b1", "stray"]),
            "expected exactly one root body (a body no joint moves), found 2: ['b0', 'stray']; connect or remove",
            id="two_roots",
        ),
        pytest.param(
            _named_ir([("j", "b1", "b0"), ("k", "b0", "b1")], ["b0", "b1"]),
            "expected exactly one root body (a body no joint moves), found 0: []",
            id="no_root",
        ),
        pytest.param(
            _named_ir([("mount", "world", "b0"), ("j", "b0", "b1")], ["b0", "b1", "stray"]),
            "bodies ['stray'] are moved by no joint, and the joint 'mount' from 'world' holds the root 'b0'",
            id="stray_beside_a_world_joint",
        ),
    ],
)
def test_a_body_outside_the_one_tree_raises(ir, message):
    """Every body but the root is moved by a joint. A stray body beside a joint from ``"world"`` was once dropped
    from the model without a word."""
    with pytest.raises(ModelError, match=re.escape(message)):
        build_model(ir)


def test_a_world_link_of_the_description_is_the_universe():
    """A URDF may name the universe as a link ``world``: it is no body of the model and no second root."""
    model = build_model(_named_ir([("mount", "world", "base"), ("j", "base", "tip")], ["world", "base", "tip"]))
    assert model.body_names == ("universe", "base", "tip")


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


@pytest.mark.parametrize("enabled", [True, False], ids=["internal_checks_on", "internal_checks_off"])
@pytest.mark.parametrize(
    ("where", "pose", "message"),
    [
        ("joint", [0, 0, 0.1, 0, 0, 0, 1.1], "joint 'j': origin has a quaternion of norm 1.1; pass a unit quaternion"),
        (
            "root",
            [0, 0, 0.1, 0, 0, 0, 0.9],
            "joint 'root': origin has a quaternion of norm 0.9; pass a unit quaternion",
        ),
        ("frame", [0, 0, 0, 0.6, 0, 0, 0.6], "frame 'tool': placement has a quaternion of norm 0.848528; pass a unit"),
        ("joint", [0, 0, 0, 0, 0, 1], "joint 'j': origin has shape (6,), expected (7,) [tx, ty, tz, qx, qy, qz, qw]"),
    ],
)
def test_a_pose_off_unit_norm_is_the_callers_error(monkeypatch, enabled, where, pose, message):
    """A joint ``origin`` or frame ``placement`` that is not a ``(7,)`` pose with a unit quaternion raises
    ``ModelError`` naming it, whatever the internal-checks switch says (it once raised a "library bug" with the
    switch on and built a non-unit placement with it off)."""
    # Private on purpose: the internal-checks switch has no public name, and this error must not depend on it.
    monkeypatch.setattr("better_robot._internal_checks.ENABLED", enabled)
    given = torch.tensor(pose, dtype=torch.float32)
    builder = ModelBuilder("arm")
    builder.add_body("base")
    builder.add_body("link")
    if where == "root":
        builder.add_joint("root", kind=JointFreeFlyer(), parent="world", child="base", origin=given)
    builder.add_joint(
        "j",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="base",
        child="link",
        origin=given if where == "joint" else None,
    )
    if where == "frame":
        builder.add_frame("tool", parent="link", placement=given)
    with pytest.raises(ModelError, match=re.escape(message)):
        build_model(builder.finalize(), dtype=torch.float64)


def test_a_quaternion_normalised_in_float32_builds_in_float64():
    """The unit tolerance is float32's: a quaternion rounded to float32 is a unit quaternion in a float64 model."""
    builder = ModelBuilder("arm")
    builder.add_body("base")
    builder.add_body("link")
    half = math.sqrt(0.5)
    origin = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, half, half], dtype=torch.float32)
    builder.add_joint("j", kind=JointRevolute(axis=(0.0, 0.0, 1.0)), parent="base", child="link", origin=origin)
    model = build_model(builder.finalize(), dtype=torch.float64)
    assert model.joint_placements[2].tolist() == origin.double().tolist()
