"""Tests for the generic kinematic-tree builder."""

from __future__ import annotations

import pytest
import torch

from better_robot.io import build_model
from better_robot.io.builders.kinematic_tree import (
    build_kinematic_tree_body,
    build_kinematic_tree_model,
)
from better_robot.io.builders.smpl_like import (
    JOINT_NAMES,
    PARENTS,
    _default_offsets_tensor,
    make_smpl_like_body,
)
from better_robot.io.ir import IRModel


def test_minimal_tree_single_body():
    """A 1-body tree — just a free-flyer root, no children."""
    ir = build_kinematic_tree_body(
        name="single",
        joint_names=["root"],
        parents=[-1],
        translations=torch.zeros(1, 3),
        root_kind="free_flyer",
    )
    assert isinstance(ir, IRModel)
    assert len(ir.bodies) == 1
    assert len(ir.joints) == 1
    assert ir.joints[0].kind == "free_flyer"
    assert ir.joints[0].parent_body == "world"
    assert ir.joints[0].child_body == "root"
    assert ir.root_body == "root"


def test_three_body_tree_free_flyer_spherical():
    """A 3-body tree: free-flyer root + 2 spherical children."""
    translations = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    model = build_kinematic_tree_model(
        name="tiny",
        joint_names=["a", "b", "c"],
        parents=[-1, 0, 1],
        translations=translations,
        mass_per_body=1.0,
    )
    # universe + root(free_flyer) + 2 spherical = 4 joints
    assert model.njoints == 4
    # free_flyer(7) + 2*spherical(4) = 15
    assert model.nq == 15
    # free_flyer(6) + 2*spherical(3) = 12
    assert model.nv == 12


def test_per_body_mass_sequence():
    translations = torch.zeros(3, 3, dtype=torch.float32)
    ir = build_kinematic_tree_body(
        name="m",
        joint_names=["a", "b", "c"],
        parents=[-1, 0, 0],
        translations=translations,
        mass_per_body=[1.0, 2.0, 3.0],
    )
    masses = [b.mass for b in ir.bodies]
    assert masses == [1.0, 2.0, 3.0]


def test_mass_sequence_length_mismatch_raises():
    translations = torch.zeros(2, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="mass_per_body length"):
        build_kinematic_tree_body(
            name="m",
            joint_names=["a", "b"],
            parents=[-1, 0],
            translations=translations,
            mass_per_body=[1.0, 2.0, 3.0],
        )


def test_parents_length_mismatch_raises():
    with pytest.raises(ValueError, match="parents length"):
        build_kinematic_tree_body(
            name="m",
            joint_names=["a", "b"],
            parents=[-1],
            translations=torch.zeros(2, 3),
        )


def test_parents_root_not_minus_one_raises():
    with pytest.raises(ValueError, match="parents\\[0\\] must be -1"):
        build_kinematic_tree_body(
            name="m",
            joint_names=["a", "b"],
            parents=[0, 0],
            translations=torch.zeros(2, 3),
        )


def test_translations_bad_shape_raises():
    with pytest.raises(ValueError, match="translations shape"):
        build_kinematic_tree_body(
            name="m",
            joint_names=["a", "b"],
            parents=[-1, 0],
            translations=torch.zeros(3, 3),
        )


def test_parent_index_out_of_range_raises():
    with pytest.raises(ValueError, match="must reference an earlier index"):
        build_kinematic_tree_body(
            name="m",
            joint_names=["a", "b", "c"],
            parents=[-1, 0, 5],
            translations=torch.zeros(3, 3),
        )


def test_parent_forward_reference_raises():
    """A non-root joint may only reference an earlier index."""
    with pytest.raises(ValueError, match="must reference an earlier index"):
        build_kinematic_tree_body(
            name="m",
            joint_names=["a", "b", "c"],
            parents=[-1, 2, 0],
            translations=torch.zeros(3, 3),
        )


def test_smpl_parity_ir():
    """`make_smpl_like_body()` must match a direct `build_kinematic_tree_body` call."""
    ir_smpl = make_smpl_like_body(height=1.75, mass=70.0)
    ir_direct = build_kinematic_tree_body(
        name="smpl_body",
        joint_names=JOINT_NAMES,
        parents=PARENTS,
        translations=_default_offsets_tensor(1.75),
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=70.0 / 24.0,
    )
    assert ir_smpl.name == ir_direct.name
    assert ir_smpl.root_body == ir_direct.root_body
    assert len(ir_smpl.bodies) == len(ir_direct.bodies) == 24
    assert len(ir_smpl.joints) == len(ir_direct.joints) == 24
    for b0, b1 in zip(ir_smpl.bodies, ir_direct.bodies):
        assert b0.name == b1.name
        assert b0.mass == b1.mass
    for j0, j1 in zip(ir_smpl.joints, ir_direct.joints):
        assert j0.name == j1.name
        assert j0.kind == j1.kind
        assert j0.parent_body == j1.parent_body
        assert j0.child_body == j1.child_body
        assert torch.equal(j0.origin, j1.origin)


def test_smpl_like_accepts_joint_offsets_override():
    """Shape-aware callers can supply custom offsets; default path ignored."""
    offsets = torch.full((24, 3), 0.5, dtype=torch.float32)
    offsets[0].zero_()
    ir = make_smpl_like_body(joint_offsets=offsets)
    assert len(ir.joints) == 24
    # Root joint origin is translations[0] — all zeros.
    assert torch.equal(ir.joints[0].origin[:3], torch.zeros(3))
    # Second joint (left_hip) origin translation must match offsets[1].
    assert torch.allclose(ir.joints[1].origin[:3], torch.full((3,), 0.5))


def test_build_model_runs_on_generic_tree():
    """End-to-end: build a 3-body tree and push it through build_model."""
    translations = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        dtype=torch.float32,
    )
    ir = build_kinematic_tree_body(
        name="tiny",
        joint_names=["a", "b", "c"],
        parents=[-1, 0, 1],
        translations=translations,
        mass_per_body=1.0,
    )
    model = build_model(ir)
    # 3 user bodies + 1 universe body
    assert model.nbodies == 4
    assert model.q_neutral.shape == (model.nq,)
