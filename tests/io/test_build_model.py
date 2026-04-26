"""Tests for ``build_model`` — IR → frozen Model factory.

See ``docs/concepts/parsers_and_ir.md §3``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.io.ir import IRError
from better_robot.data_model.model import Model
from better_robot.data_model.joint_models import (
    JointFixed,
    JointFreeFlyer,
    JointRX,
    JointRZ,
    JointPX,
    JointSpherical,
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
)


def _id7() -> torch.Tensor:
    return torch.tensor([0., 0., 0., 0., 0., 0., 1.])


def _simple_arm_ir():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z("j1", parent="base", child="link1",
                     origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
                     lower=-math.pi, upper=math.pi,
                     velocity_limit=2.0, effort_limit=100.)
    return b.finalize()


def test_returns_model():
    model = build_model(_simple_arm_ir())
    assert isinstance(model, Model)


def test_frozen():
    model = build_model(_simple_arm_ir())
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        model.nq = 99  # type: ignore[misc]


import dataclasses


def test_joint_count():
    # 2 bodies in IR → 2 + 2 model joints (universe + root + ir joints)
    model = build_model(_simple_arm_ir())
    assert model.njoints == 3  # universe, root_joint, j1


def test_universe_joint_zero():
    model = build_model(_simple_arm_ir())
    assert model.joint_names[0] == "universe"
    assert model.joint_models[0].kind == "universe"
    assert model.parents[0] == -1


def test_root_joint_is_fixed_by_default():
    model = build_model(_simple_arm_ir())
    assert model.joint_names[1] == "root_joint"
    assert model.joint_models[1].kind == "fixed"
    assert model.parents[1] == 0


def test_revolute_z_mapped_to_jrz():
    model = build_model(_simple_arm_ir())
    # joint index 2 = j1 (revolute around Z)
    assert model.joint_models[2].kind == "revolute_rz"


def test_nq_nv():
    model = build_model(_simple_arm_ir())
    # JointUniverse(0) + JointFixed(0) + JointRZ(1) → nq=1, nv=1
    assert model.nq == 1
    assert model.nv == 1


def test_idx_qs():
    model = build_model(_simple_arm_ir())
    # universe: idx_q=0, fixed: idx_q=0, j1: idx_q=0
    assert model.idx_qs[0] == 0
    assert model.idx_qs[1] == 0
    assert model.idx_qs[2] == 0


def test_limits():
    model = build_model(_simple_arm_ir())
    assert abs(float(model.lower_pos_limit[0]) - (-math.pi)) < 1e-5
    assert abs(float(model.upper_pos_limit[0]) - math.pi) < 1e-5
    assert abs(float(model.velocity_limit[0]) - 2.0) < 1e-5
    assert abs(float(model.effort_limit[0]) - 100.0) < 1e-5


def test_joint_placements_shape():
    model = build_model(_simple_arm_ir())
    assert model.joint_placements.shape == (model.njoints, 7)


def test_body_inertias_shape():
    model = build_model(_simple_arm_ir())
    assert model.body_inertias.shape == (model.nbodies, 10)


def test_root_body_inertia():
    model = build_model(_simple_arm_ir())
    # body 1 = base (mass=0.5)
    assert abs(float(model.body_inertias[1, 0]) - 0.5) < 1e-5


def test_child_body_inertia():
    model = build_model(_simple_arm_ir())
    # body 2 = link1 (mass=1.0)
    assert abs(float(model.body_inertias[2, 0]) - 1.0) < 1e-5


def test_body_frames_created():
    model = build_model(_simple_arm_ir())
    # Should have body_universe, body_base, body_link1
    frame_names = set(model.frame_names)
    assert "body_universe" in frame_names
    assert "body_base" in frame_names
    assert "body_link1" in frame_names


def test_name_to_id_dicts():
    model = build_model(_simple_arm_ir())
    assert model.joint_id("universe") == 0
    assert model.joint_id("root_joint") == 1
    assert model.joint_id("j1") == 2
    assert model.body_id("base") == 1
    assert model.body_id("link1") == 2


def test_topo_order_parent_before_child():
    model = build_model(_simple_arm_ir())
    order = model.topo_order
    for i, j in enumerate(order):
        p = model.parents[j]
        if p >= 0:
            assert order.index(p) < i


def _branching_tree_ir():
    """Y-shape with two-level depth on each arm.

    Tree (joints declared left-branch first, both levels):
        root → left_j1 → left_j2
        root → right_j1 → right_j2

    Under DFS, the left arm is fully traversed before the right — so joint 3
    is ``left_j2`` (not ``right_j1``). Under BFS, siblings are visited before
    grandchildren, so joint 3 would be ``right_j1``.
    """
    b = ModelBuilder("y_tree")
    b.add_body("root")
    b.add_body("left_link1")
    b.add_body("left_link2")
    b.add_body("right_link1")
    b.add_body("right_link2")
    b.add_revolute_z("left_j1", parent="root", child="left_link1",
                     origin=_id7(), lower=-1., upper=1.)
    b.add_revolute_z("left_j2", parent="left_link1", child="left_link2",
                     origin=_id7(), lower=-1., upper=1.)
    b.add_revolute_z("right_j1", parent="root", child="right_link1",
                     origin=_id7(), lower=-1., upper=1.)
    b.add_revolute_z("right_j2", parent="right_link1", child="right_link2",
                     origin=_id7(), lower=-1., upper=1.)
    return b.finalize()


def test_idx_qs_dfs_order_on_branching_tree():
    """Exact DFS-ordered ``joint_names``/``idx_qs`` on a Y-shape.

    Under DFS, the left arm is fully descended before the right sibling:
    ``(universe, root, left_j1, left_j2, right_j1, right_j2)``. Under BFS
    it would be ``(..., left_j1, right_j1, left_j2, right_j2)`` — this
    test would fail under the old ordering.
    """
    model = build_model(_branching_tree_ir())
    assert model.joint_names == (
        "universe", "root_joint",
        "left_j1", "left_j2", "right_j1", "right_j2",
    )
    # Four revolute joints → idx_qs increments 1 per joint after the fixed root
    assert model.idx_qs == (0, 0, 0, 1, 2, 3)
    assert model.idx_vs == (0, 0, 0, 1, 2, 3)
    # DFS signature: topo_order is the identity tuple
    assert model.topo_order == (0, 1, 2, 3, 4, 5)
    # DFS signature: each subtree is a contiguous range of topo_order positions
    assert model.subtrees[2] == (2, 3)  # left_j1 subtree = {left_j1, left_j2}
    assert model.subtrees[4] == (4, 5)  # right_j1 subtree = {right_j1, right_j2}


def test_free_flyer_root_joint():
    ir = _simple_arm_ir()
    model = build_model(ir, root_joint=JointFreeFlyer())
    assert model.joint_models[1].kind == "free_flyer"
    # nq = 7 (free flyer) + 1 (revolute) = 8
    assert model.nq == 8
    assert model.nv == 7


def test_free_flyer_shortcut_via_load():
    from better_robot.io import load
    ir_fn = _simple_arm_ir
    model = load(lambda: _simple_arm_ir(), free_flyer=True)
    assert model.joint_models[1].kind == "free_flyer"


def test_revolute_unaligned():
    b = ModelBuilder("ua")
    b.add_body("base")
    b.add_body("child")
    b.add_revolute("j1", parent="base", child="child",
                   axis=torch.tensor([0.707, 0.707, 0.0]),
                   origin=_id7(), lower=-1., upper=1.)
    ir = b.finalize()
    model = build_model(ir)
    assert model.joint_models[2].kind == "revolute_unaligned"


def test_continuous_joint():
    b = ModelBuilder("cont")
    b.add_body("base")
    b.add_body("wheel")
    b.add_revolute(
        "wheel_spin", parent="base", child="wheel",
        axis=torch.tensor([0., 0., 1.]), origin=_id7(),
        unbounded=True,
    )
    ir = b.finalize()
    model = build_model(ir)
    assert model.joint_models[2].kind == "revolute_unbounded"
    assert model.nq == 2  # cos,sin
    assert model.nv == 1


def test_spherical_joint():
    b = ModelBuilder("ball")
    b.add_body("base")
    b.add_body("child")
    b.add_spherical("j1", parent="base", child="child", origin=_id7())
    ir = b.finalize()
    model = build_model(ir)
    assert model.joint_models[2].kind == "spherical"
    assert model.nq == 4
    assert model.nv == 3


def test_multi_joint_chain():
    b = ModelBuilder("chain")
    b.add_body("root")
    b.add_body("l1")
    b.add_body("l2")
    b.add_body("l3")
    b.add_revolute_z("j1", parent="root", child="l1",
                     origin=_id7(), lower=-1., upper=1.)
    b.add_revolute_y("j2", parent="l1", child="l2",
                     origin=_id7(), lower=-1., upper=1.)
    b.add_prismatic_x("j3", parent="l2", child="l3",
                      origin=_id7(), lower=0., upper=0.5)
    ir = b.finalize()
    model = build_model(ir)
    assert model.nq == 3
    assert model.nv == 3
    assert model.njoints == 5  # universe + root + 3 ir joints


def test_mimic_joint():
    b = ModelBuilder("mimic")
    b.add_body("root")
    b.add_body("finger1")
    b.add_body("finger2")
    b.add_revolute_z("main_j", parent="root", child="finger1",
                     origin=_id7(), lower=-1., upper=1.)
    b.add_revolute_z("mimic_j", parent="root", child="finger2",
                     origin=_id7(), lower=-1., upper=1.,
                     mimic_source="main_j", mimic_multiplier=0.5, mimic_offset=0.1)
    ir = b.finalize()
    model = build_model(ir)
    # mimic_source for mimic_j should point to main_j
    main_j_midx = model.joint_id("main_j")
    mimic_j_midx = model.joint_id("mimic_j")
    assert model.mimic_source[mimic_j_midx] == main_j_midx
    assert abs(float(model.mimic_multiplier[mimic_j_midx]) - 0.5) < 1e-6
    assert abs(float(model.mimic_offset[mimic_j_midx]) - 0.1) < 1e-6


def test_mimic_bad_source_raises():
    b = ModelBuilder("m")
    b.add_body("root")
    b.add_body("child")
    b.add_revolute_z("j1", parent="root", child="child",
                     origin=_id7(), lower=-1., upper=1.,
                     mimic_source="nonexistent_joint")
    ir = b.finalize()
    with pytest.raises(IRError, match="Mimic source"):
        build_model(ir)


def test_q_neutral():
    b = ModelBuilder("n")
    b.add_body("root")
    b.add_body("child")
    b.add_revolute_z("j1", parent="root", child="child",
                     origin=_id7(), lower=-1., upper=1.)
    ir = b.finalize()
    model = build_model(ir)
    assert model.q_neutral.shape == (model.nq,)
    assert float(model.q_neutral[0]) == pytest.approx(0.0)


def test_ir_frame_added():
    from better_robot.io.ir import IRFrame
    b = ModelBuilder("f")
    b.add_body("root")
    b.add_body("child")
    b.add_fixed("j1", parent="root", child="child", origin=_id7())
    b.add_frame("tip", parent_body="child",
                placement=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]))
    ir = b.finalize()
    model = build_model(ir)
    assert "tip" in model.frame_name_to_id


def test_device_dtype_kwarg():
    ir = _simple_arm_ir()
    model = build_model(ir, dtype=torch.float64)
    assert model.joint_placements.dtype == torch.float64
    assert model.lower_pos_limit.dtype == torch.float64
