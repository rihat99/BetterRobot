"""Tests for the branching-tree fixture — free-flyer + spherical Model + build_model.

Exercises ``build_model`` / ``load`` on a representative 24-body free-flyer +
23-spherical tree (neutral joint names ``j0``..``j23``).
"""

from __future__ import annotations


from better_robot.io import build_model, load
from better_robot.io.ir import IRModel
from better_robot.model.joints import JointFreeFlyer, JointSpherical

from tests.support.branching_tree import JOINT_NAMES, make_branching_tree_body


def test_branching_model():
    branching_model = load(make_branching_tree_body)
    assert branching_model is not None
    assert isinstance(branching_model.joints[1], JointFreeFlyer)
    spherical_count = sum(1 for jm in branching_model.joints if isinstance(jm, JointSpherical))
    assert spherical_count == 23
    # universe + root_joint(free_flyer, from IR "root") + 23 spherical = 25
    assert branching_model.njoints == 25
    # free_flyer(7) + 23*spherical(4) = 7 + 92 = 99
    assert branching_model.nq == 99
    # free_flyer(6) + 23*spherical(3) = 6 + 69 = 75
    assert branching_model.nv == 75
    assert "j0" in branching_model.body_names
    for bname in JOINT_NAMES:
        assert bname in branching_model.body_names, f"Missing body: {bname}"
    assert all(parent < joint for joint, parent in enumerate(branching_model.parents) if joint)
    assert branching_model.q_neutral.shape == (branching_model.nq,)


def test_branching_ir_directly():
    ir = make_branching_tree_body()
    assert isinstance(ir, IRModel)
    # 24 bodies: root + 23 segments (no explicit "world" body)
    assert len(ir.bodies) == 24
    # 24 joints: 1 free_flyer (world→root) + 23 spherical
    assert len(ir.joints) == 24
    # root_body is "j0" — world is only a sentinel, not a real body
    assert ir.root_body == "j0"


def test_branching_custom_height():
    ir = make_branching_tree_body(height=1.8, mass=80.0)
    model = build_model(ir)
    assert model.nq == 99  # topology doesn't change with height
