"""Tests for ``make_smpl_like_body`` — SMPL-topology IRModel + build_model.

Phase 3 pass criterion:
 - ``br.load(make_smpl_like_body)`` returns a ``Model`` with a free-flyer
   root and 23 spherical joints.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.io import load, build_model
from better_robot.io.builders.smpl_like import make_smpl_like_body
from better_robot.data_model.joint_models import JointFreeFlyer, JointSpherical


@pytest.fixture(scope="module")
def smpl_model():
    return load(make_smpl_like_body)


def test_smpl_model_builds(smpl_model):
    assert smpl_model is not None


def test_smpl_root_joint_is_free_flyer(smpl_model):
    assert smpl_model.joint_models[1].kind == "free_flyer"


def test_smpl_has_23_spherical_joints(smpl_model):
    spherical_count = sum(
        1 for jm in smpl_model.joint_models if jm.kind == "spherical"
    )
    assert spherical_count == 23


def test_smpl_njoints(smpl_model):
    # universe + root_joint(free_flyer, from IR "root") + 23 spherical = 25
    assert smpl_model.njoints == 25


def test_smpl_nq(smpl_model):
    # free_flyer(7) + 23*spherical(4) = 7 + 92 = 99
    assert smpl_model.nq == 99


def test_smpl_nv(smpl_model):
    # free_flyer(6) + 23*spherical(3) = 6 + 69 = 75
    assert smpl_model.nv == 75


def test_smpl_body_names_include_pelvis(smpl_model):
    assert "pelvis" in smpl_model.body_name_to_id


def test_smpl_all_joint_names_present(smpl_model):
    expected_bodies = [
        "pelvis", "left_hip", "right_hip", "spine1",
        "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3",
        "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hand", "right_hand",
    ]
    for bname in expected_bodies:
        assert bname in smpl_model.body_name_to_id, f"Missing body: {bname}"


def test_smpl_topo_order_valid(smpl_model):
    order = smpl_model.topo_order
    for pos, j in enumerate(order):
        p = smpl_model.parents[j]
        if p >= 0:
            assert list(order).index(p) < pos


def test_smpl_ir_directly():
    ir = make_smpl_like_body()
    from better_robot.io.ir import IRModel
    assert isinstance(ir, IRModel)
    # 24 bodies: pelvis + 23 segments (no explicit "world" body)
    assert len(ir.bodies) == 24
    # 24 joints: 1 free_flyer (world→pelvis) + 23 spherical
    assert len(ir.joints) == 24
    # root_body is "pelvis" — world is only a sentinel, not a real body
    assert ir.root_body == "pelvis"


def test_smpl_custom_height():
    ir = make_smpl_like_body(height=1.8, mass=80.0)
    model = build_model(ir)
    assert model.nq == 99  # topology doesn't change with height


def test_smpl_q_neutral_shape(smpl_model):
    assert smpl_model.q_neutral.shape == (smpl_model.nq,)
