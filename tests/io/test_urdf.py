"""Tests for the URDF parser and ``br.load()`` with URDF files.

Phase 3 pass criteria:
 - ``br.load("panda.urdf")`` returns a ``Model`` equivalent to the current
   ``RobotModel`` (joint count, body count, limits).
 - ``br.load("g1.urdf", free_flyer=True)`` returns a ``Model`` whose
   ``joint_models[1]`` is ``JointFreeFlyer`` and whose ``nq`` is 7 + 29 = 36.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.io import load
from better_robot.io.parsers.urdf import parse_urdf


# ── helpers ──────────────────────────────────────────────────────────────────


def _get_panda_urdf():
    """Load the panda URDF path via robot_descriptions."""
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description

    return panda_description.URDF_PATH


def _get_g1_urdf():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description

    return getattr(g1_description, "URDF_PATH", None) or pytest.skip("g1 URDF not available")


# ── parse_urdf unit tests ────────────────────────────────────────────────────


def test_parse_urdf_root_body_has_no_incoming_joint():
    path = _get_panda_urdf()
    ir = parse_urdf(path)
    child_bodies = {j.child_body for j in ir.joints}
    assert ir.root_body not in child_bodies


# ── build_model integration tests ────────────────────────────────────────────


@pytest.fixture(scope="module")
def panda_model():
    path = _get_panda_urdf()
    return load(path)


@pytest.fixture(scope="module")
def panda_ir():
    path = _get_panda_urdf()
    return parse_urdf(path)


def test_panda_nq(panda_model):
    """Panda exposes 7 arm + 1 source finger coordinate."""
    assert panda_model.nq == 8
    assert panda_model.nq_full == 9


def test_panda_identity_mimic_is_accepted_with_metadata(panda_model, panda_ir):
    mimic_joints = [joint for joint in panda_ir.joints if joint.mimic_source is not None]
    assert mimic_joints

    for joint in mimic_joints:
        assert joint.mimic_multiplier == 1.0
        assert joint.mimic_offset == 0.0
        joint_id = panda_model.joint_id(joint.name)
        source_id = panda_model.joint_id(joint.mimic_source)
        assert panda_model.mimic_source[joint_id] == source_id
        assert panda_model.mimic_multiplier[joint_id].item() == pytest.approx(1.0)
        assert panda_model.mimic_offset[joint_id].item() == pytest.approx(0.0)
        assert panda_model.nqs[joint_id] == 0
        assert panda_model.nvs[joint_id] == 0
        assert panda_model.nqs_full[joint_id] == 1
        assert panda_model.nvs_full[joint_id] == 1


def test_panda_limits_finite_for_actuated(panda_model):
    # All 7 actuated revolute joints should have finite limits
    assert torch.all(torch.isfinite(panda_model.lower_pos_limit))
    assert torch.all(torch.isfinite(panda_model.upper_pos_limit))


def test_panda_joint_kinds(panda_model):
    """Panda joints should only be revolute, prismatic, or fixed."""
    for j in range(2, panda_model.njoints):
        kind = panda_model.joint_models[j].kind
        assert kind.startswith("revolute") or kind.startswith("prismatic") or kind == "fixed", (
            f"Unexpected joint kind {kind!r} at index {j}"
        )


def test_panda_name_to_id_has_all_joints(panda_model, panda_ir):
    for j in panda_ir.joints:
        assert j.name in panda_model.joint_name_to_id


def test_panda_body_frame_names(panda_model, panda_ir):
    for body in panda_ir.bodies:
        assert f"body_{body.name}" in panda_model.frame_name_to_id


def test_load_accepts_yourdfpy_object():
    pytest.importorskip("yourdfpy")
    import yourdfpy

    path = _get_panda_urdf()
    urdf = yourdfpy.URDF.load(path)
    model = load(urdf)
    assert model.nq == 8
    assert model.nq_full == 9


# ── G1 floating-base tests ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def g1_model():
    path = _get_g1_urdf()
    return load(path, free_flyer=True)


def test_g1_root_joint_is_free_flyer(g1_model):
    assert g1_model.joint_models[1].kind == "free_flyer"


@pytest.mark.parametrize(
    "attr, expected",
    [
        pytest.param("nq", 36, id="nq"),  # 29 actuated joints + 7 free-flyer DOFs
        pytest.param("nv", 35, id="nv"),  # 29 actuated + 6 free-flyer velocity DOFs
    ],
)
def test_g1_dims(g1_model, attr, expected):
    assert getattr(g1_model, attr) == expected


def test_load_dtype_and_device_kwargs():
    model = load(_get_panda_urdf(), dtype=torch.float64, device=torch.device("cpu"))
    assert model.values.joint_placements.dtype == torch.float64
    assert model.values.joint_placements.device.type == "cpu"


def test_inertia_is_rotated_from_inertial_frame_into_link_frame(tmp_path):
    """An ``<inertial><origin rpy=...>`` rotates the inertia tensor (URDF spec, Pinocchio)."""
    pin = pytest.importorskip("pinocchio")
    urdf = tmp_path / "r.urdf"
    urdf.write_text(
        '<robot name="r"><link name="base"/>'
        '<link name="l1"><inertial><origin xyz="0.1 0.2 0.3" rpy="0.3 -0.5 1.1"/><mass value="2"/>'
        '<inertia ixx="1" ixy="0.1" ixz="0" iyy="2" iyz="0" izz="3"/></inertial></link>'
        '<joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    body = next(b for b in parse_urdf(urdf).bodies if b.name == "l1")
    expected = torch.tensor(pin.buildModelFromUrdf(str(urdf)).inertias[1].inertia)
    torch.testing.assert_close(body.inertia.to(torch.float64), expected)
