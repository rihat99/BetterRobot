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

import better_robot as br_v2
from better_robot.io import load
from better_robot.io.parsers.urdf import parse_urdf
from better_robot.io.build_model import build_model
from better_robot.data_model.joint_models import JointFreeFlyer


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

def test_parse_urdf_returns_irmodel():
    path = _get_panda_urdf()
    from better_robot.io.ir import IRModel
    ir = parse_urdf(path)
    assert isinstance(ir, IRModel)


def test_parse_urdf_has_bodies_and_joints():
    path = _get_panda_urdf()
    ir = parse_urdf(path)
    assert len(ir.bodies) > 0
    assert len(ir.joints) > 0
    assert ir.root_body != ""


def test_parse_urdf_root_body_has_no_incoming_joint():
    path = _get_panda_urdf()
    ir = parse_urdf(path)
    child_bodies = {j.child_body for j in ir.joints}
    assert ir.root_body not in child_bodies


def test_parse_urdf_accepts_yourdfpy_object():
    pytest.importorskip("yourdfpy")
    import yourdfpy
    path = _get_panda_urdf()
    urdf = yourdfpy.URDF.load(path)
    ir = parse_urdf(urdf)
    assert len(ir.joints) > 0


# ── build_model integration tests ────────────────────────────────────────────

@pytest.fixture(scope="module")
def panda_model():
    path = _get_panda_urdf()
    return load(path)


@pytest.fixture(scope="module")
def panda_ir():
    path = _get_panda_urdf()
    return parse_urdf(path)


def test_panda_model_is_frozen(panda_model):
    import dataclasses
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        panda_model.nq = 99  # type: ignore[misc]


def test_panda_nq(panda_model):
    """Panda has 7 revolute + 2 prismatic finger joints → nq=9."""
    assert panda_model.nq == 9


def test_panda_njoints(panda_model, panda_ir):
    # njoints = 2 (universe + root_joint) + len(ir.joints)
    assert panda_model.njoints == 2 + len(panda_ir.joints)


def test_panda_nbodies_equals_njoints(panda_model):
    assert panda_model.nbodies == panda_model.njoints


def test_panda_joint_placements_shape(panda_model):
    assert panda_model.joint_placements.shape == (panda_model.njoints, 7)


def test_panda_body_inertias_shape(panda_model):
    assert panda_model.body_inertias.shape == (panda_model.nbodies, 10)


def test_panda_limits_finite_for_actuated(panda_model):
    # All 7 actuated revolute joints should have finite limits
    assert torch.all(torch.isfinite(panda_model.lower_pos_limit))
    assert torch.all(torch.isfinite(panda_model.upper_pos_limit))


def test_panda_root_joint_is_fixed(panda_model):
    assert panda_model.joint_models[1].kind == "fixed"


def test_panda_joint_kinds(panda_model):
    """Panda joints should only be revolute, prismatic, or fixed."""
    for j in range(2, panda_model.njoints):
        kind = panda_model.joint_models[j].kind
        assert (kind.startswith("revolute") or kind.startswith("prismatic")
                or kind == "fixed"), \
            f"Unexpected joint kind {kind!r} at index {j}"


def test_panda_name_to_id_has_all_joints(panda_model, panda_ir):
    for j in panda_ir.joints:
        assert j.name in panda_model.joint_name_to_id


def test_panda_topology_parents_before_children(panda_model):
    order = panda_model.topo_order
    for pos, j in enumerate(order):
        p = panda_model.parents[j]
        if p >= 0:
            assert list(order).index(p) < pos


def test_panda_topo_order_starts_at_universe(panda_model):
    assert panda_model.topo_order[0] == 0


def test_panda_body_frame_names(panda_model, panda_ir):
    for body in panda_ir.bodies:
        assert f"body_{body.name}" in panda_model.frame_name_to_id


def test_load_accepts_yourdfpy_object():
    pytest.importorskip("yourdfpy")
    import yourdfpy
    path = _get_panda_urdf()
    urdf = yourdfpy.URDF.load(path)
    model = load(urdf)
    assert model.nq == 9


# ── G1 floating-base tests ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def g1_model():
    path = _get_g1_urdf()
    return load(path, free_flyer=True)


def test_g1_root_joint_is_free_flyer(g1_model):
    assert g1_model.joint_models[1].kind == "free_flyer"


def test_g1_nq(g1_model):
    # G1 has 29 actuated joints + 7 free-flyer DOFs = 36
    assert g1_model.nq == 36


def test_g1_nv(g1_model):
    # G1: 29 actuated + 6 free-flyer velocity DOFs = 35
    assert g1_model.nv == 35
