"""Tests for data_model/model.py and data_model/data.py."""

import torch

from better_robot.data_model.model import Model
from better_robot.data_model.data import Data
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder


def _make_simple_model() -> Model:
    """Minimal 3-joint model: universe → fixed → RZ → RZ."""
    builder = ModelBuilder("test")
    base = builder.add_body("base")
    link1 = builder.add_body("j1")
    link2 = builder.add_body("j2")
    builder.add_revolute_z("j1", parent=base, child=link1, lower=-3.14, upper=3.14)
    builder.add_revolute_z("j2", parent=link1, child=link2, lower=-3.14, upper=3.14)
    return build_model(builder.finalize())


def test_model_joint_id():
    model = _make_simple_model()
    assert model.joint_id("j1") == 2
    assert model.joint_id("j2") == 3


def test_model_body_id():
    model = _make_simple_model()
    assert model.body_id("base") == 1


def test_model_create_data():
    model = _make_simple_model()
    data = model.create_data()
    assert data.q.shape == (2,)


def test_model_create_data_batch():
    model = _make_simple_model()
    data = model.create_data(batch_shape=(4,))
    assert data.q.shape == (4, 2)


def test_model_integrate_difference():
    model = _make_simple_model()
    q = torch.zeros(2)
    v = torch.tensor([0.5, -0.3])
    q_new = model.integrate(q, v)
    v_back = model.difference(q, q_new)
    assert torch.allclose(v, v_back, atol=1e-5)


def test_model_integrate_shape():
    model = _make_simple_model()
    q = torch.zeros(3, 2)  # batch of 3
    v = torch.randn(3, 2)
    q_new = model.integrate(q, v)
    assert q_new.shape == (3, 2)


def test_model_random_configuration():
    model = _make_simple_model()
    q = model.random_configuration()
    assert q.shape == (2,)
    # Should be within limits
    assert (q >= -3.14).all()
    assert (q <= 3.14).all()


def test_model_to_device():
    model = _make_simple_model()
    # Just test that .to() returns a new Model with the same dtype
    model2 = model.to(dtype=torch.float64)
    assert model2.joint_placements.dtype == torch.float64
    assert model2.q_neutral.dtype == torch.float64


def test_model_subtree():
    model = _make_simple_model()
    st = model.get_subtree(2)
    assert set(st) == {2, 3}


def test_model_support():
    model = _make_simple_model()
    sp = model.get_support(3)
    assert sp == (0, 1, 2, 3)


# ──────────────────────────── Data ──────────────────────────────────────


def test_data_batch_shape():
    d = Data(q=torch.zeros(3, 5))
    assert d.batch_shape == (3,)


def test_data_reset():
    d = Data(q=torch.zeros(5))
    d.joint_pose_world = torch.zeros(4, 7)
    d._kinematics_level = 2
    d.reset()
    assert d.joint_pose_world is None
    assert d._kinematics_level == 0
    # q should still be there
    assert d.q is not None


def test_data_clone():
    d = Data(q=torch.tensor([1.0, 2.0, 3.0]))
    d.com_position = torch.tensor([0.1, 0.2, 0.3])
    d_clone = d.clone()
    assert torch.allclose(d_clone.q, d.q)
    # Mutation of clone should not affect original
    d_clone.q[0] = 99.0
    assert d.q[0] != 99.0
