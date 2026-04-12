"""Tests for data_model/model.py and data_model/data.py."""
import torch
import pytest
import dataclasses

from better_robot.data_model.model import Model
from better_robot.data_model.data import Data
from better_robot.data_model.frame import Frame
from better_robot.data_model.joint_models.fixed import JointUniverse, JointFixed
from better_robot.data_model.joint_models.revolute import JointRZ
from better_robot.data_model.topology import (
    topo_sort, build_children, build_subtrees, build_supports,
)


def _make_simple_model() -> Model:
    """Minimal 3-joint model: universe → fixed → RZ → RZ."""
    parents = (-1, 0, 1, 2)
    joint_models = (JointUniverse(), JointFixed(), JointRZ(), JointRZ())
    nqs = (0, 0, 1, 1)
    nvs = (0, 0, 1, 1)

    # idx_qs: cumulative sum of nqs
    idx_qs = (0, 0, 0, 1)
    idx_vs = (0, 0, 0, 1)
    nq = 2
    nv = 2

    topo = topo_sort(parents)
    children = build_children(parents)
    subtrees = build_subtrees(parents)
    supports = build_supports(parents)

    n = len(parents)
    return Model(
        njoints=n,
        nbodies=n,
        nframes=0,
        nq=nq,
        nv=nv,
        name="test",
        joint_names=("universe", "base", "j1", "j2"),
        body_names=("universe", "base", "j1", "j2"),
        frame_names=(),
        joint_name_to_id={"universe": 0, "base": 1, "j1": 2, "j2": 3},
        body_name_to_id={"universe": 0, "base": 1, "j1": 2, "j2": 3},
        frame_name_to_id={},
        parents=parents,
        children=children,
        subtrees=subtrees,
        supports=supports,
        topo_order=topo,
        joint_models=joint_models,
        nqs=nqs,
        nvs=nvs,
        idx_qs=idx_qs,
        idx_vs=idx_vs,
        joint_placements=torch.zeros(n, 7),
        body_inertias=torch.zeros(n, 10),
        lower_pos_limit=torch.full((nq,), -3.14),
        upper_pos_limit=torch.full((nq,), 3.14),
        velocity_limit=torch.full((nv,), 10.0),
        effort_limit=torch.full((nv,), 50.0),
        rotor_inertia=torch.zeros(nv),
        armature=torch.zeros(nv),
        friction=torch.zeros(nv),
        damping=torch.zeros(nv),
        gravity=torch.zeros(6),
        mimic_multiplier=torch.ones(n),
        mimic_offset=torch.zeros(n),
        mimic_source=tuple(range(n)),
        frames=(),
        q_neutral=torch.zeros(nq),
    )


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
    assert data._model_id == id(model)


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
    assert (q <=  3.14).all()


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
    d = Data(_model_id=0, q=torch.zeros(3, 5))
    assert d.batch_shape == (3,)


def test_data_reset():
    d = Data(_model_id=0, q=torch.zeros(5))
    d.oMi = torch.zeros(4, 7)
    d._kinematics_level = 2
    d.reset()
    assert d.oMi is None
    assert d._kinematics_level == 0
    # q should still be there
    assert d.q is not None


def test_data_clone():
    d = Data(_model_id=42, q=torch.tensor([1., 2., 3.]))
    d.com = torch.tensor([0.1, 0.2, 0.3])
    d_clone = d.clone()
    assert d_clone._model_id == 42
    assert torch.allclose(d_clone.q, d.q)
    # Mutation of clone should not affect original
    d_clone.q[0] = 99.0
    assert d.q[0] != 99.0
