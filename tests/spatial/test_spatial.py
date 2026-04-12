"""Tests for spatial/ — Motion, Force, Inertia, Symmetric3, ops."""
import torch
import pytest
from better_robot.spatial.symmetric3 import Symmetric3
from better_robot.spatial.motion import Motion
from better_robot.spatial.force import Force
from better_robot.spatial.inertia import Inertia
from better_robot.spatial import ops


# ──────────────────────────── Symmetric3 ────────────────────────────────

def test_symmetric3_roundtrip():
    M = torch.randn(3, 3)
    M = (M + M.T) / 2.0  # make symmetric
    S = Symmetric3.from_matrix(M)
    M_back = S.to_matrix()
    assert torch.allclose(M_back, M, atol=1e-6)


def test_symmetric3_add():
    s1 = Symmetric3(torch.ones(6))
    s2 = Symmetric3(torch.ones(6) * 2.0)
    s3 = s1.add(s2)
    assert torch.allclose(s3.data, torch.ones(6) * 3.0)


# ──────────────────────────── Motion ────────────────────────────────────

def test_motion_zero():
    m = Motion.zero(batch_shape=(3,))
    assert m.data.shape == (3, 6)
    assert torch.all(m.data == 0)


def test_motion_linear_angular():
    data = torch.arange(6.).view(1, 6)
    m = Motion(data)
    assert torch.allclose(m.linear, data[..., :3])
    assert torch.allclose(m.angular, data[..., 3:])


def test_motion_add_sub():
    a = Motion(torch.ones(6))
    b = Motion(torch.ones(6) * 2.0)
    assert torch.allclose((a + b).data, torch.ones(6) * 3.0)
    assert torch.allclose((b - a).data, torch.ones(6))


def test_motion_cross_motion_antisymmetric():
    """m1 × m2 = -(m2 × m1)."""
    m1 = Motion(torch.randn(6))
    m2 = Motion(torch.randn(6))
    r1 = m1.cross_motion(m2)
    r2 = m2.cross_motion(m1)
    assert torch.allclose(r1.data, -r2.data, atol=1e-6)


def test_motion_se3_action_identity():
    """Ad(I) * v == v."""
    from better_robot.lie import se3
    T = se3.identity()
    m = Motion(torch.randn(6))
    result = m.se3_action(T)
    assert torch.allclose(result.data, m.data, atol=1e-5)


# ──────────────────────────── Force ─────────────────────────────────────

def test_force_zero():
    f = Force.zero()
    assert f.data.shape == (6,)
    assert torch.all(f.data == 0)


def test_force_add():
    a = Force(torch.ones(6))
    b = Force(torch.ones(6))
    assert torch.allclose((a + b).data, torch.ones(6) * 2.0)


def test_force_se3_action_identity():
    """Dual action with identity = identity."""
    from better_robot.lie import se3
    T = se3.identity()
    f = Force(torch.randn(6))
    result = f.se3_action(T)
    assert torch.allclose(result.data, f.data, atol=1e-5)


# ──────────────────────────── Inertia ────────────────────────────────────

def test_inertia_zero():
    I = Inertia.zero()
    assert I.data.shape == (10,)
    assert torch.all(I.data == 0)


def test_inertia_from_sphere():
    I = Inertia.from_sphere(mass=1.0, radius=0.5)
    expected_I = 0.4 * 1.0 * 0.5 ** 2
    assert torch.allclose(I.mass, torch.tensor(1.0))
    assert torch.allclose(I.inertia_matrix.diagonal(), torch.tensor([expected_I] * 3), atol=1e-5)


def test_inertia_add():
    I1 = Inertia.from_sphere(mass=1.0, radius=0.5)
    I2 = Inertia.from_sphere(mass=1.0, radius=0.5)
    I_sum = I1.add(I2)
    assert torch.allclose(I_sum.mass, torch.tensor(2.0))


def test_inertia_apply_returns_force():
    """I * v returns a Force."""
    I = Inertia.from_sphere(mass=1.0, radius=0.5)
    v = Motion(torch.ones(6))
    f = I.apply(v)
    assert isinstance(f, Force)
    assert f.data.shape == (6,)


def test_inertia_matrix_shape():
    I = Inertia.from_box(mass=2.0, size=torch.tensor([1.0, 2.0, 3.0]))
    M = I.inertia_matrix
    assert M.shape == (3, 3)
    # Should be symmetric
    assert torch.allclose(M, M.T, atol=1e-6)


# ──────────────────────────── ops ────────────────────────────────────────

def test_ops_ad_shape():
    m = Motion(torch.randn(6))
    A = ops.ad(m)
    assert A.shape == (6, 6)


def test_ops_cross_mm():
    m1 = Motion(torch.randn(6))
    m2 = Motion(torch.randn(6))
    r1 = ops.cross_mm(m1, m2)
    r2 = m1.cross_motion(m2)
    assert torch.allclose(r1.data, r2.data, atol=1e-7)


def test_ops_act_motion_identity():
    from better_robot.lie import se3
    T = se3.identity()
    m = Motion(torch.randn(6))
    result = ops.act_motion(T, m)
    assert torch.allclose(result.data, m.data, atol=1e-5)
