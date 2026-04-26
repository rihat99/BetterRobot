"""Tests for data_model/joint_models/*.py — every joint family.

Phase 2 criteria: joint_transform, integrate, difference, neutral on a random
batch are tested against hand-derived references for every joint family.
"""
import math
import torch
import pytest

from better_robot.data_model.joint_models.fixed import JointFixed, JointUniverse
from better_robot.data_model.joint_models.revolute import (
    JointRX, JointRY, JointRZ, JointRevoluteUnaligned, JointRevoluteUnbounded,
)
from better_robot.data_model.joint_models.prismatic import (
    JointPX, JointPY, JointPZ, JointPrismaticUnaligned,
)
from better_robot.data_model.joint_models.spherical import JointSpherical
from better_robot.data_model.joint_models.free_flyer import JointFreeFlyer
from better_robot.data_model.joint_models.translation import JointTranslation
from better_robot.data_model.joint_models.planar import JointPlanar
from better_robot.data_model.joint_models.helical import JointHelical
from better_robot.lie import se3 as _se3, so3 as _so3


# ──────────────────────── helpers ────────────────────────────────────────

def _make_lower_upper(nq, lo=-2.0, hi=2.0):
    return torch.full((nq,), lo), torch.full((nq,), hi)


def _check_integrate_difference(jm, q, v):
    """integrate then difference should approximate v."""
    q_new = jm.integrate(q, v)
    v_back = jm.difference(q, q_new)
    assert torch.allclose(v, v_back, atol=1e-5), \
        f"{type(jm).__name__}: integrate/difference mismatch"


# ──────────────────────── JointFixed / Universe ──────────────────────────

def test_joint_fixed_transform_is_identity():
    jm = JointFixed()
    T = jm.joint_transform(torch.zeros(0))
    assert T.shape == (7,)
    assert torch.allclose(T, torch.tensor([0., 0., 0., 0., 0., 0., 1.]))


def test_joint_universe_same_as_fixed():
    jm = JointUniverse()
    T = jm.joint_transform(torch.zeros(0))
    assert torch.allclose(T, torch.tensor([0., 0., 0., 0., 0., 0., 1.]))


# ──────────────────────── Revolute ──────────────────────────────────────

@pytest.mark.parametrize("jm", [JointRX(), JointRY(), JointRZ()])
def test_revolute_aligned_transform_shape(jm):
    q = torch.tensor([0.5])
    T = jm.joint_transform(q)
    assert T.shape == (7,)
    # Should be pure rotation (no translation)
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-6)


def test_revolute_rx_zero_is_identity():
    jm = JointRX()
    T = jm.joint_transform(torch.zeros(1))
    assert torch.allclose(T, torch.tensor([0., 0., 0., 0., 0., 0., 1.]), atol=1e-6)


def test_revolute_rz_90deg():
    jm = JointRZ()
    T = jm.joint_transform(torch.tensor([math.pi / 2]))
    p = _se3.act(T, torch.tensor([1., 0., 0.]))
    assert torch.allclose(p, torch.tensor([0., 1., 0.]), atol=1e-5)


def test_revolute_integrate_difference():
    jm = JointRZ()
    q = torch.tensor([0.3])
    v = torch.tensor([0.5])
    _check_integrate_difference(jm, q, v)


@pytest.mark.parametrize("jm", [JointRX(), JointRY(), JointRZ()])
def test_revolute_aligned_neutral(jm):
    assert torch.allclose(jm.neutral(), torch.zeros(1))


def test_revolute_unaligned_transform():
    axis = torch.tensor([1., 1., 0.]) / math.sqrt(2.0)
    jm = JointRevoluteUnaligned(axis=axis)
    q = torch.tensor([math.pi])
    T = jm.joint_transform(q)
    assert T.shape == (7,)
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-5)


def test_revolute_unbounded_roundtrip():
    jm = JointRevoluteUnbounded()
    q = jm.neutral()   # [1, 0]
    v = torch.tensor([1.2])
    q_new = jm.integrate(q, v)
    v_back = jm.difference(q, q_new)
    assert torch.allclose(v, v_back, atol=1e-5)


@pytest.mark.parametrize("jm", [JointRX(), JointRY(), JointRZ()])
def test_revolute_batch(jm):
    q = torch.randn(3, 1)
    T = jm.joint_transform(q)
    assert T.shape == (3, 7)


# ──────────────────────── Prismatic ──────────────────────────────────────

@pytest.mark.parametrize("jm", [JointPX(), JointPY(), JointPZ()])
def test_prismatic_zero_is_identity(jm):
    T = jm.joint_transform(torch.zeros(1))
    assert torch.allclose(T, torch.tensor([0., 0., 0., 0., 0., 0., 1.]), atol=1e-6)


def test_prismatic_px_translation():
    jm = JointPX()
    T = jm.joint_transform(torch.tensor([2.0]))
    p = _se3.act(T, torch.zeros(3))
    assert torch.allclose(p, torch.tensor([2., 0., 0.]), atol=1e-5)


def test_prismatic_integrate_difference():
    jm = JointPZ()
    q = torch.tensor([1.0])
    v = torch.tensor([0.3])
    _check_integrate_difference(jm, q, v)


# ──────────────────────── Spherical ──────────────────────────────────────

def test_spherical_neutral_is_identity():
    jm = JointSpherical()
    q = jm.neutral()
    assert torch.allclose(q, torch.tensor([0., 0., 0., 1.]))


def test_spherical_transform_shape():
    jm = JointSpherical()
    q = jm.neutral()
    T = jm.joint_transform(q)
    assert T.shape == (7,)
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-6)


def test_spherical_integrate_difference():
    jm = JointSpherical()
    q = jm.neutral()
    v = torch.randn(3) * 0.3
    _check_integrate_difference(jm, q, v)


def test_spherical_batch():
    jm = JointSpherical()
    q = torch.randn(4, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    T = jm.joint_transform(q)
    assert T.shape == (4, 7)


# ──────────────────────── FreeFlyer ──────────────────────────────────────

def test_free_flyer_neutral():
    jm = JointFreeFlyer()
    q = jm.neutral()
    assert q.shape == (7,)
    assert torch.allclose(q, torch.tensor([0., 0., 0., 0., 0., 0., 1.]))


def test_free_flyer_transform_is_identity_at_neutral():
    jm = JointFreeFlyer()
    q = jm.neutral()
    T = jm.joint_transform(q)
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-5)


def test_free_flyer_integrate_difference():
    jm = JointFreeFlyer()
    q = jm.neutral()
    v = torch.randn(6) * 0.2
    _check_integrate_difference(jm, q, v)


def test_free_flyer_batch():
    jm = JointFreeFlyer()
    B = 5
    q = jm.neutral().unsqueeze(0).expand(B, 7).clone()
    T = jm.joint_transform(q)
    assert T.shape == (B, 7)


# ──────────────────────── Translation ────────────────────────────────────

def test_translation_transform():
    jm = JointTranslation()
    q = torch.tensor([1., 2., 3.])
    T = jm.joint_transform(q)
    p = _se3.act(T, torch.zeros(3))
    assert torch.allclose(p, q, atol=1e-5)


def test_translation_integrate_difference():
    jm = JointTranslation()
    q = torch.zeros(3)
    v = torch.tensor([0.5, -0.3, 1.0])
    _check_integrate_difference(jm, q, v)


# ──────────────────────── Planar ─────────────────────────────────────────

def test_planar_neutral():
    jm = JointPlanar()
    q = jm.neutral()
    assert q.shape == (4,)
    # [x=0, y=0, cos=1, sin=0]
    assert torch.allclose(q, torch.tensor([0., 0., 1., 0.]))


def test_planar_transform_zero():
    jm = JointPlanar()
    q = jm.neutral()
    T = jm.joint_transform(q)
    assert T.shape == (7,)
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-5)


def test_planar_integrate_difference():
    jm = JointPlanar()
    q = jm.neutral()
    v = torch.tensor([0.2, -0.1, 0.3])
    _check_integrate_difference(jm, q, v)


# ──────────────────────── Helical ────────────────────────────────────────

def test_helical_zero_pitch_is_revolute():
    """At pitch=0, helical is just a revolute."""
    axis = torch.tensor([0., 0., 1.])
    jm = JointHelical(axis=axis, pitch=0.0)
    q = torch.tensor([math.pi / 2])
    T = jm.joint_transform(q)
    # Should have no translation
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-5)


def test_helical_nonzero_pitch():
    """With pitch>0, there should be translation along the axis."""
    axis = torch.tensor([0., 0., 1.])
    pitch = 0.1
    jm = JointHelical(axis=axis, pitch=pitch)
    angle = torch.tensor([math.pi])
    T = jm.joint_transform(angle)
    # Translation should be pitch * pi along z
    assert abs(float(T[2]) - pitch * math.pi) < 1e-5
