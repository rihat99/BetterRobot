"""Tests for data_model/joint_models/*.py — every joint family.

Phase 2 criteria: joint_placement, integrate, difference, neutral on a random
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
    JointPX, JointPY, JointPZ,
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

_IDENTITY_POSE = [0., 0., 0., 0., 0., 0., 1.]


@pytest.mark.parametrize(
    "make_value, expected, atol",
    [
        pytest.param(lambda: JointFixed().joint_placement(torch.zeros(0)), _IDENTITY_POSE, 1e-8, id="fixed_transform"),
        pytest.param(
            lambda: JointUniverse().joint_placement(torch.zeros(0)), _IDENTITY_POSE, 1e-8, id="universe_transform"
        ),
        pytest.param(lambda: JointRX().joint_placement(torch.zeros(1)), _IDENTITY_POSE, 1e-6, id="rx_zero_transform"),
        pytest.param(lambda: JointPX().joint_placement(torch.zeros(1)), _IDENTITY_POSE, 1e-6, id="px_zero_transform"),
        pytest.param(lambda: JointPY().joint_placement(torch.zeros(1)), _IDENTITY_POSE, 1e-6, id="py_zero_transform"),
        pytest.param(lambda: JointPZ().joint_placement(torch.zeros(1)), _IDENTITY_POSE, 1e-6, id="pz_zero_transform"),
        pytest.param(lambda: JointRX().neutral(), [0.], 1e-8, id="rx_neutral"),
        pytest.param(lambda: JointRY().neutral(), [0.], 1e-8, id="ry_neutral"),
        pytest.param(lambda: JointRZ().neutral(), [0.], 1e-8, id="rz_neutral"),
        pytest.param(lambda: JointSpherical().neutral(), [0., 0., 0., 1.], 1e-8, id="spherical_neutral"),
        pytest.param(lambda: JointFreeFlyer().neutral(), _IDENTITY_POSE, 1e-8, id="free_flyer_neutral"),
        # Was a translation-only check (T[:3] == 0, atol 1e-5); the full identity pose is stricter.
        pytest.param(
            lambda: JointFreeFlyer().joint_placement(JointFreeFlyer().neutral()),
            _IDENTITY_POSE,
            1e-5,
            id="free_flyer_transform_at_neutral",
        ),
        # [x=0, y=0, cos=1, sin=0]
        pytest.param(lambda: JointPlanar().neutral(), [0., 0., 1., 0.], 1e-8, id="planar_neutral"),
        pytest.param(
            lambda: JointPlanar().joint_placement(JointPlanar().neutral()),
            _IDENTITY_POSE,
            1e-5,
            id="planar_transform_at_neutral",
        ),
    ],
)
def test_identity_and_neutral(make_value, expected, atol):
    value = make_value()
    expected = torch.tensor(expected)
    assert value.shape == expected.shape
    assert torch.allclose(value, expected, atol=atol)


# ──────────────────────── Revolute ──────────────────────────────────────


def test_revolute_rz_90deg():
    jm = JointRZ()
    T = jm.joint_placement(torch.tensor([math.pi / 2]))
    p = _se3.act(T, torch.tensor([1., 0., 0.]))
    assert torch.allclose(p, torch.tensor([0., 1., 0.]), atol=1e-5)


def test_revolute_integrate_difference():
    jm = JointRZ()
    q = torch.tensor([0.3])
    v = torch.tensor([0.5])
    _check_integrate_difference(jm, q, v)


def test_revolute_unaligned_transform():
    axis = torch.tensor([1., 1., 0.]) / math.sqrt(2.0)
    jm = JointRevoluteUnaligned(axis=axis)
    q = torch.tensor([math.pi])
    T = jm.joint_placement(q)
    assert T.shape == (7,)
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-5)


def test_revolute_unbounded_roundtrip():
    jm = JointRevoluteUnbounded()
    q = jm.neutral()   # [1, 0]
    v = torch.tensor([1.2])
    q_new = jm.integrate(q, v)
    v_back = jm.difference(q, q_new)
    assert torch.allclose(v, v_back, atol=1e-5)


# ──────────────────────── Prismatic ──────────────────────────────────────

def test_prismatic_px_translation():
    jm = JointPX()
    T = jm.joint_placement(torch.tensor([2.0]))
    p = _se3.act(T, torch.zeros(3))
    assert torch.allclose(p, torch.tensor([2., 0., 0.]), atol=1e-5)


def test_prismatic_integrate_difference():
    jm = JointPZ()
    q = torch.tensor([1.0])
    v = torch.tensor([0.3])
    _check_integrate_difference(jm, q, v)


# ──────────────────────── Spherical ──────────────────────────────────────

def test_spherical_integrate_difference():
    jm = JointSpherical()
    q = jm.neutral()
    v = torch.randn(3) * 0.3
    _check_integrate_difference(jm, q, v)


@pytest.mark.parametrize("theta", [0.0, 1e-9], ids=["identity", "theta_1e-9"])
def test_spherical_identical_difference_has_finite_gradient(theta):
    """An identical-quaternion difference evaluates Log at SO3 identity."""
    jm = JointSpherical()
    omega = torch.tensor([theta, 0.0, 0.0], dtype=torch.float64)
    q = _so3.exp(omega).detach().requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda value: jm.difference(value, value),
        (q,),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.autograd.gradgradcheck(
        lambda value: jm.difference(value, value),
        (q,),
        atol=1e-6,
        rtol=1e-5,
    )

    loss = jm.difference(q, q).square().sum()
    loss.backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))


# ──────────────────────── FreeFlyer ──────────────────────────────────────

def test_free_flyer_integrate_difference():
    jm = JointFreeFlyer()
    q = jm.neutral()
    v = torch.randn(6) * 0.2
    _check_integrate_difference(jm, q, v)


# ──────────────────────── Translation ────────────────────────────────────

def test_translation_transform():
    jm = JointTranslation()
    q = torch.tensor([1., 2., 3.])
    T = jm.joint_placement(q)
    p = _se3.act(T, torch.zeros(3))
    assert torch.allclose(p, q, atol=1e-5)


def test_translation_integrate_difference():
    jm = JointTranslation()
    q = torch.zeros(3)
    v = torch.tensor([0.5, -0.3, 1.0])
    _check_integrate_difference(jm, q, v)


# ──────────────────────── Planar ─────────────────────────────────────────

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
    T = jm.joint_placement(q)
    # Should have no translation
    assert torch.allclose(T[:3], torch.zeros(3), atol=1e-5)


def test_helical_nonzero_pitch():
    """With pitch>0, there should be translation along the axis."""
    axis = torch.tensor([0., 0., 1.])
    pitch = 0.1
    jm = JointHelical(axis=axis, pitch=pitch)
    angle = torch.tensor([math.pi])
    T = jm.joint_placement(angle)
    # Translation should be pitch * pi along z
    assert abs(float(T[2]) - pitch * math.pi) < 1e-5
