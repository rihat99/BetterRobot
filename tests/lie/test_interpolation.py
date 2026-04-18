"""Tests for SLERP (SO3) and ScLERP (SE3) in lie/so3.py and lie/se3.py."""
import math
import pytest
import torch
from better_robot.lie import se3, so3


# ────────────────────────────── SLERP (SO3) ──────────────────────────────


def test_slerp_endpoints():
    """t=0 → q1 exactly; t=1 → ±q2."""
    q1 = so3.exp(torch.tensor([0.3, -0.4, 0.5]))
    q2 = so3.exp(torch.tensor([-0.2, 0.6, 0.1]))
    out0 = so3.slerp(q1, q2, 0.0)
    out1 = so3.slerp(q1, q2, 1.0)
    assert torch.allclose(out0, q1, atol=1e-6)
    assert torch.allclose(out1.abs(), q2.abs(), atol=1e-6)


def test_slerp_midpoint_axis_angle():
    """SLERP at t=0.5 on a Z-axis π/2 rotation produces a π/4 rotation."""
    axis = torch.tensor([0., 0., 1.])
    q1 = so3.identity()
    q2 = so3.from_axis_angle(axis, torch.tensor(math.pi / 2))
    mid = so3.slerp(q1, q2, 0.5)
    expected = so3.from_axis_angle(axis, torch.tensor(math.pi / 4))
    assert torch.allclose(mid.abs(), expected.abs(), atol=1e-6)


def test_slerp_antipodal():
    """dot(q1,q2)<0 → takes the shortest arc."""
    q1 = so3.identity()
    # A small rotation whose quaternion we deliberately negate.
    q2_short = so3.from_axis_angle(torch.tensor([0., 0., 1.]), torch.tensor(0.1))
    q2 = -q2_short  # same rotation, antipodal quaternion
    out = so3.slerp(q1, q2, 0.1)
    expected = so3.from_axis_angle(torch.tensor([0., 0., 1.]), torch.tensor(0.01))
    assert torch.allclose(out.abs(), expected.abs(), atol=1e-5)


def test_slerp_near_identical():
    """q2 = q1 + tiny noise → LERP fallback, no NaN, stays near q1."""
    torch.manual_seed(0)
    q1 = so3.normalize(torch.randn(4))
    q2 = so3.normalize(q1 + 1e-8 * torch.randn(4))
    out = so3.slerp(q1, q2, 0.5)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.abs(), q1.abs(), atol=1e-5)


def test_slerp_identical():
    """q1 == q2 → returns q1 for all t (no 0/0)."""
    q = so3.exp(torch.tensor([0.2, -0.3, 0.4]))
    for t in [0.0, 0.3, 0.7, 1.0]:
        out = so3.slerp(q, q, t)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, q, atol=1e-6)


def test_slerp_constant_speed():
    """‖log(q1⁻¹ · slerp(t))‖ is linear in t."""
    q1 = so3.exp(torch.tensor([0.1, 0.2, -0.3]))
    q2 = so3.exp(torch.tensor([0.6, -0.4, 0.2]))
    ts = torch.linspace(0.0, 1.0, 5)
    dists = []
    for t in ts:
        q_t = so3.slerp(q1, q2, t.item())
        dists.append(so3.log(so3.compose(so3.inverse(q1), q_t)).norm().item())
    total = dists[-1]
    for i, t in enumerate(ts):
        assert abs(dists[i] - t.item() * total) < 1e-5


def test_slerp_unit_norm():
    """Output is unit-norm."""
    torch.manual_seed(1)
    q1 = so3.normalize(torch.randn(4))
    q2 = so3.normalize(torch.randn(4))
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out = so3.slerp(q1, q2, t)
        assert torch.allclose(out.norm(), torch.tensor(1.0), atol=1e-6)


def test_slerp_batched():
    """Batched inputs (5, 4) with scalar t → (5, 4)."""
    torch.manual_seed(2)
    q1 = so3.exp(torch.randn(5, 3) * 0.3)
    q2 = so3.exp(torch.randn(5, 3) * 0.3)
    out = so3.slerp(q1, q2, 0.3)
    assert out.shape == (5, 4)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_slerp_t_broadcast():
    """q1,q2 shared; t as time grid produces a trajectory."""
    q1 = so3.identity().expand(10, 4).clone()
    q2 = so3.from_axis_angle(
        torch.tensor([0., 0., 1.]), torch.tensor(math.pi / 2)
    ).expand(10, 4).clone()
    ts = torch.linspace(0.0, 1.0, 10)
    out = so3.slerp(q1, q2, ts)
    assert out.shape == (10, 4)
    # First frame ≈ identity, last frame ≈ ±q2.
    assert torch.allclose(out[0], so3.identity(), atol=1e-6)
    assert torch.allclose(out[-1].abs(), q2[-1].abs(), atol=1e-5)


def test_slerp_gradient_finite():
    """Autograd through slerp at endpoints gives finite grads."""
    q1 = so3.exp(torch.tensor([0.1, 0.2, 0.3])).requires_grad_(True)
    q2 = so3.exp(torch.tensor([-0.2, 0.3, 0.1])).requires_grad_(True)
    for t in [0.0, 0.5, 1.0]:
        out = so3.slerp(q1, q2, t)
        (g1,) = torch.autograd.grad(out.sum(), q1, retain_graph=True)
        assert torch.isfinite(g1).all(), f"NaN/Inf grad at t={t}"


# ────────────────────────────── ScLERP (SE3) ─────────────────────────────


def test_sclerp_endpoints():
    """t=0 → T1; t=1 → T2 (±q on quaternion part)."""
    T1 = se3.exp(torch.tensor([0.1, 0.2, -0.3, 0.2, -0.1, 0.3]))
    T2 = se3.exp(torch.tensor([-0.2, 0.3, 0.5, -0.4, 0.2, 0.1]))
    out0 = se3.sclerp(T1, T2, 0.0)
    out1 = se3.sclerp(T1, T2, 1.0)
    assert torch.allclose(out0[:3], T1[:3], atol=1e-5)
    assert torch.allclose(out0[3:].abs(), T1[3:].abs(), atol=1e-5)
    assert torch.allclose(out1[:3], T2[:3], atol=1e-5)
    assert torch.allclose(out1[3:].abs(), T2[3:].abs(), atol=1e-5)


def test_sclerp_pure_translation():
    """Identity rotation + translation diff → LERP of translation."""
    T1 = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    T2 = torch.tensor([2., 0., 0., 0., 0., 0., 1.])
    out = se3.sclerp(T1, T2, 0.5)
    assert torch.allclose(out[:3], torch.tensor([1., 0., 0.]), atol=1e-5)
    assert torch.allclose(out[3:], torch.tensor([0., 0., 0., 1.]), atol=1e-5)


def test_sclerp_pure_rotation():
    """Zero translation + rotation diff → translation stays 0, rotation matches so3.slerp."""
    axis = torch.tensor([0., 0., 1.])
    T1 = se3.identity()
    q2 = so3.from_axis_angle(axis, torch.tensor(math.pi / 2))
    T2 = torch.cat([torch.zeros(3), q2])
    out = se3.sclerp(T1, T2, 0.5)
    expected_q = so3.from_axis_angle(axis, torch.tensor(math.pi / 4))
    assert torch.allclose(out[:3], torch.zeros(3), atol=1e-5)
    assert torch.allclose(out[3:].abs(), expected_q.abs(), atol=1e-5)


def test_sclerp_constant_speed():
    """‖log(T1⁻¹ · sclerp(t))‖ is linear in t."""
    T1 = se3.exp(torch.tensor([0.2, 0.1, -0.3, 0.1, 0.2, -0.1]))
    T2 = se3.exp(torch.tensor([0.5, -0.4, 0.2, 0.3, -0.2, 0.4]))
    ts = torch.linspace(0.0, 1.0, 5)
    dists = []
    for t in ts:
        T_t = se3.sclerp(T1, T2, t.item())
        dists.append(se3.log(se3.compose(se3.inverse(T1), T_t)).norm().item())
    total = dists[-1]
    for i, t in enumerate(ts):
        assert abs(dists[i] - t.item() * total) < 1e-5


def test_sclerp_batched():
    """Batched (4, 7) inputs with scalar t → (4, 7)."""
    torch.manual_seed(3)
    T1 = se3.exp(torch.randn(4, 6) * 0.3)
    T2 = se3.exp(torch.randn(4, 6) * 0.3)
    out = se3.sclerp(T1, T2, 0.4)
    assert out.shape == (4, 7)
    # Quaternion part stays unit-norm.
    assert torch.allclose(out[..., 3:].norm(dim=-1), torch.ones(4), atol=1e-5)


def test_sclerp_identity_path():
    """T1 == T2 → returns T1 for all t."""
    T = se3.exp(torch.tensor([0.3, -0.2, 0.1, 0.2, 0.1, -0.3]))
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out = se3.sclerp(T, T, t)
        assert torch.isfinite(out).all()
        assert torch.allclose(out[:3], T[:3], atol=1e-5)
        assert torch.allclose(out[3:].abs(), T[3:].abs(), atol=1e-5)


def test_sclerp_vs_decoupled():
    """ScLERP ≠ (SLERP(rot) ‖ LERP(trans)) in general — coupling is non-trivial."""
    T1 = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    axis = torch.tensor([0., 0., 1.])
    q2 = so3.from_axis_angle(axis, torch.tensor(math.pi))  # 180° about Z
    T2 = torch.cat([torch.tensor([2., 0., 0.]), q2])

    t = 0.5
    out = se3.sclerp(T1, T2, t)

    # Decoupled version: slerp rotation, lerp translation.
    q_dec = so3.slerp(T1[3:], T2[3:], t)
    p_dec = (1 - t) * T1[:3] + t * T2[:3]
    decoupled = torch.cat([p_dec, q_dec])

    # Translations should differ — screw coupling curves the straight-line path.
    assert not torch.allclose(out[:3], decoupled[:3], atol=1e-3)
