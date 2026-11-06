"""SE(3) group operations of ``better_robot.lie.se3`` on seeded random poses and hand-built cases."""

import math

import pytest
import torch

from better_robot.lie import se3

SEEDS = [0, 1, 2]


def _random_poses(seed: int, batch: int = 8, scale: float = 1.0) -> torch.Tensor:
    """``(batch, 7)`` float64 poses ``exp(xi)`` with ``xi`` drawn from ``N(0, scale^2)``."""
    generator = torch.Generator().manual_seed(seed)
    return se3.exp(scale * torch.randn(batch, 6, generator=generator, dtype=torch.float64))


def test_identity_batch():
    T = se3.identity(batch_shape=(3, 2))
    assert T.shape == (3, 2, 7)
    assert torch.equal(T[..., 6], torch.ones(3, 2))


@pytest.mark.parametrize("seed", SEEDS)
def test_compose_with_inverse_is_identity(seed):
    """T * T^{-1} is the identity, up to the quaternion sign."""
    T = _random_poses(seed)
    result = se3.compose(T, se3.inverse(T))
    identity = se3.identity(batch_shape=(8,), dtype=torch.float64)
    torch.testing.assert_close(result[..., :3], identity[..., :3], atol=1e-12, rtol=0.0)
    torch.testing.assert_close(result[..., 3:].abs(), identity[..., 3:], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_exp_log_round_trip(seed):
    """log(exp(xi)) == xi for rotation angles below pi."""
    generator = torch.Generator().manual_seed(seed)
    xi = 0.5 * torch.randn(8, 6, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(se3.log(se3.exp(xi)), xi, atol=1e-12, rtol=0.0)


def test_act_translation():
    """A pure translation shifts the point."""
    T = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    result = se3.act(T, torch.zeros(3))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("seed", SEEDS)
def test_adjoint_inverse_inverts_adjoint(seed):
    """Ad(T) @ Ad(T)^-1 == I, and Ad(T)^-1 == Ad(T^-1)."""
    T = _random_poses(seed)
    adjoint_inverse = se3.adjoint_inverse(T)
    eye = torch.eye(6, dtype=torch.float64).expand(8, 6, 6)
    torch.testing.assert_close(se3.adjoint(T) @ adjoint_inverse, eye, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(adjoint_inverse, se3.adjoint(se3.inverse(T)), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_adjoint_maps_twists_through_conjugation(seed):
    """Ad(T) @ xi == log(T * exp(xi) * T^-1) for small xi."""
    T = _random_poses(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    xi = 0.1 * torch.randn(8, 6, generator=generator, dtype=torch.float64)
    conjugated = se3.log(se3.compose(se3.compose(T, se3.exp(xi)), se3.inverse(T)))
    torch.testing.assert_close((se3.adjoint(T) @ xi.unsqueeze(-1)).squeeze(-1), conjugated, atol=1e-10, rtol=0.0)


def test_from_axis_angle_pure_rotation():
    """A rotation about Z by pi/2 maps (1, 0, 0) to (0, 1, 0)."""
    T = se3.from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    result = se3.act(T, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0.0)


def test_from_translation():
    """A pure translation by the given vector."""
    T = se3.from_translation(torch.tensor([2.5, 0.0, 0.0]))
    result = se3.act(T, torch.zeros(3))
    torch.testing.assert_close(result, torch.tensor([2.5, 0.0, 0.0]))


def test_normalize():
    """normalize projects the quaternion part onto the unit sphere."""
    T = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    torch.testing.assert_close(se3.normalize(T)[3:].norm(), torch.tensor(1.0))
