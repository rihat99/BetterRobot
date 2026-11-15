"""SO(3) group operations of ``better_robot.lie.so3`` on seeded random rotations and hand-built cases."""

import itertools
import math

import pytest
import torch

from better_robot.lie import so3

SEEDS = [0, 1, 2]


def _random_quaternions(seed: int, batch: int = 8) -> torch.Tensor:
    """``(batch, 4)`` float64 unit quaternions, normalized Gaussian draws."""
    generator = torch.Generator().manual_seed(seed)
    return so3.normalize(torch.randn(batch, 4, generator=generator, dtype=torch.float64))


@pytest.mark.parametrize("seed", SEEDS)
def test_compose_with_inverse_is_identity(seed):
    """q * q^{-1} is the identity, up to the quaternion sign."""
    q = _random_quaternions(seed)
    result = so3.compose(q, so3.inverse(q))
    expected = so3.identity(batch_shape=(8,), dtype=torch.float64)
    torch.testing.assert_close(result.abs(), expected, atol=1e-12, rtol=0.0)


def test_act_rotation():
    """A rotation about Z by pi/2 maps (1, 0, 0) to (0, 1, 0)."""
    q = so3.from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    result = so3.act(q, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_act_matches_rotation_matrix(seed):
    """act(q, p) == to_matrix(q) @ p."""
    q = _random_quaternions(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    p = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    expected = (so3.to_matrix(q) @ p.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(so3.act(q, p), expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_from_matrix_round_trip(seed):
    """from_matrix(to_matrix(q)) == q, up to the quaternion sign."""
    q = _random_quaternions(seed)
    q_back = so3.from_matrix(so3.to_matrix(q))
    sign = torch.sign((q_back * q).sum(dim=-1, keepdim=True))
    torch.testing.assert_close(sign * q_back, q, atol=1e-12, rtol=0.0)


def test_normalize():
    """normalize returns a unit quaternion."""
    q_norm = so3.normalize(torch.tensor([0.0, 0.0, 0.0, 2.0]))
    torch.testing.assert_close(q_norm.norm(), torch.tensor(1.0))


def _rotations_with_equal_diagonal_entries() -> torch.Tensor:
    """``(30, 3, 3)`` exact float64 rotations: the 24 with entries in {-1, 0, 1}, then the six by pi about the face
    diagonals ``a``, ``R = a aᵀ - I`` for ``|a|² = 2``."""
    rotations = []
    for columns in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.float64)
            matrix[range(3), list(columns)] = torch.tensor(signs, dtype=torch.float64)
            if torch.det(matrix) > 0:
                rotations.append(matrix)
    for axis in [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]:
        a = torch.tensor(axis, dtype=torch.float64)
        rotations.append(torch.outer(a, a) - torch.eye(3, dtype=torch.float64))
    return torch.stack(rotations)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_from_matrix_sign_rule_at_equal_diagonal_entries(dtype):
    """``qw > 0`` when the trace is positive, else the component of the largest diagonal entry, the last of equal
    ones, is positive: one quaternion per rotation, the same at ties on every device."""
    R = _rotations_with_equal_diagonal_entries()
    q = so3.from_matrix(R.to(dtype))
    torch.testing.assert_close(so3.to_matrix(q), R.to(dtype), atol=1e-6, rtol=0.0)
    for matrix, quaternion in zip(R, q):
        diagonal = matrix.diagonal().tolist()
        largest = max(index for index in range(3) if diagonal[index] == max(diagonal))
        assert float(quaternion[3 if sum(diagonal) > 0 else largest]) > 0
