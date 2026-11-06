"""Every old ``lie`` form equals its replacement on seeded inputs.

The old forms are the value classes ``SE3``, ``SO3`` (alias ``Pose``), the names ``se3.adjoint_inv`` and
``*_jacobian_inv_*``, ``so3.adjoint`` and ``se3.apply_base``. Each test pins one old form to the tensor function that
replaces it (MIGRATION.md, ``lie``), so callers move to the replacement with no change in value; the module goes with
the old forms. Random cases draw from a ``torch.Generator`` seeded by the ``seed`` in the test id.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import Pose, SE3, SO3, se3, so3, tangents

SEEDS = [0, 1, 2]


def _poses(seed: int, batch: tuple[int, ...] = (4,), dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return se3.exp(0.5 * torch.randn(*batch, 6, generator=generator, dtype=dtype))


def _rotations(seed: int, batch: tuple[int, ...] = (4,), dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return so3.exp(0.5 * torch.randn(*batch, 3, generator=generator, dtype=dtype))


def _points(seed: int, batch: tuple[int, ...] = (4,), dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(*batch, 3, generator=torch.Generator().manual_seed(seed + 100), dtype=dtype)


# (functional module, value class, sampler)
_GROUPS = [
    pytest.param((so3, SO3, _rotations), id="SO3"),
    pytest.param((se3, SE3, _poses), id="SE3"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", _GROUPS)
@pytest.mark.parametrize("seed", SEEDS)
def test_value_class_matches_functions(seed, group, dtype):
    """``@``, ``inverse``, ``log``, ``exp``, ``normalize``, ``identity`` equal the functions bit for bit."""
    functional, value_class, sample = group
    a, b, p = sample(seed, dtype=dtype), sample(seed + 10, dtype=dtype), _points(seed, dtype=dtype)
    assert torch.equal((value_class(a) @ value_class(b)).tensor, functional.compose(a, b))
    assert torch.equal(value_class(a).compose(value_class(b)).tensor, functional.compose(a, b))
    assert torch.equal(value_class(a) @ p, functional.act(a, p))
    assert torch.equal(value_class(a).inverse().tensor, functional.inverse(a))
    assert torch.equal(value_class(a).normalize().tensor, functional.normalize(a))
    tangent = value_class(a).log()
    assert torch.equal(tangent, functional.log(a))
    assert torch.equal(value_class.exp(tangent).tensor, functional.exp(tangent))
    assert torch.equal(
        value_class.identity(batch_shape=(2,), dtype=dtype).tensor,
        functional.identity(batch_shape=(2,), dtype=dtype),
    )
    assert value_class(a).__matmul__(1.0) is NotImplemented


@pytest.mark.parametrize("group", _GROUPS)
def test_value_class_rejects_scalar_multiplication(group):
    _, value_class, sample = group
    value = value_class(sample(0, (1,), torch.float32))
    with pytest.raises(TypeError):
        _ = value * 2.0
    with pytest.raises(TypeError):
        _ = 2.0 * value


@pytest.mark.parametrize("seed", SEEDS)
def test_so3_value_class_matrix_conversions(seed):
    q = _rotations(seed)
    assert torch.equal(SO3(q).to_matrix(), so3.to_matrix(q))
    assert torch.equal(SO3.from_matrix(so3.to_matrix(q)).tensor, so3.from_matrix(so3.to_matrix(q)))


@pytest.mark.parametrize("seed", SEEDS)
def test_se3_value_class_parts_and_adjoints(seed):
    T = _poses(seed, (2, 3))
    assert Pose is SE3
    assert torch.equal(SE3(T).translation, T[..., :3])
    assert torch.equal(SE3(T).rotation.tensor, T[..., 3:7])
    assert torch.equal(SE3(T).adjoint(), se3.adjoint(T))
    assert torch.equal(SE3(T).adjoint_inv(), se3.adjoint_inverse(T))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (se3.adjoint_inv, se3.adjoint_inverse),
        (tangents.right_jacobian_inv_so3, tangents.right_jacobian_inverse_so3),
        (tangents.left_jacobian_inv_so3, tangents.left_jacobian_inverse_so3),
        (tangents.right_jacobian_inv_se3, tangents.right_jacobian_inverse_se3),
        (tangents.left_jacobian_inv_se3, tangents.left_jacobian_inverse_se3),
    ],
    ids=["adjoint", "right_so3", "left_so3", "right_se3", "left_se3"],
)
def test_inv_names_are_the_inverse_functions(old, new):
    """Each ``*_inv*`` name is the same function object as its ``*_inverse*`` replacement, so every value agrees."""
    assert old is new


@pytest.mark.parametrize("seed", SEEDS)
def test_so3_adjoint_equals_to_matrix(seed):
    q = _rotations(seed)
    assert torch.equal(so3.adjoint(q), so3.to_matrix(q))


@pytest.mark.parametrize("seed", SEEDS)
def test_apply_base_equals_compose_with_a_broadcast_base(seed):
    """``apply_base(base, poses)`` is ``compose(base[..., None, :], poses)``."""
    base, poses = _poses(seed, (2,)), _poses(seed + 10, (2, 5))
    assert torch.equal(se3.apply_base(base, poses), se3.compose(base.unsqueeze(-2), poses))
