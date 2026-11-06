"""Every old ``lie`` form equals its replacement on seeded inputs.

The old forms are the value classes ``SE3``, ``SO3`` (alias ``Pose``), the names ``se3.adjoint_inv`` and
``*_jacobian_inv_*``, ``so3.adjoint`` and ``se3.apply_base``. Each test pins one old form to the tensor function that
replaces it (MIGRATION.md, ``lie``), so callers move to the replacement with no change in value; the module goes with
the old forms. The value tests run in float32 and float64, on two batch shapes, and on four input cases: random, a
rotation near pi, a zero twist (identity), and a far pose (``|p| = 1e3``). Random cases draw from a
``torch.Generator`` seeded by the ``seed`` in the test id.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pytest
import torch

from better_robot.lie import Pose, SE3, SO3, se3, so3, tangents

SEEDS = [0, 1, 2]
DTYPES = {"float32": torch.float32, "float64": torch.float64}
BATCHES = {"5": (5,), "2x3": (2, 3)}
CASES = ["random", "near_pi", "zero_twist", "far_pose"]


@dataclass(frozen=True)
class _Sample:
    """Seeded inputs of one case: poses ``(*batch, 7)``, rotations ``(*batch, 4)``, points ``(*batch, 3)``."""

    pose: torch.Tensor
    other_pose: torch.Tensor
    rotation: torch.Tensor
    other_rotation: torch.Tensor
    point: torch.Tensor
    batch: tuple[int, ...]


def _twists(generator: torch.Generator, batch: tuple[int, ...], dtype: torch.dtype, case: str) -> torch.Tensor:
    """Twists ``(*batch, 6)``: random, rotating by pi - 1e-6, zero, or translating by 1e3."""
    twist = 0.5 * torch.randn(*batch, 6, generator=generator, dtype=dtype)
    direction = torch.nn.functional.normalize(torch.randn(*batch, 3, generator=generator, dtype=dtype), dim=-1)
    if case == "near_pi":
        return torch.cat((twist[..., :3], (math.pi - 1e-6) * direction), dim=-1)
    if case == "zero_twist":
        return torch.zeros_like(twist)
    if case == "far_pose":
        return torch.cat((1e3 * direction, twist[..., 3:]), dim=-1)
    return twist


@pytest.fixture(
    params=list(itertools.product(SEEDS, DTYPES, BATCHES, CASES)),
    ids=lambda param: "-".join((param[1], param[2], param[3], f"seed{param[0]}")),
)
def sample(request: pytest.FixtureRequest) -> _Sample:
    seed, dtype_name, batch_name, case = request.param
    dtype, batch = DTYPES[dtype_name], BATCHES[batch_name]
    generator = torch.Generator().manual_seed(seed)
    twist, other_twist = _twists(generator, batch, dtype, case), _twists(generator, batch, dtype, case)
    return _Sample(
        pose=se3.exp(twist),
        other_pose=se3.exp(other_twist),
        rotation=so3.exp(twist[..., 3:]),
        other_rotation=so3.exp(other_twist[..., 3:]),
        point=torch.randn(*batch, 3, generator=generator, dtype=dtype),
        batch=batch,
    )


# (functional module, value class, name of the sample field)
_GROUPS = [
    pytest.param((so3, SO3, "rotation"), id="SO3"),
    pytest.param((se3, SE3, "pose"), id="SE3"),
]


@pytest.mark.parametrize("group", _GROUPS)
def test_value_class_matches_functions(sample: _Sample, group) -> None:
    """``@``, ``inverse``, ``log``, ``exp``, ``normalize``, ``identity`` equal the functions bit for bit."""
    functional, value_class, field = group
    a, b, p = getattr(sample, field), getattr(sample, f"other_{field}"), sample.point
    assert torch.equal((value_class(a) @ value_class(b)).tensor, functional.compose(a, b))
    assert torch.equal(value_class(a).compose(value_class(b)).tensor, functional.compose(a, b))
    assert torch.equal(value_class(a) @ p, functional.act(a, p))
    assert torch.equal(value_class(a).inverse().tensor, functional.inverse(a))
    assert torch.equal(value_class(a).normalize().tensor, functional.normalize(a))
    tangent = value_class(a).log()
    assert torch.equal(tangent, functional.log(a))
    assert torch.equal(value_class.exp(tangent).tensor, functional.exp(tangent))
    assert torch.equal(
        value_class.identity(batch_shape=sample.batch, dtype=a.dtype).tensor,
        functional.identity(batch_shape=sample.batch, dtype=a.dtype),
    )
    assert value_class(a).__matmul__(1.0) is NotImplemented


@pytest.mark.parametrize("group", _GROUPS)
def test_value_class_rejects_scalar_multiplication(group) -> None:
    _, value_class, _ = group
    value = value_class(so3.identity(batch_shape=(1,)) if value_class is SO3 else se3.identity(batch_shape=(1,)))
    with pytest.raises(TypeError):
        _ = value * 2.0
    with pytest.raises(TypeError):
        _ = 2.0 * value


def test_so3_value_class_matrix_conversions(sample: _Sample) -> None:
    q = sample.rotation
    assert torch.equal(SO3(q).to_matrix(), so3.to_matrix(q))
    assert torch.equal(SO3.from_matrix(so3.to_matrix(q)).tensor, so3.from_matrix(so3.to_matrix(q)))


def test_se3_value_class_parts_and_adjoints(sample: _Sample) -> None:
    T = sample.pose
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
def test_inv_names_are_the_inverse_functions(old, new) -> None:
    """Each ``*_inv*`` name is the same function object as its ``*_inverse*`` replacement, so every value agrees."""
    assert old is new


def test_so3_adjoint_equals_to_matrix(sample: _Sample) -> None:
    q = sample.rotation
    assert torch.equal(so3.adjoint(q), so3.to_matrix(q))


def test_apply_base_equals_compose_with_a_broadcast_base(sample: _Sample) -> None:
    """``apply_base(base, poses)`` is ``compose(base[..., None, :], poses)`` for link poses ``(*batch, 3, 7)``."""
    base = sample.pose
    poses = torch.stack((sample.other_pose, sample.pose, se3.inverse(sample.other_pose)), dim=-2)
    assert torch.equal(se3.apply_base(base, poses), se3.compose(base.unsqueeze(-2), poses))
