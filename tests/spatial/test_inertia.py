"""Packed ``(..., 10)`` inertias of ``better_robot.spatial``: the spatial matrix, packing, and shape constructors.

Oracles: Pinocchio 3 (``pin.Inertia.FromSphere``, ``FromBox``, ``FromCapsule``, ``FromEllipsoid``) for the shape
constructors, compared in float64, and Pinocchio's closed form of ``Inertia * Motion`` for the matrix. Random cases
draw from a ``torch.Generator`` seeded by the ``seed`` in the test id.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest
import torch

from better_robot.spatial import (
    inertia_from_mass_com,
    inertia_matrix,
    inertia_of_box,
    inertia_of_capsule,
    inertia_of_ellipsoid,
    inertia_of_sphere,
)

SEEDS = [0, 1, 2]
TOLERANCE = 1e-12
F64 = torch.float64


def _random_inertia(seed: int, batch: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seeded ``mass (batch,)``, ``com (batch, 3)`` and symmetric positive ``rotational_inertia (batch, 3, 3)``."""
    generator = torch.Generator().manual_seed(seed)
    spread = torch.randn(batch, 3, 3, generator=generator, dtype=F64)
    mass = torch.rand(batch, generator=generator, dtype=F64) + 0.5
    com = torch.randn(batch, 3, generator=generator, dtype=F64)
    return mass, com, spread @ spread.mT + 0.1 * torch.eye(3, dtype=F64)


def _momentum_closed_form(mass: torch.Tensor, com: torch.Tensor, rotational: torch.Tensor, twist: torch.Tensor):
    """``I * v`` written out as Pinocchio's ``Inertia::__mult__``: ``f = m (v - c × w)``, ``n = I_c w + c × f``."""
    linear = mass[..., None] * (twist[..., :3] - torch.linalg.cross(com, twist[..., 3:]))
    angular = (rotational @ twist[..., 3:, None]).squeeze(-1) + torch.linalg.cross(com, linear)
    return torch.cat([linear, angular], dim=-1)


def _principal_moments(inertia: torch.Tensor) -> np.ndarray:
    return inertia[..., 4:7].numpy()


@pytest.mark.parametrize("seed", SEEDS)
def test_inertia_from_mass_com_packs_the_layout(seed: int) -> None:
    mass, com, rotational = _random_inertia(seed)
    inertia = inertia_from_mass_com(mass, com, rotational_inertia=rotational)
    assert inertia.shape == (5, 10)
    assert torch.equal(inertia[:, 0], mass)
    assert torch.equal(inertia[:, 1:4], com)
    upper = rotational[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    torch.testing.assert_close(inertia[:, 4:10], upper, atol=1e-15, rtol=0.0)


def test_inertia_from_mass_com_stores_the_symmetric_part_and_broadcasts() -> None:
    rotational = torch.tensor([[1.0, 0.2, 0.0], [0.4, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=F64)
    inertia = inertia_from_mass_com(
        torch.tensor(2.0, dtype=F64), torch.zeros(4, 3, dtype=F64), rotational_inertia=rotational
    )
    assert inertia.shape == (4, 10)
    torch.testing.assert_close(inertia[0, 4:10], torch.tensor([1.0, 2.0, 3.0, 0.3, 0.0, 0.0], dtype=F64))


@pytest.mark.parametrize("seed", SEEDS)
def test_inertia_matrix_times_twist_is_the_momentum(seed: int) -> None:
    mass, com, rotational = _random_inertia(seed)
    twist = torch.randn(5, 6, generator=torch.Generator().manual_seed(seed + 100), dtype=F64)
    inertia = inertia_from_mass_com(mass, com, rotational_inertia=rotational)
    momentum = (inertia_matrix(inertia) @ twist.unsqueeze(-1)).squeeze(-1)
    error = float((momentum - _momentum_closed_form(mass, com, rotational, twist)).abs().max())
    assert error <= TOLERANCE


def test_inertia_matrix_with_the_com_at_the_origin_is_block_diagonal() -> None:
    inertia = inertia_from_mass_com(
        torch.tensor(2.0, dtype=F64),
        torch.zeros(3, dtype=F64),
        rotational_inertia=torch.diag(torch.tensor([0.4, 0.5, 0.6], dtype=F64)),
    )
    expected = torch.diag(torch.tensor([2.0, 2.0, 2.0, 0.4, 0.5, 0.6], dtype=F64))
    assert torch.equal(inertia_matrix(inertia), expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_shapes_match_pinocchio(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    mass, radius, length = (torch.rand(3, generator=generator, dtype=F64) + 0.2).unbind()
    size = torch.rand(3, generator=generator, dtype=F64) + 0.1
    semi_axes = torch.rand(3, generator=generator, dtype=F64) + 0.1
    m, r, h = float(mass), float(radius), float(length)
    cases = {
        "sphere": (inertia_of_sphere(mass, radius=radius), pin.Inertia.FromSphere(m, r)),
        "box": (inertia_of_box(mass, size=size), pin.Inertia.FromBox(m, *size.tolist())),
        "capsule": (inertia_of_capsule(mass, radius=radius, length=length), pin.Inertia.FromCapsule(m, r, h)),
        "ellipsoid": (
            inertia_of_ellipsoid(mass, semi_axes=semi_axes),
            pin.Inertia.FromEllipsoid(m, *semi_axes.tolist()),
        ),
    }
    for name, (inertia, oracle) in cases.items():
        assert inertia.shape == (10,), name
        assert float(inertia[0]) == m, name
        assert torch.equal(inertia[1:4], torch.zeros(3, dtype=F64)), name
        assert torch.equal(inertia[7:10], torch.zeros(3, dtype=F64)), name
        error = float(np.abs(_principal_moments(inertia) - np.diag(oracle.inertia)).max())
        assert error <= TOLERANCE, name


def test_shape_limits_reduce_to_the_sphere() -> None:
    """A capsule of length 0 and an ellipsoid with equal semi-axes are spheres of the same mass and radius."""
    mass, radius = torch.tensor(3.0, dtype=F64), torch.tensor(0.7, dtype=F64)
    sphere = inertia_of_sphere(mass, radius=radius)
    torch.testing.assert_close(inertia_of_capsule(mass, radius=radius, length=torch.tensor(0.0, dtype=F64)), sphere)
    torch.testing.assert_close(inertia_of_ellipsoid(mass, semi_axes=radius.expand(3)), sphere)
    cube = inertia_of_box(mass, size=torch.full((3,), 2.0, dtype=F64))
    torch.testing.assert_close(cube[4:7], torch.full((3,), 3.0 * 8.0 / 12.0, dtype=F64))


def test_shapes_broadcast_and_keep_the_dtype() -> None:
    mass = torch.tensor([1.0, 2.0])
    assert inertia_of_sphere(mass, radius=torch.tensor(0.5)).shape == (2, 10)
    assert inertia_of_box(mass[:, None], size=torch.ones(3, 3)).shape == (2, 3, 10)
    assert inertia_of_capsule(mass, radius=torch.tensor(0.1), length=torch.tensor([0.2, 0.3])).shape == (2, 10)
    ellipsoid = inertia_of_ellipsoid(mass, semi_axes=torch.ones(2, 3))
    assert ellipsoid.shape == (2, 10)
    assert ellipsoid.dtype == torch.float32


def test_shapes_are_differentiable() -> None:
    mass, radius, length = (torch.tensor(value, dtype=F64, requires_grad=True) for value in (1.5, 0.3, 0.8))
    assert torch.autograd.gradcheck(lambda m, r, h: inertia_of_capsule(m, radius=r, length=h), (mass, radius, length))
    semi_axes = torch.tensor([0.2, 0.3, 0.4], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda m, a: inertia_of_ellipsoid(m, semi_axes=a), (mass, semi_axes))


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        pytest.param(
            lambda: inertia_of_sphere(1.0, radius=torch.tensor(0.1)),
            TypeError,
            "mass must be a floating",
            id="python-float",
        ),
        pytest.param(
            lambda: inertia_of_sphere(torch.tensor(1), radius=torch.tensor(0.1)), TypeError, "torch.int64", id="integer"
        ),
        pytest.param(
            lambda: inertia_of_sphere(torch.tensor(1.0), radius=torch.tensor(0.1, dtype=F64)),
            ValueError,
            r"radius.dtype=torch.float64 differs from mass.dtype=torch.float32",
            id="dtype",
        ),
        pytest.param(
            lambda: inertia_of_box(torch.tensor(1.0), size=torch.ones(2)),
            ValueError,
            r"size must have shape \(..., 3\)",
            id="box-size",
        ),
        pytest.param(
            lambda: inertia_of_ellipsoid(torch.tensor(1.0), semi_axes=torch.ones(())),
            ValueError,
            "semi_axes must have shape",
            id="semi-axes",
        ),
        pytest.param(
            lambda: inertia_from_mass_com(torch.tensor(1.0), torch.zeros(3), rotational_inertia=torch.eye(4)),
            ValueError,
            r"rotational_inertia must have shape \(..., 3, 3\)",
            id="rotational-inertia",
        ),
        pytest.param(
            lambda: inertia_from_mass_com(
                torch.tensor(1.0), torch.zeros(3, dtype=F64), rotational_inertia=torch.eye(3)
            ),
            ValueError,
            "com.dtype",
            id="com-dtype",
        ),
    ],
)
def test_constructors_reject_wrong_types_shapes_and_dtypes(build, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        build()


@pytest.mark.cuda
def test_constructors_reject_mixed_devices() -> None:
    with pytest.raises(ValueError, match="radius is on cuda"):
        inertia_of_sphere(torch.tensor(1.0), radius=torch.tensor(0.1, device="cuda"))
