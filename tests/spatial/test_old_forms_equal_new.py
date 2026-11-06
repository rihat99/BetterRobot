"""Every old ``spatial`` form equals its replacement on seeded inputs.

The old forms are the value classes ``Motion``, ``Force``, ``Inertia``, ``Symmetric3``, the operators of
``spatial.ops`` and the re-exports ``spatial.SE3``, ``SO3``, ``Pose``. Each test pins one old form to the tensor
expression that replaces it (MIGRATION.md, ``spatial``), so callers move to the replacement with no change in value;
the module goes with the old forms. The value tests run in float32 and float64, on two batch shapes, and on four
input cases: random, a rotation near pi, a zero twist (identity pose), and a far pose (``|p| = 1e3``). Where the
replacement is a different formula (cross products by ``cross`` instead of a ``hat`` matmul, frame changes without
the 6x6 adjoint) the values agree to rounding: ``ROUNDING_ULPS`` times the dtype's eps times the largest magnitude
compared (observed: at most 16).

Not pinned, because the old forms are wrong and their replacements match Pinocchio: ``Inertia.from_capsule``
(transverse moment) and ``Inertia.from_ellipsoid`` (twice the moments), both in ``test_inertia.py``; the centre of
mass of a massless body under ``Inertia.se3_action`` (set to zero instead of moved), in ``test_algebra.py``; and the
rotational inertia of ``Inertia.se3_action`` at a far pose, which the old form recovers from the 6x6 matrix by a
cancellation of terms of size ``m |p|^2`` (at ``|p| = 1e3`` it is off by up to 1e-9 in float64 and 0.5 in float32),
so there ``transform_inertia`` is asserted against Pinocchio's ``SE3.act(Inertia)`` here. Random cases draw from a
``torch.Generator`` seeded by the ``seed`` in the test id.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import pytest
import torch

from better_robot import lie
from better_robot.lie import SE3, se3, so3
from better_robot.spatial import (
    Force,
    Inertia,
    Motion,
    Pose,
    SO3,
    Symmetric3,
    cross_force,
    cross_motion,
    inertia_from_mass_com,
    inertia_matrix,
    inertia_of_box,
    inertia_of_mesh,
    inertia_of_sphere,
    ops,
    rotational_inertia,
    transform_force,
    transform_inertia,
    transform_motion,
)
from better_robot.spatial import SE3 as SpatialSE3

SEEDS = [0, 1, 2]
DTYPES = {"float32": torch.float32, "float64": torch.float64}
BATCHES = {"5": (5,), "2x3": (2, 3)}
CASES = ["random", "near_pi", "zero_twist", "far_pose"]
ROUNDING_ULPS = 64


@dataclass(frozen=True)
class _Sample:
    """Seeded inputs of one case, each with leading axes ``batch``.

    ``twist``, ``motion``, ``wrench`` ``(..., 6)``, ``pose`` ``(..., 7)``, ``mass`` ``(...)``, ``com`` ``(..., 3)``,
    ``rotational`` ``(..., 3, 3)`` and ``inertia`` ``(..., 10)``.
    """

    twist: torch.Tensor
    motion: torch.Tensor
    wrench: torch.Tensor
    pose: torch.Tensor
    mass: torch.Tensor
    com: torch.Tensor
    rotational: torch.Tensor
    inertia: torch.Tensor
    case: str


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

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*batch, *shape, generator=generator, dtype=dtype)

    spread = randn(3, 3)
    rotational = spread @ spread.mT + 0.1 * torch.eye(3, dtype=dtype)
    mass = torch.rand(*batch, generator=generator, dtype=dtype) + 0.5
    com = randn(3)
    return _Sample(
        twist=_twists(generator, batch, dtype, case),
        motion=randn(6),
        wrench=randn(6),
        pose=se3.exp(_twists(generator, batch, dtype, case)),
        mass=mass,
        com=com,
        rotational=rotational,
        inertia=inertia_from_mass_com(mass, com, rotational_inertia=rotational),
        case=case,
    )


def _close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal to rounding: ``|actual - expected| <= ROUNDING_ULPS * eps * max(1, max |expected|)``."""
    scale = max(1.0, float(expected.abs().max()))
    tolerance = ROUNDING_ULPS * torch.finfo(expected.dtype).eps * scale
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)


def _close_inertia(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """`_close` on mass with centre of mass and on the rotational entries apart, so neither sets the other's scale."""
    _close(actual[..., :4], expected[..., :4])
    _close(actual[..., 4:], expected[..., 4:])


def _matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _pinocchio_transform_inertia(pose: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    """Packed ``pin.SE3.act(pin.Inertia)`` per pose and inertia, computed in float64, returned like ``inertia``."""
    packed = []
    for one_pose, one_inertia in zip(pose.double().reshape(-1, 7), inertia.double().reshape(-1, 10), strict=True):
        placement = pin.SE3(so3.to_matrix(one_pose[3:7]).numpy(), one_pose[:3].numpy())
        ixx, iyy, izz, ixy, ixz, iyz = one_inertia[4:10].tolist()
        rotational = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        moved = placement.act(pin.Inertia(float(one_inertia[0]), one_inertia[1:4].numpy(), rotational))
        about_com = moved.inertia
        packed.append(
            [moved.mass, *moved.lever, *np.diag(about_com), about_com[0, 1], about_com[0, 2], about_com[1, 2]]
        )
    return torch.tensor(packed, dtype=torch.float64).reshape(inertia.shape).to(inertia.dtype)


@pytest.mark.parametrize("value_class", [Motion, Force], ids=["Motion", "Force"])
def test_motion_and_force_are_their_tensors(sample: _Sample, value_class) -> None:
    """Parts, ``+``, ``-``, negation, ``zero`` and ``Motion.compose`` are the tensor operations."""
    a, b = sample.twist, sample.motion
    assert torch.equal(value_class(a).linear, a[..., :3])
    assert torch.equal(value_class(a).angular, a[..., 3:])
    assert torch.equal((value_class(a) + value_class(b)).data, a + b)
    assert torch.equal((value_class(a) - value_class(b)).data, a - b)
    assert torch.equal((-value_class(a)).data, -a)
    assert torch.equal(value_class.zero(batch_shape=a.shape[:-1], dtype=a.dtype).data, torch.zeros_like(a))
    if value_class is Motion:
        assert torch.equal(Motion(a).compose(Motion(b)).data, a + b)


def test_cross_products_equal_cross_motion_and_cross_force(sample: _Sample) -> None:
    """``Motion.cross_*``, ``ops.cross_mm``, ``cross_mf``, ``ad`` and ``ad_star`` equal the two functions."""
    twist, motion, wrench = sample.twist, sample.motion, sample.wrench
    expected_motion, expected_force = cross_motion(twist, motion), cross_force(twist, wrench)
    _close(Motion(twist).cross_motion(Motion(motion)).data, expected_motion)
    _close(ops.cross_mm(Motion(twist), Motion(motion)).data, expected_motion)
    _close(_matvec(ops.ad(Motion(twist)), motion), expected_motion)
    _close(Motion(twist).cross_force(Force(wrench)).data, expected_force)
    _close(ops.cross_mf(Motion(twist), Force(wrench)).data, expected_force)
    # ``ad_star`` reads its argument's data as the twist, despite the ``Force`` annotation.
    _close(_matvec(ops.ad_star(Force(twist)), wrench), expected_force)


def test_force_cross_motion_only_raises() -> None:
    """``Force.cross_motion`` never computed anything; its replacement is ``cross_force(twist, wrench)``."""
    with pytest.raises(NotImplementedError, match="Motion.cross_force"):
        Force(torch.zeros(6)).cross_motion(Motion(torch.zeros(6)))


def test_frame_changes_equal_transform_functions(sample: _Sample) -> None:
    """``se3_action`` (with a pose tensor or an ``SE3``) and ``ops.act_*`` equal ``transform_*(pose, value)``.

    At the far pose the old inertia form loses the rotational inertia to cancellation, so ``transform_inertia`` is
    held to Pinocchio there instead.
    """
    pose, motion, wrench, inertia = sample.pose, sample.motion, sample.wrench, sample.inertia
    for transform in (pose, SE3(pose)):
        _close(Motion(motion).se3_action(transform).data, transform_motion(pose, motion))
        _close(Force(wrench).se3_action(transform).data, transform_force(pose, wrench))
    _close(ops.act_motion(pose, Motion(motion)).data, transform_motion(pose, motion))
    _close(ops.act_force(pose, Force(wrench)).data, transform_force(pose, wrench))
    if sample.case == "far_pose":
        _close_inertia(transform_inertia(pose, inertia), _pinocchio_transform_inertia(pose, inertia))
        return
    for transform in (pose, SE3(pose)):
        _close_inertia(Inertia(inertia).se3_action(transform).data, transform_inertia(pose, inertia))
    _close_inertia(ops.act_inertia(pose, Inertia(inertia)).data, transform_inertia(pose, inertia))


def test_inertia_accessors_and_algebra_equal_the_packed_tensor(sample: _Sample) -> None:
    """``mass``, ``com``, the 3x3 ``inertia_matrix``, ``_to_6x6``, ``apply``, ``add`` and ``zero``, bit for bit."""
    inertia, motion = sample.inertia, sample.motion
    assert torch.equal(Inertia(inertia).mass, inertia[..., 0])
    assert torch.equal(Inertia(inertia).com, inertia[..., 1:4])
    assert torch.equal(Inertia(inertia).inertia_matrix, rotational_inertia(inertia))
    assert torch.equal(Inertia(inertia)._to_6x6(), inertia_matrix(inertia))
    assert torch.equal(Inertia(inertia).apply(Motion(motion)).data, _matvec(inertia_matrix(inertia), motion))
    assert torch.equal(Inertia(inertia).add(Inertia(inertia.flip(0))).data, inertia + inertia.flip(0))
    assert torch.equal(
        Inertia.zero(batch_shape=inertia.shape[:-1], dtype=inertia.dtype).data, torch.zeros_like(inertia)
    )


def test_inertia_from_mass_com_forms_equal_inertia_from_mass_com(sample: _Sample) -> None:
    """``from_mass_com_matrix`` and ``from_mass_com_sym3`` (packed ``[xx, yy, zz, xy, xz, yz]``), bit for bit."""
    mass, com, rotational = sample.mass, sample.com, sample.rotational
    expected = inertia_from_mass_com(mass, com, rotational_inertia=rotational)
    assert torch.equal(Inertia.from_mass_com_matrix(mass, com, rotational).data, expected)
    assert torch.equal(Inertia.from_mass_com_sym3(mass, com, expected[..., 4:10]).data, expected)
    first = (0,) * mass.ndim
    assert torch.equal(
        Inertia.from_mass_com_sym3(mass[first], com[first], expected[first][4:10]).data,
        inertia_from_mass_com(mass[first], com[first], rotational_inertia=rotational[first]),
    )


def test_sphere_and_box_constructors_equal_inertia_of_shape(sample: _Sample) -> None:
    """``from_sphere`` and ``from_box`` take Python floats one body at a time; bit for bit in float64.

    In float32 the old forms compute in float64 and round once, the new ones compute in float32: equal to rounding.
    """
    mass, radius = sample.mass, sample.com[..., 0].abs() + 0.1
    size = sample.com.abs() + 0.1
    dtype = mass.dtype
    old_sphere = [
        Inertia.from_sphere(float(m), float(r), dtype=dtype).data
        for m, r in zip(mass.flatten(), radius.flatten(), strict=True)
    ]
    old_box = [
        Inertia.from_box(float(m), s, dtype=dtype).data
        for m, s in zip(mass.flatten(), size.reshape(-1, 3), strict=True)
    ]
    pairs = [
        (torch.stack(old_sphere).reshape(*mass.shape, 10), inertia_of_sphere(mass, radius=radius)),
        (torch.stack(old_box).reshape(*mass.shape, 10), inertia_of_box(mass, size=size)),
    ]
    for old, new in pairs:
        if dtype == torch.float64:
            assert torch.equal(old, new)
        else:
            _close_inertia(old, new)


def test_mesh_constructor_equals_inertia_of_mesh(sample: _Sample) -> None:
    """``from_mesh(...).data`` equals ``inertia_of_mesh`` bit for bit on a box placed by the sample's poses."""
    corners = torch.tensor(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=sample.pose.dtype
    )
    faces = torch.tensor(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1]]
        + [[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )
    vertices = se3.act(sample.pose.unsqueeze(-2), corners * (sample.com.abs().unsqueeze(-2) + 0.1))
    density = sample.mass + 0.5
    assert torch.equal(
        Inertia.from_mesh(vertices, faces, density=density).data, inertia_of_mesh(vertices, faces, density=density)
    )
    assert torch.equal(Inertia.from_mesh(vertices, faces, 2.5).data, inertia_of_mesh(vertices, faces, density=2.5))


def test_symmetric3_is_the_packed_rotational_inertia(sample: _Sample) -> None:
    """``Symmetric3`` is ``inertia[..., 4:10]``: ``from_matrix``, ``to_matrix``, ``add`` and ``zero``, bit for bit."""
    inertia, rotational = sample.inertia, sample.rotational
    packed = inertia[..., 4:10]
    assert torch.equal(Symmetric3.from_matrix(rotational).data, packed)
    assert torch.equal(Symmetric3(packed).to_matrix(), rotational_inertia(inertia))
    assert torch.equal(Symmetric3(packed).add(Symmetric3(packed.flip(0))).data, packed + packed.flip(0))
    assert torch.equal(
        Symmetric3.zero(batch_shape=packed.shape[:-1], dtype=packed.dtype).data, torch.zeros_like(packed)
    )


def test_spatial_lie_classes_are_the_lie_classes() -> None:
    assert SpatialSE3 is lie.SE3
    assert SO3 is lie.SO3
    assert Pose is lie.Pose
