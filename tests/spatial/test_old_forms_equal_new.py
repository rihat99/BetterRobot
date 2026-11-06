"""Every old ``spatial`` form equals its replacement on seeded inputs.

The old forms are the value classes ``Motion``, ``Force``, ``Inertia``, ``Symmetric3``, the operators of
``spatial.ops`` and the re-exports ``spatial.SE3``, ``SO3``, ``Pose``. Each test pins one old form to the tensor
expression that replaces it (MIGRATION.md, ``spatial``), so callers move to the replacement with no change in value;
the module goes with the old forms. Where the replacement is a different formula (cross products by ``cross`` instead
of a ``hat`` matmul, frame changes without the 6x6 adjoint) the values agree to float64 rounding, ``ROUNDING``.

Not pinned, because the old forms are wrong and their replacements match Pinocchio in ``test_inertia.py``:
``Inertia.from_capsule`` (transverse moment), ``Inertia.from_ellipsoid`` (twice the moments), and the centre of mass
of a massless body under ``Inertia.se3_action`` (set to zero instead of moved). Random cases draw from a
``torch.Generator`` seeded by the ``seed`` in the test id.
"""

from __future__ import annotations

import pytest
import torch

from better_robot import lie
from better_robot.lie import SE3, se3
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
F64 = torch.float64
ROUNDING = 1e-13


def _samples(seed: int, dtype: torch.dtype = F64) -> dict[str, torch.Tensor]:
    """Seeded twists, wrenches ``(5, 6)``, poses ``(5, 7)``, rotational inertias ``(5, 3, 3)`` and inertias ``(5, 10)``."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype)

    spread = randn(5, 3, 3)
    rotational = spread @ spread.mT + 0.1 * torch.eye(3, dtype=dtype)
    mass = torch.rand(5, generator=generator, dtype=dtype) + 0.5
    com = randn(5, 3)
    return {
        "twist": randn(5, 6),
        "motion": randn(5, 6),
        "force": randn(5, 6),
        "pose": se3.exp(randn(5, 6)),
        "mass": mass,
        "com": com,
        "rotational": rotational,
        "inertia": inertia_from_mass_com(mass, com, rotational_inertia=rotational),
    }


def _close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=ROUNDING, rtol=0.0)


def _matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("value_class", [Motion, Force], ids=["Motion", "Force"])
def test_motion_and_force_are_their_tensors(value_class, seed: int, dtype: torch.dtype) -> None:
    """Parts, ``+``, ``-``, negation, ``zero`` and ``Motion.compose`` are the tensor operations."""
    sample = _samples(seed, dtype)
    a, b = sample["twist"], sample["motion"]
    assert torch.equal(value_class(a).linear, a[..., :3])
    assert torch.equal(value_class(a).angular, a[..., 3:])
    assert torch.equal((value_class(a) + value_class(b)).data, a + b)
    assert torch.equal((value_class(a) - value_class(b)).data, a - b)
    assert torch.equal((-value_class(a)).data, -a)
    assert torch.equal(value_class.zero(batch_shape=(2,), dtype=dtype).data, torch.zeros(2, 6, dtype=dtype))
    if value_class is Motion:
        assert torch.equal(Motion(a).compose(Motion(b)).data, a + b)


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_products_equal_cross_motion_and_cross_force(seed: int) -> None:
    """``Motion.cross_*``, ``ops.cross_mm``, ``cross_mf``, ``ad`` and ``ad_star`` equal the two functions."""
    sample = _samples(seed)
    twist, motion, force = sample["twist"], sample["motion"], sample["force"]
    expected_motion, expected_force = cross_motion(twist, motion), cross_force(twist, force)
    _close(Motion(twist).cross_motion(Motion(motion)).data, expected_motion)
    _close(ops.cross_mm(Motion(twist), Motion(motion)).data, expected_motion)
    _close(_matvec(ops.ad(Motion(twist)), motion), expected_motion)
    _close(Motion(twist).cross_force(Force(force)).data, expected_force)
    _close(ops.cross_mf(Motion(twist), Force(force)).data, expected_force)
    # ``ad_star`` reads its argument's data as the twist, despite the ``Force`` annotation.
    _close(_matvec(ops.ad_star(Force(twist)), force), expected_force)


def test_force_cross_motion_only_raises() -> None:
    """``Force.cross_motion`` never computed anything; its replacement is ``cross_force(twist, force)``."""
    sample = _samples(0)
    with pytest.raises(NotImplementedError, match="Motion.cross_force"):
        Force(sample["force"]).cross_motion(Motion(sample["twist"]))


@pytest.mark.parametrize("seed", SEEDS)
def test_frame_changes_equal_transform_functions(seed: int) -> None:
    """``se3_action`` (with a pose tensor or an ``SE3``) and ``ops.act_*`` equal ``transform_*(pose, value)``."""
    sample = _samples(seed)
    pose, motion, force, inertia = sample["pose"], sample["motion"], sample["force"], sample["inertia"]
    for transform in (pose, SE3(pose)):
        _close(Motion(motion).se3_action(transform).data, transform_motion(pose, motion))
        _close(Force(force).se3_action(transform).data, transform_force(pose, force))
        _close(Inertia(inertia).se3_action(transform).data, transform_inertia(pose, inertia))
    _close(ops.act_motion(pose, Motion(motion)).data, transform_motion(pose, motion))
    _close(ops.act_force(pose, Force(force)).data, transform_force(pose, force))
    _close(ops.act_inertia(pose, Inertia(inertia)).data, transform_inertia(pose, inertia))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", SEEDS)
def test_inertia_accessors_and_algebra_equal_the_packed_tensor(seed: int, dtype: torch.dtype) -> None:
    """``mass``, ``com``, the 3x3 ``inertia_matrix``, ``_to_6x6``, ``apply``, ``add`` and ``zero``, bit for bit."""
    sample = _samples(seed, dtype)
    inertia, motion = sample["inertia"], sample["motion"]
    assert torch.equal(Inertia(inertia).mass, inertia[..., 0])
    assert torch.equal(Inertia(inertia).com, inertia[..., 1:4])
    assert torch.equal(Inertia(inertia).inertia_matrix, rotational_inertia(inertia))
    assert torch.equal(Inertia(inertia)._to_6x6(), inertia_matrix(inertia))
    assert torch.equal(Inertia(inertia).apply(Motion(motion)).data, _matvec(inertia_matrix(inertia), motion))
    assert torch.equal(Inertia(inertia).add(Inertia(inertia.flip(0))).data, inertia + inertia.flip(0))
    assert torch.equal(Inertia.zero(batch_shape=(2,), dtype=dtype).data, torch.zeros(2, 10, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", SEEDS)
def test_inertia_from_mass_com_forms_equal_inertia_from_mass_com(seed: int, dtype: torch.dtype) -> None:
    """``from_mass_com_matrix`` and ``from_mass_com_sym3`` (packed ``[xx, yy, zz, xy, xz, yz]``), bit for bit."""
    sample = _samples(seed, dtype)
    mass, com, rotational = sample["mass"], sample["com"], sample["rotational"]
    expected = inertia_from_mass_com(mass, com, rotational_inertia=rotational)
    assert torch.equal(Inertia.from_mass_com_matrix(mass, com, rotational).data, expected)
    assert torch.equal(Inertia.from_mass_com_sym3(mass, com, expected[..., 4:10]).data, expected)
    assert torch.equal(
        Inertia.from_mass_com_sym3(mass[0], com[0], expected[0, 4:10]).data,
        inertia_from_mass_com(mass[0], com[0], rotational_inertia=rotational[0]),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_sphere_box_and_mesh_constructors_equal_inertia_of_shape(seed: int) -> None:
    """``from_sphere``, ``from_box`` (Python floats, here float64) and ``from_mesh(...).data``, bit for bit."""
    generator = torch.Generator().manual_seed(seed)
    mass, radius = (torch.rand(2, generator=generator, dtype=F64) + 0.2).unbind()
    size = torch.rand(3, generator=generator, dtype=F64) + 0.1
    assert torch.equal(
        Inertia.from_sphere(float(mass), float(radius), dtype=F64).data, inertia_of_sphere(mass, radius=radius)
    )
    assert torch.equal(Inertia.from_box(float(mass), size, dtype=F64).data, inertia_of_box(mass, size=size))

    # A box of edges ``size``, outward wound, placed by two random poses: vertices (2, 8, 3).
    corners = torch.tensor([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=F64)
    faces = torch.tensor(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1]]
        + [[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )
    poses = se3.exp(torch.randn(2, 1, 6, generator=generator, dtype=F64))
    vertices = se3.act(poses, corners * size)
    density = torch.rand(2, generator=generator, dtype=F64) + 0.5
    assert torch.equal(
        Inertia.from_mesh(vertices, faces, density=density).data, inertia_of_mesh(vertices, faces, density=density)
    )
    assert torch.equal(Inertia.from_mesh(vertices, faces, 2.5).data, inertia_of_mesh(vertices, faces, density=2.5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", SEEDS)
def test_symmetric3_is_the_packed_rotational_inertia(seed: int, dtype: torch.dtype) -> None:
    """``Symmetric3`` is ``inertia[..., 4:10]``: ``from_matrix``, ``to_matrix``, ``add`` and ``zero``, bit for bit."""
    sample = _samples(seed, dtype)
    inertia, rotational = sample["inertia"], sample["rotational"]
    packed = inertia[..., 4:10]
    assert torch.equal(Symmetric3.from_matrix(rotational).data, packed)
    assert torch.equal(Symmetric3(packed).to_matrix(), rotational_inertia(inertia))
    assert torch.equal(Symmetric3(packed).add(Symmetric3(packed.flip(0))).data, packed + packed.flip(0))
    assert torch.equal(Symmetric3.zero(batch_shape=(2,), dtype=dtype).data, torch.zeros(2, 6, dtype=dtype))


def test_spatial_lie_classes_are_the_lie_classes() -> None:
    assert SpatialSE3 is lie.SE3
    assert SO3 is lie.SO3
    assert Pose is lie.Pose
