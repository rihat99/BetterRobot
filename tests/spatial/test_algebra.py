"""Spatial cross products and frame changes of ``better_robot.spatial`` against Pinocchio and their identities.

Oracle: Pinocchio 3 (``pin.Motion.cross``, ``pin.SE3.act`` on ``pin.Motion``, ``pin.Force`` and ``pin.Inertia``),
compared in float64 on seeded random twists, wrenches, poses and inertias. Random cases draw from a
``torch.Generator`` seeded by the ``seed`` in the test id.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.lie import se3, so3
from better_robot.spatial import (
    cross_force,
    cross_motion,
    inertia_from_mass_com,
    inertia_matrix,
    rotational_inertia,
    transform_force,
    transform_inertia,
    transform_motion,
)

SEEDS = [0, 1, 2, 3]
TOLERANCE = 1e-12


def _samples(seed: int, batch: int = 6, dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """Seeded twists ``(batch, 6)``, wrenches ``(batch, 6)``, poses ``(batch, 7)`` and inertias ``(batch, 10)``."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype)

    spread = randn(batch, 3, 3)
    inertia = inertia_from_mass_com(
        torch.rand(batch, generator=generator, dtype=dtype) + 0.5,
        randn(batch, 3),
        rotational_inertia=spread @ spread.mT + 0.1 * torch.eye(3, dtype=dtype),
    )
    return {
        "twist": randn(batch, 6),
        "motion": randn(batch, 6),
        "force": randn(batch, 6),
        "pose": se3.exp(randn(batch, 6)),
        "inertia": inertia,
    }


def _pin_pose(pose: torch.Tensor) -> pin.SE3:
    return pin.SE3(so3.to_matrix(pose[3:7]).numpy(), pose[:3].numpy())


def _pin_inertia(inertia: torch.Tensor) -> pin.Inertia:
    ixx, iyy, izz, ixy, ixz, iyz = inertia[4:10].tolist()
    rotational = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return pin.Inertia(float(inertia[0]), inertia[1:4].numpy(), rotational)


def _max_error(actual: torch.Tensor, expected: np.ndarray) -> float:
    return float((actual - torch.from_numpy(np.asarray(expected))).abs().max())


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_products_match_pinocchio(seed: int) -> None:
    sample = _samples(seed)
    for twist, motion, force in zip(sample["twist"], sample["motion"], sample["force"], strict=True):
        pin_twist = pin.Motion(twist.numpy())
        motion_error = _max_error(cross_motion(twist, motion), pin_twist.cross(pin.Motion(motion.numpy())).vector)
        force_error = _max_error(cross_force(twist, force), pin_twist.cross(pin.Force(force.numpy())).vector)
        assert motion_error <= TOLERANCE
        assert force_error <= TOLERANCE


@pytest.mark.parametrize("seed", SEEDS)
def test_frame_changes_match_pinocchio(seed: int) -> None:
    sample = _samples(seed)
    for pose, motion, force, inertia in zip(
        sample["pose"], sample["motion"], sample["force"], sample["inertia"], strict=True
    ):
        placement = _pin_pose(pose)
        moved = placement.act(_pin_inertia(inertia))
        motion_error = _max_error(transform_motion(pose, motion), placement.act(pin.Motion(motion.numpy())).vector)
        force_error = _max_error(transform_force(pose, force), placement.act(pin.Force(force.numpy())).vector)
        inertia_error = _max_error(inertia_matrix(transform_inertia(pose, inertia)), moved.matrix())
        com_error = _max_error(transform_inertia(pose, inertia)[1:4], moved.lever)
        assert motion_error <= TOLERANCE
        assert force_error <= TOLERANCE
        assert inertia_error <= TOLERANCE
        assert com_error <= TOLERANCE


@pytest.mark.parametrize("seed", SEEDS)
def test_inertia_matrix_matches_pinocchio(seed: int) -> None:
    for inertia in _samples(seed)["inertia"]:
        assert _max_error(inertia_matrix(inertia), _pin_inertia(inertia).matrix()) <= TOLERANCE


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_products_are_dual_and_antisymmetric(seed: int) -> None:
    """``twist ×m twist = 0``, ``a ×m b = -(b ×m a)`` and ``(twist ×f force) · motion = -force · (twist ×m motion)``."""
    sample = _samples(seed)
    twist, motion, force = sample["twist"], sample["motion"], sample["force"]
    torch.testing.assert_close(cross_motion(twist, twist), torch.zeros_like(twist), atol=1e-15, rtol=0.0)
    torch.testing.assert_close(cross_motion(twist, motion), -cross_motion(motion, twist), atol=1e-15, rtol=0.0)
    power = (cross_force(twist, force) * motion).sum(-1)
    torch.testing.assert_close(power, -(force * cross_motion(twist, motion)).sum(-1), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_frame_changes_compose_invert_and_keep_power_and_momentum(seed: int) -> None:
    sample = _samples(seed)
    pose, motion, force, inertia = sample["pose"], sample["motion"], sample["force"], sample["inertia"]
    inverse = se3.inverse(pose)
    torch.testing.assert_close(transform_motion(inverse, transform_motion(pose, motion)), motion)
    torch.testing.assert_close(transform_force(inverse, transform_force(pose, force)), force)
    torch.testing.assert_close(transform_inertia(inverse, transform_inertia(pose, inertia)), inertia)
    torch.testing.assert_close(
        (transform_force(pose, force) * transform_motion(pose, motion)).sum(-1), (force * motion).sum(-1)
    )
    momentum = (inertia_matrix(inertia) @ motion.unsqueeze(-1)).squeeze(-1)
    moved_momentum = inertia_matrix(transform_inertia(pose, inertia)) @ transform_motion(pose, motion).unsqueeze(-1)
    torch.testing.assert_close(moved_momentum.squeeze(-1), transform_force(pose, momentum))
    adjoint = (se3.adjoint(pose) @ motion.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(transform_motion(pose, motion), adjoint)


def test_transform_inertia_keeps_the_centre_of_mass_of_a_massless_body() -> None:
    """A zero mass still carries its centre of mass through the pose, so the round trip is exact."""
    sample = _samples(0, batch=2)
    massless = sample["inertia"].index_fill(-1, torch.tensor([0]), 0.0)
    moved = transform_inertia(sample["pose"], massless)
    torch.testing.assert_close(moved[..., 1:4], se3.act(sample["pose"], massless[..., 1:4]))
    torch.testing.assert_close(transform_inertia(se3.inverse(sample["pose"]), moved), massless)


def test_leading_axes_broadcast() -> None:
    sample = _samples(1, batch=4)
    pose, twist = sample["pose"][0], sample["twist"]
    assert transform_motion(pose, twist).shape == (4, 6)
    assert transform_force(pose, twist[:, None, :].expand(4, 3, 6)).shape == (4, 3, 6)
    assert transform_inertia(sample["pose"][:, None], sample["inertia"][0]).shape == (4, 1, 10)
    assert cross_motion(twist[0], twist).shape == (4, 6)
    assert cross_force(twist[:, None], twist).shape == (4, 4, 6)
    torch.testing.assert_close(transform_motion(pose, twist)[2], transform_motion(pose, twist[2]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype: torch.dtype) -> None:
    sample = _samples(2, dtype=dtype)
    outputs = [
        cross_motion(sample["twist"], sample["motion"]),
        cross_force(sample["twist"], sample["force"]),
        transform_motion(sample["pose"], sample["motion"]),
        transform_force(sample["pose"], sample["force"]),
        transform_inertia(sample["pose"], sample["inertia"]),
    ]
    assert all(output.dtype == dtype for output in outputs)


def test_frame_changes_are_differentiable() -> None:
    sample = _samples(3, batch=2)
    pose = sample["pose"].requires_grad_(True)
    inertia = sample["inertia"].requires_grad_(True)
    motion = sample["motion"].requires_grad_(True)
    assert torch.autograd.gradcheck(transform_motion, (pose, motion))
    assert torch.autograd.gradcheck(transform_force, (pose, motion))
    assert torch.autograd.gradcheck(transform_inertia, (pose, inertia))
    assert torch.autograd.gradcheck(cross_motion, (motion, motion.detach().flip(0).requires_grad_(True)))
    assert torch.autograd.gradcheck(cross_force, (motion, motion.detach().flip(0).requires_grad_(True)))


# Each public function with its parameters as (name, trailing size), in call order.
_CHECKED = {
    "cross_motion": (cross_motion, ("twist", 6), ("other", 6)),
    "cross_force": (cross_force, ("twist", 6), ("wrench", 6)),
    "transform_motion": (transform_motion, ("pose", 7), ("twist", 6)),
    "transform_force": (transform_force, ("pose", 7), ("wrench", 6)),
    "transform_inertia": (transform_inertia, ("pose", 7), ("inertia", 10)),
    "inertia_matrix": (inertia_matrix, ("inertia", 10)),
    "rotational_inertia": (rotational_inertia, ("inertia", 10)),
}


@pytest.mark.parametrize("name", list(_CHECKED))
def test_public_functions_check_type_dtype_trailing_size_and_device(name: str) -> None:
    """Each public function rejects a non-tensor, an integer dtype, a wrong trailing size and mixed dtypes or devices.

    A non-tensor raises ``TypeError``, the rest ``ValueError`` with the fix in the message; the wrong size is one
    entry too many, so ``inertia_matrix`` of an ``(..., 11)`` tensor is among the cases.
    """
    function, *parameters = _CHECKED[name]
    valid = [torch.zeros(2, size) for _, size in parameters]
    for index, (parameter, size) in enumerate(parameters):
        bad_inputs = [
            (torch.zeros(2, size).tolist(), TypeError, rf"{parameter} must be a torch.Tensor, got list"),
            (
                torch.zeros(2, size, dtype=torch.int64),
                ValueError,
                rf"{parameter}.dtype=torch.int64 is not float32 or float64",
            ),
            (
                torch.zeros(2, size + 1),
                ValueError,
                rf"{parameter} must have shape \(\.\.\., {size}\), got \(2, {size + 1}\)",
            ),
        ]
        for bad, error, match in bad_inputs:
            arguments = [*valid[:index], bad, *valid[index + 1 :]]
            with pytest.raises(error, match=match):
                function(*arguments)
    if len(parameters) == 2:
        (first, size_first), (second, size_second) = parameters
        with pytest.raises(ValueError, match=rf"{second}.dtype=torch.float64 != {first}.dtype=torch.float32"):
            function(torch.zeros(size_first), torch.zeros(size_second, dtype=torch.float64))
        # A meta tensor has a device of its own, so the device check runs without an accelerator.
        with pytest.raises(ValueError, match=rf"{second}.device=meta != {first}.device=cpu; pass {second}.to\(cpu\)"):
            function(torch.zeros(size_first), torch.zeros(size_second, device="meta"))
        with pytest.raises(
            ShapeError, match=rf"{first} has batch axes \(3,\) and {second} \(4,\), which do not broadcast"
        ):
            function(torch.zeros(3, size_first), torch.zeros(4, size_second))
