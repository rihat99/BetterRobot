"""Euler and homogeneous-matrix interoperability conventions."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.lie import se3, so3


def test_from_euler_matches_pypose_consumer_fixture() -> None:
    """Pinned PyPose ``euler2SO3`` result used by the motion consumer."""
    euler = torch.tensor([-math.pi / 2.0, 0.0, 0.0], dtype=torch.float64)
    expected = torch.tensor(
        [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float64
    )
    torch.testing.assert_close(so3.from_euler(euler), expected)


def test_from_euler_pins_extrinsic_xyz_order() -> None:
    euler = torch.tensor([0.31, -0.27, 0.43], dtype=torch.float64)
    roll = so3.to_matrix(
        so3.from_axis_angle(
            torch.tensor([1.0, 0.0, 0.0], dtype=euler.dtype), euler[0]
        )
    )
    pitch = so3.to_matrix(
        so3.from_axis_angle(
            torch.tensor([0.0, 1.0, 0.0], dtype=euler.dtype), euler[1]
        )
    )
    yaw = so3.to_matrix(
        so3.from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0], dtype=euler.dtype), euler[2]
        )
    )
    torch.testing.assert_close(so3.to_matrix(so3.from_euler(euler)), yaw @ pitch @ roll)


def test_euler_quaternion_roundtrip_batched_away_from_gimbal_lock() -> None:
    euler = torch.tensor(
        [
            [[0.2, -0.3, 0.4], [-0.7, 0.5, -0.2]],
            [[1.0, 0.25, -0.8], [-1.2, -0.4, 0.6]],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(so3.to_euler(so3.from_euler(euler)), euler)


def test_from_euler_gradcheck() -> None:
    euler = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(so3.from_euler, (euler,))


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_se3_homogeneous_roundtrip_and_batch_shapes(
    batch_shape: tuple[int, ...],
) -> None:
    generator = torch.Generator().manual_seed(41)
    tangent = torch.randn((*batch_shape, 6), generator=generator, dtype=torch.float64) * 0.3
    pose = se3.exp(tangent)

    matrix = se3.to_matrix(pose)
    recovered = se3.from_matrix(matrix)

    assert matrix.shape == (*batch_shape, 4, 4)
    assert recovered.shape == (*batch_shape, 7)
    bottom = matrix.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(*batch_shape, 4)
    torch.testing.assert_close(matrix[..., 3, :], bottom)
    torch.testing.assert_close(se3.to_matrix(recovered), matrix)
    torch.testing.assert_close(
        recovered[..., 3:].norm(dim=-1),
        torch.ones(batch_shape, dtype=matrix.dtype),
    )
    rotation = matrix[..., :3, :3]
    eye = torch.eye(3, dtype=matrix.dtype).expand(*batch_shape, 3, 3)
    torch.testing.assert_close(rotation.mT @ rotation, eye)


def test_se3_matrix_roundtrip_preserves_translation_and_rotation() -> None:
    euler = torch.tensor([0.2, -0.4, 0.3], dtype=torch.float64)
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = so3.to_matrix(so3.from_euler(euler))
    matrix[:3, 3] = torch.tensor([1.2, -0.7, 0.4], dtype=torch.float64)
    torch.testing.assert_close(se3.to_matrix(se3.from_matrix(matrix)), matrix)
