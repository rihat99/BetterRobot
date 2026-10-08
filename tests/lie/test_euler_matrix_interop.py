"""Euler and homogeneous-matrix interoperability conventions."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.lie import se3, so3


def test_from_euler_matches_pypose_consumer_fixture() -> None:
    """Pinned PyPose ``euler2SO3`` result used by the motion consumer."""
    euler = torch.tensor([-math.pi / 2.0, 0.0, 0.0], dtype=torch.float64)
    expected = torch.tensor([-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float64)
    torch.testing.assert_close(so3.from_euler(euler), expected)


def test_from_euler_pins_extrinsic_xyz_order() -> None:
    euler = torch.tensor([0.31, -0.27, 0.43], dtype=torch.float64)
    roll = so3.to_matrix(so3.from_axis_angle(torch.tensor([1.0, 0.0, 0.0], dtype=euler.dtype), euler[0]))
    pitch = so3.to_matrix(so3.from_axis_angle(torch.tensor([0.0, 1.0, 0.0], dtype=euler.dtype), euler[1]))
    yaw = so3.to_matrix(so3.from_axis_angle(torch.tensor([0.0, 0.0, 1.0], dtype=euler.dtype), euler[2]))
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


@pytest.mark.parametrize(
    ("function", "value"),
    [
        pytest.param(so3.from_euler, torch.zeros(3, dtype=torch.int64), id="so3-from-euler"),
        pytest.param(
            so3.to_euler,
            torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            id="so3-to-euler",
        ),
        pytest.param(se3.from_matrix, torch.eye(4, dtype=torch.int64), id="se3-from-matrix"),
        pytest.param(
            se3.to_matrix,
            torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.int64),
            id="se3-to-matrix",
        ),
    ],
)
def test_euler_and_matrix_interop_reject_integer_tensors(function, value: torch.Tensor) -> None:
    with pytest.raises(TypeError, match="floating torch.Tensor"):
        function(value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pitch_sign", [1.0, -1.0])
def test_to_euler_gimbal_lock_puts_combined_angle_in_roll(dtype, pitch_sign):
    """At pitch ``+/-pi/2`` yaw is 0, pitch is exact and the round trip through ``from_euler`` is exact."""
    euler = torch.tensor([[0.3, pitch_sign * math.pi / 2, -0.5], [-2.0, pitch_sign * math.pi / 2, 2.5]], dtype=dtype)
    quaternion = so3.from_euler(euler)
    recovered = so3.to_euler(quaternion)
    combined = euler[:, 0] - pitch_sign * euler[:, 2]
    torch.testing.assert_close(recovered[:, 1], euler[:, 1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(recovered[:, 2], torch.zeros_like(combined), rtol=0.0, atol=0.0)
    torch.testing.assert_close(torch.cos(recovered[:, 0]), torch.cos(combined))
    torch.testing.assert_close(torch.sin(recovered[:, 0]), torch.sin(combined))
    round_trip = so3.from_euler(recovered)
    torch.testing.assert_close(round_trip * torch.sign((round_trip * quaternion).sum(-1, keepdim=True)), quaternion)
    leaf = quaternion.clone().requires_grad_()
    so3.to_euler(leaf).sum().backward()
    assert torch.isfinite(leaf.grad).all()


def test_to_euler_near_gimbal_lock_float32_keeps_regular_round_trip():
    """0.5 mrad from the lock float32 still uses the regular formulas (error 1.5e-5; a 1e-3 rad band gave 4.8e-4)."""
    quaternion = so3.from_euler(torch.tensor([[0.3, math.pi / 2 - 5e-4, 0.7]], dtype=torch.float32))
    round_trip = so3.from_euler(so3.to_euler(quaternion))
    round_trip = round_trip * torch.sign((round_trip * quaternion).sum(-1, keepdim=True))
    assert (round_trip - quaternion).abs().max().item() <= 3.5e-5


def test_to_euler_off_gimbal_lock_matches_regular_formula_bitwise():
    """The gimbal-lock branch leaves every other input bit-identical to the regular closed form."""
    quaternion = torch.randn(4096, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    qx, qy, qz, qw = so3.normalize(quaternion).unbind(-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square()))
    pitch = torch.asin((2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy.square() + qz.square()))
    assert torch.equal(so3.to_euler(quaternion), torch.stack((roll, pitch, yaw), dim=-1))
