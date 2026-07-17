"""Batched manifold-aware trajectory kernel smoothing."""

from __future__ import annotations

import torch

from better_robot.lie import se3, so3
from better_robot.tasks import Trajectory, smooth_trajectory


def _reference_iterative_mean(q: torch.Tensor, kernel: torch.Tensor, *, kind: str) -> torch.Tensor:
    """Looped consumer reference, intentionally simple and unbatched."""
    kernel = kernel / kernel.sum()
    radius = kernel.numel() // 2
    interpolate = so3.slerp if kind == "so3" else se3.sclerp
    outputs = []
    for center in range(q.shape[-2]):
        indices = [min(max(center + offset, 0), q.shape[-2] - 1) for offset in range(-radius, radius + 1)]
        mean = q[indices[0]]
        accumulated = kernel[0]
        for index, weight in zip(indices[1:], kernel[1:], strict=True):
            accumulated = accumulated + weight
            mean = interpolate(mean, q[index], weight / accumulated)
        outputs.append(mean)
    return torch.stack(outputs)


def test_constant_quaternion_trajectory_is_bit_identical() -> None:
    quaternion = so3.from_euler(torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64))
    q = quaternion.expand(8, 4).clone()
    trajectory = Trajectory(t=torch.arange(8, dtype=q.dtype), q=q)
    smoothed = smooth_trajectory(trajectory, torch.tensor([1.0, 2.0, 1.0]))
    assert torch.equal(smoothed.q, q)


def test_smoothing_reduces_noisy_rotation_variance() -> None:
    angles = torch.tensor(
        [0.0, 0.5, -0.4, 0.6, -0.5, 0.45, -0.35, 0.0], dtype=torch.float64
    )
    euler = torch.zeros((angles.numel(), 3), dtype=angles.dtype)
    euler[:, 2] = angles
    trajectory = Trajectory(t=torch.arange(angles.numel(), dtype=angles.dtype), q=so3.from_euler(euler))
    smoothed = smooth_trajectory(trajectory, torch.ones(3, dtype=angles.dtype))
    before = so3.to_euler(trajectory.q)[..., 2].var()
    after = so3.to_euler(smoothed.q)[..., 2].var()
    assert after < before * 0.2


def test_smoothing_matches_iterative_mean_value_and_gradient() -> None:
    euler = torch.tensor(
        [
            [0.10, -0.20, 0.30],
            [0.25, -0.10, 0.45],
            [0.35, 0.05, 0.60],
            [0.20, 0.15, 0.75],
            [0.05, 0.10, 0.90],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    kernel = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float64)
    q = so3.from_euler(euler)
    trajectory = Trajectory(t=torch.arange(5, dtype=q.dtype), q=q)

    actual = smooth_trajectory(trajectory, kernel, kind="so3").q
    reference = _reference_iterative_mean(q, kernel, kind="so3")
    torch.testing.assert_close(actual, reference)

    coefficient = torch.linspace(-0.7, 1.1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grad = torch.autograd.grad((actual * coefficient).sum(), euler, retain_graph=True)[0]
    reference_grad = torch.autograd.grad((reference * coefficient).sum(), euler)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


def test_batched_smoothing_matches_looped_batches() -> None:
    generator = torch.Generator().manual_seed(32)
    euler = torch.randn((2, 3, 7, 3), generator=generator, dtype=torch.float64) * 0.3
    q = so3.from_euler(euler)
    t = torch.arange(7, dtype=q.dtype).expand(2, 3, 7)
    kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], dtype=q.dtype)

    batched = smooth_trajectory(Trajectory(t=t, q=q), kernel, kind="so3").q
    looped = torch.stack(
        [
            torch.stack(
                [
                    smooth_trajectory(
                        Trajectory(t=t[i, j], q=q[i, j]), kernel, kind="so3"
                    ).q
                    for j in range(q.shape[1])
                ]
            )
            for i in range(q.shape[0])
        ]
    )
    torch.testing.assert_close(batched, looped)


def test_se3_smoothing_uses_sclerp_and_preserves_other_channels() -> None:
    q = se3.identity(batch_shape=(5,), dtype=torch.float64)
    q = q.clone()
    q[:, 0] = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0], dtype=q.dtype)
    v = torch.randn((5, 6), dtype=q.dtype)
    trajectory = Trajectory(t=torch.arange(5, dtype=q.dtype), q=q, v=v)
    kernel = torch.ones(3, dtype=q.dtype)

    smoothed = smooth_trajectory(trajectory, kernel, kind="se3")
    reference = _reference_iterative_mean(q, kernel, kind="se3")

    torch.testing.assert_close(smoothed.q, reference)
    assert smoothed.v is v
    torch.testing.assert_close(smoothed.q[..., 3:].norm(dim=-1), torch.ones(5, dtype=q.dtype))


def test_antipodal_quaternion_samples_are_hemisphere_aligned() -> None:
    quaternion = so3.from_euler(torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64))
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=quaternion.dtype)
    q = signs.unsqueeze(-1) * quaternion
    trajectory = Trajectory(t=torch.arange(5, dtype=q.dtype), q=q)
    smoothed = smooth_trajectory(trajectory, torch.ones(3, dtype=q.dtype))
    torch.testing.assert_close(smoothed.q, quaternion.expand_as(q))
    assert torch.isfinite(smoothed.q).all()


def test_single_knot_trajectory_is_supported() -> None:
    q = so3.from_euler(torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64))
    trajectory = Trajectory(t=torch.zeros(1, dtype=q.dtype), q=q)
    smoothed = smooth_trajectory(trajectory, torch.ones(5, dtype=q.dtype))
    assert torch.equal(smoothed.q, q)
