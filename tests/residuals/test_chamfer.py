"""Masked Chamfer value, padding, batching, and correspondence-gradient tests."""

from __future__ import annotations

import torch

from better_robot.residuals.chamfer import MaskedChamferResidual


def test_masked_bidirectional_chamfer_matches_toy_reference_and_drops_empty_frame() -> None:
    residual = MaskedChamferResidual(
        2,
        2,
        3,
        vertex_weights="vertex_weights",
        chunk_size=1,
    )
    source = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.1, 0.0]],
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [2.0, 0.1, 0.0]],
        ],
        dtype=torch.float32,
    )
    source_validity = torch.ones(2, 2, dtype=torch.bool)
    target_validity = torch.tensor(
        [[True, True, False], [False, False, False]],
    )
    vertex_weights = torch.tensor([[2.0, 0.5], [1.0, 1.0]])

    actual = residual(
        {
            "points": source,
            "target_points": target,
            "point_validity": source_validity,
            "target_validity": target_validity,
            "vertex_weights": vertex_weights,
        }
    )
    expected = torch.tensor(
        [2.0, 0.5 * 5.0**0.5, 1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    assert residual.dim == 10
    torch.testing.assert_close(actual, expected)


def test_chamfer_batched_values_match_sequential_with_shared_padded_target() -> None:
    residual = MaskedChamferResidual(1, 2, 2, chunk_size=1)
    source = torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            [[[0.0, 0.5, 0.0], [2.0, -0.5, 0.0]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]], dtype=torch.float32)
    source_validity = torch.ones(2, 1, 2, dtype=torch.bool)
    target_validity = torch.ones(1, 2, dtype=torch.bool)
    batched = residual(
        {
            "points": source,
            "target_points": target,
            "point_validity": source_validity,
            "target_validity": target_validity,
        }
    )

    assert batched.shape == (2, 4)
    for index in range(2):
        sequential = residual(
            {
                "points": source[index],
                "target_points": target,
                "point_validity": source_validity[index],
                "target_validity": target_validity,
            }
        )
        torch.testing.assert_close(batched[index], sequential)


def test_chamfer_gradient_uses_selected_valid_correspondence_only() -> None:
    residual = MaskedChamferResidual(1, 1, 3, bidirectional=False, chunk_size=1)
    source = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    target = torch.tensor(
        [[[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.1, 0.0, 0.0]]],
        requires_grad=True,
    )
    value = residual(
        {
            "points": source,
            "target_points": target,
            "point_validity": torch.ones(1, 1, dtype=torch.bool),
            "target_validity": torch.tensor([[True, True, False]]),
        }
    )
    source_gradient, target_gradient = torch.autograd.grad(value.sum(), (source, target))

    torch.testing.assert_close(value, torch.ones(1))
    torch.testing.assert_close(source_gradient, torch.tensor([[[-1.0, 0.0, 0.0]]]))
    torch.testing.assert_close(target_gradient[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(target_gradient[0, 1:], torch.zeros(2, 3))


def test_chamfer_masked_nan_padding_has_zero_value_and_gradients() -> None:
    residual = MaskedChamferResidual(1, 1, 1, chunk_size=1)
    source = torch.zeros((1, 1, 3), requires_grad=True)
    target = torch.full((1, 1, 3), torch.nan, requires_grad=True)

    value = residual(
        {
            "points": source,
            "target_points": target,
            "point_validity": torch.ones(1, 1, dtype=torch.bool),
            "target_validity": torch.zeros(1, 1, dtype=torch.bool),
        }
    )
    source_gradient, target_gradient = torch.autograd.grad(
        value.sum(),
        (source, target),
    )

    torch.testing.assert_close(value, torch.zeros(2))
    assert torch.isfinite(source_gradient).all()
    assert torch.isfinite(target_gradient).all()
    torch.testing.assert_close(source_gradient, torch.zeros_like(source_gradient))
    torch.testing.assert_close(target_gradient, torch.zeros_like(target_gradient))
