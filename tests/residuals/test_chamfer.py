"""Masked Chamfer value, padding, batching, and correspondence-gradient tests."""

from __future__ import annotations

import pytest
import torch

from better_robot.residuals.chamfer import MaskedChamferResidual


def test_masked_bidirectional_chamfer_matches_toy_reference_and_drops_empty_frame() -> None:
    source = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    target = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.1, 0.0]],
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [2.0, 0.1, 0.0]],
        ]
    )
    item = MaskedChamferResidual(
        source,
        target,
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([[True, True, False], [False, False, False]]),
        vertex_weights=torch.tensor([[2.0, 0.5], [1.0, 1.0]]),
        chunk_size=1,
    )
    expected = torch.tensor([2.0, 0.5 * 5.0**0.5, 1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert item.dim == 10
    torch.testing.assert_close(item.error(), expected)


def test_chamfer_batched_values_match_sequential_with_shared_padded_target() -> None:
    source = torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            [[[0.0, 0.5, 0.0], [2.0, -0.5, 0.0]]],
        ]
    )
    target = torch.tensor([[[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]]])
    source_validity = torch.ones(2, 1, 2, dtype=torch.bool)
    target_validity = torch.ones(1, 2, dtype=torch.bool)
    batched = MaskedChamferResidual(
        source,
        target,
        source_validity,
        target_validity,
        chunk_size=1,
    ).error()

    assert batched.shape == (2, 4)
    for index in range(2):
        sequential = MaskedChamferResidual(
            source[index],
            target,
            source_validity[index],
            target_validity,
            chunk_size=1,
        ).error()
        torch.testing.assert_close(batched[index], sequential)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("target", torch.zeros(1, 3)),
        ("target_validity", torch.ones(1, dtype=torch.bool)),
    ],
)
def test_chamfer_rejects_missing_configured_frame_axis(argument: str, value: torch.Tensor) -> None:
    inputs = {
        "source": torch.zeros(2, 1, 3),
        "target": torch.zeros(2, 1, 3),
        "source_validity": torch.ones(2, 1, dtype=torch.bool),
        "target_validity": torch.ones(2, 1, dtype=torch.bool),
    }
    inputs[argument] = value

    with pytest.raises(ValueError, match=argument):
        MaskedChamferResidual(**inputs, bidirectional=False)


def test_chamfer_gradient_uses_selected_valid_correspondence_only() -> None:
    source = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    target = torch.tensor(
        [[[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.1, 0.0, 0.0]]],
        requires_grad=True,
    )
    value = MaskedChamferResidual(
        source,
        target,
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[True, True, False]]),
        bidirectional=False,
        chunk_size=1,
    ).error()
    source_gradient, target_gradient = torch.autograd.grad(value.sum(), (source, target))

    torch.testing.assert_close(value, torch.ones(1))
    torch.testing.assert_close(source_gradient, torch.tensor([[[-1.0, 0.0, 0.0]]]))
    torch.testing.assert_close(target_gradient[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(target_gradient[0, 1:], torch.zeros(2, 3))


def test_chamfer_masked_nan_padding_has_zero_value_and_gradients() -> None:
    source = torch.zeros((1, 1, 3), requires_grad=True)
    target = torch.full((1, 1, 3), torch.nan, requires_grad=True)
    value = MaskedChamferResidual(
        source,
        target,
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
        chunk_size=1,
    ).error()
    source_gradient, target_gradient = torch.autograd.grad(value.sum(), (source, target))

    torch.testing.assert_close(value, torch.zeros(2))
    assert torch.isfinite(source_gradient).all()
    assert torch.isfinite(target_gradient).all()
    torch.testing.assert_close(source_gradient, torch.zeros_like(source_gradient))
    torch.testing.assert_close(target_gradient, torch.zeros_like(target_gradient))
