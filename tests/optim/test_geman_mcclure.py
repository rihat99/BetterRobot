"""Geman--McClure robust-kernel contract tests."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim import GemanMcClure, RobustKernel


def test_geman_mcclure_matches_closed_form_and_is_bounded() -> None:
    kernel = GemanMcClure(c=1.0)
    squared_norm = torch.tensor(
        [0.0, 0.01, 0.25, 1.0, 4.0, 1.0e6],
        dtype=torch.float64,
    )

    expected_rho = 0.5 * squared_norm / (1.0 + squared_norm)
    expected_weight = 1.0 / (1.0 + squared_norm).square()

    torch.testing.assert_close(kernel.rho(squared_norm), expected_rho)
    torch.testing.assert_close(kernel.weight(squared_norm), expected_weight)
    assert bool(torch.all(kernel.rho(squared_norm) >= 0.0))
    assert bool(torch.all(kernel.rho(squared_norm) < 0.5))
    assert bool(torch.all((kernel.weight(squared_norm) >= 0.0) & (kernel.weight(squared_norm) <= 1.0)))


def test_geman_mcclure_is_public_and_structural() -> None:
    assert isinstance(GemanMcClure(c=2.0), RobustKernel)


@pytest.mark.parametrize("c", [0.0, -1.0, float("inf"), float("nan")])
def test_geman_mcclure_rejects_invalid_scale(c: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GemanMcClure(c=c)
