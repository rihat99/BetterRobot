"""Right Jacobian of SE(3) parity with Pinocchio's ``Jexp6``.

Pinocchio's ``Jexp6(xi)`` returns ``Jr_SE3(xi)`` (6,6) — the right Jacobian
of the exponential map. BetterRobot exposes this as
``lie.tangents.right_jacobian_se3``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from better_robot.lie.tangents import right_jacobian_se3

pin = pytest.importorskip("pinocchio")


def _random_xi(seed: int, magnitude: float = 0.5) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return (torch.rand(6, generator=rng, dtype=torch.float64) - 0.5) * 2.0 * magnitude


@pytest.mark.parametrize("seed", list(range(8)))
def test_right_jacobian_se3_matches(seed):
    xi = _random_xi(seed)
    Jr_br = right_jacobian_se3(xi).detach().cpu().double().numpy()

    motion_pin = pin.Motion(xi[:3].numpy(), xi[3:].numpy())
    Jr_pin = pin.Jexp6(motion_pin)

    np.testing.assert_allclose(Jr_br, Jr_pin, atol=1e-9, rtol=1e-7)


def test_right_jacobian_se3_small_angle():
    """At ``xi → 0``, ``Jr_SE3`` should approach the identity."""
    xi = torch.zeros(6, dtype=torch.float64)
    Jr = right_jacobian_se3(xi).detach().cpu().double().numpy()
    np.testing.assert_allclose(Jr, np.eye(6), atol=1e-10)
