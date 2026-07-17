"""P8: robust-kernel objective and IRLS-weight consistency."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from better_robot.optim.kernels.base import RobustKernel
from better_robot.optim.kernels.cauchy import Cauchy
from better_robot.optim.kernels.huber import Huber
from better_robot.optim.kernels.l2 import L2
from better_robot.optim.kernels.tukey import Tukey


@pytest.mark.parametrize(
    ("kernel", "squared_norms"),
    [
        pytest.param(L2(), (1.0e-6, 0.04, 1.0, 9.0), id="l2"),
        pytest.param(
            Huber(delta=0.7),
            (1.0e-6, 0.04, 0.25, 1.0, 9.0),
            id="huber-away-from-cutoff",
        ),
        pytest.param(
            Cauchy(c=1.3),
            (1.0e-6, 0.04, 0.64, 1.3**2, 4.0, 9.0),
            id="cauchy-including-c-squared",
        ),
        pytest.param(
            Tukey(c=1.5),
            (1.0e-6, 0.04, 0.64, 1.0, 4.0, 9.0),
            id="tukey-away-from-cutoff",
        ),
    ],
)
def test_weight_is_twice_rho_derivative_at_smooth_points(
    kernel: RobustKernel,
    squared_norms: Sequence[float],
) -> None:
    """The built-ins use normalized row weights, not raw ``rho'(s)``.

    Since the least-squares objective is ``rho(s)=s/2`` for L2 while its
    residual/Jacobian row weight is one, the contract is ``w(s)=2*rho'(s)``.
    The grid straddles every piecewise cutoff; Cauchy's smooth characteristic
    scale ``c²`` is included exactly.
    """
    s = torch.tensor(squared_norms, dtype=torch.float64, requires_grad=True)

    derivative = torch.autograd.grad(kernel.rho(s).sum(), s)[0]

    torch.testing.assert_close(
        kernel.weight(s.detach()),
        2.0 * derivative,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("kernel", "cutoff"),
    [
        pytest.param(Huber(delta=0.7), 0.7**2, id="huber-delta-squared"),
        pytest.param(Tukey(c=1.5), 1.5**2, id="tukey-c-squared"),
    ],
)
def test_piecewise_cutoff_has_consistent_one_sided_derivatives(
    kernel: RobustKernel,
    cutoff: float,
) -> None:
    """Both formulas meet the normalized IRLS derivative at the cutoff.

    A central difference would average the two pieces and could hide a bad
    branch. Checking both one-sided secants also avoids relying on whichever
    branch ``torch.where`` selects exactly at equality. Huber and Tukey are
    differentiable at their cutoffs, so no arbitrary subgradient is needed.
    """
    point = torch.tensor(cutoff, dtype=torch.float64)
    step = torch.tensor(max(1.0, cutoff) * 1.0e-6, dtype=torch.float64)
    rho_at_cutoff = kernel.rho(point)
    derivative_left = (rho_at_cutoff - kernel.rho(point - step)) / step
    derivative_right = (kernel.rho(point + step) - rho_at_cutoff) / step
    expected_derivative = 0.5 * kernel.weight(point)

    torch.testing.assert_close(
        derivative_left,
        expected_derivative,
        atol=1.0e-10,
        rtol=1.0e-4,
    )
    torch.testing.assert_close(
        derivative_right,
        expected_derivative,
        atol=1.0e-10,
        rtol=1.0e-4,
    )
