"""Regression tests for robust-kernel IRLS and LM acceptance."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.kernels.cauchy import Cauchy
from better_robot.optim.kernels.huber import Huber
from better_robot.optim.kernels.l2 import L2
from better_robot.optim.kernels.tukey import Tukey
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt


class _LocationProblem:
    """One-parameter location fit whose L2 optimum is outlier-biased."""

    cost_stack = None
    x0 = torch.tensor([25.0], dtype=torch.float64)
    lower = None
    upper = None
    nv = 1
    _nv = 1
    samples = torch.tensor([0.0, 0.1, -0.1, 100.0], dtype=torch.float64)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return x[0] - self.samples

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((self.samples.numel(), 1), dtype=x.dtype, device=x.device)

    def step(self, x: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor:
        return x + delta_v


class _RawAcceptanceHuber(Huber):
    """Reproduce the pre-M0 bug: Huber steps accepted against raw L2."""

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return 0.5 * squared_norm


class _CountingHuber(Huber):
    def __init__(self, *, delta: float) -> None:
        super().__init__(delta=delta)
        self.rho_calls = 0

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        self.rho_calls += 1
        return super().rho(squared_norm)


@pytest.mark.parametrize(
    ("kernel", "squared_norm"),
    [
        (L2(), [0.04, 0.64, 4.0]),
        (Huber(delta=0.7), [0.04, 0.25, 1.0]),
        (Cauchy(c=1.3), [0.04, 0.64, 4.0]),
        (Tukey(c=1.5), [0.04, 0.64, 4.0]),
    ],
)
def test_kernel_weight_matches_rho_derivative(kernel, squared_norm) -> None:
    """Built-ins use the normalized IRLS convention ``w = 2 rho'(s)``."""
    s = torch.tensor(squared_norm, dtype=torch.float64, requires_grad=True)

    derivative = torch.autograd.grad(kernel.rho(s).sum(), s)[0]

    torch.testing.assert_close(kernel.weight(s.detach()), 2.0 * derivative)


def test_huber_lm_accepts_on_robust_rho() -> None:
    """Robust acceptance escapes the raw-L2 optimum and fits the inliers."""
    problem = _LocationProblem()
    optimizer = LevenbergMarquardt(tol=1e-8)

    # At x0 (the sample mean), every move raises raw L2. The old mixed
    # objective therefore rejected every Huber-IRLS step and exhausted its
    # budget without moving.
    raw_acceptance = optimizer.minimize(
        problem,
        max_iter=20,
        kernel=_RawAcceptanceHuber(delta=1.0),
    )
    assert raw_acceptance.status == "maxiter"
    torch.testing.assert_close(raw_acceptance.x, problem.x0)

    kernel = _CountingHuber(delta=1.0)
    robust = optimizer.minimize(problem, max_iter=30, kernel=kernel)

    assert kernel.rho_calls >= 2
    assert robust.status == "converged"
    torch.testing.assert_close(
        robust.x,
        torch.tensor([1.0 / 3.0], dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )

    initial_residual = problem.residual(problem.x0)
    initial_robust_cost = kernel.rho(initial_residual.square()).sum()
    final_robust_cost = kernel.rho(robust.residual.square()).sum()
    initial_raw_cost = 0.5 * initial_residual.square().sum()
    final_raw_cost = 0.5 * robust.residual.square().sum()

    assert final_robust_cost < initial_robust_cost
    assert final_raw_cost > initial_raw_cost
    torch.testing.assert_close(robust.residual_norm, final_raw_cost)
