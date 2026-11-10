"""Every robust kernel honours the ``RobustKernel`` contract, checked by ``check_robust_kernel``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from better_robot import optim
from better_robot.optim import kernels
from better_robot.testing.kernels import check_robust_kernel

KERNELS = [getattr(kernels, name) for name in kernels.__all__ if name != "RobustKernel"]


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda kernel: kernel.__name__)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_honours_the_robust_kernel_contract(kernel: type, dtype: torch.dtype, seed: int) -> None:
    check_robust_kernel(kernel(), dtype=dtype, generator=torch.Generator().manual_seed(seed))


@dataclass(frozen=True)
class _HalfWeight(optim.RobustKernel):
    """The plain loss with ``weight = d rho / ds``: beside a plain term, LM would stop at the wrong minimum."""

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return 0.5 * squared_norm

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.full_like(squared_norm, 0.5)


def test_checker_rejects_the_single_derivative() -> None:
    with pytest.raises(AssertionError, match="2 d rho / ds"):
        check_robust_kernel(_HalfWeight(), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
