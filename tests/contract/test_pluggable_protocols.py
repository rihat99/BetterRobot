"""Conformance tests for the three pluggable Protocols.

* ``LinearSolver`` — every concrete solver in ``optim.solvers`` satisfies
  ``isinstance(obj, LinearSolver)`` (``runtime_checkable`` protocol).
* ``DampingStrategy`` — same for ``optim.strategies``.
* ``RobustKernel`` — same for ``optim.kernels``.

See ``docs/conventions/15_EXTENSION.md §§4–6``.
"""

from __future__ import annotations

import pytest

from better_robot.optim.kernels.base import RobustKernel
from better_robot.optim.kernels.cauchy import Cauchy
from better_robot.optim.kernels.huber import Huber
from better_robot.optim.kernels.l2 import L2
from better_robot.optim.kernels.tukey import Tukey
from better_robot.optim.solvers.base import LinearSolver
from better_robot.optim.solvers.cg import CG
from better_robot.optim.solvers.cholesky import Cholesky
from better_robot.optim.solvers.lstsq import LSTSQ
from better_robot.optim.strategies.adaptive import Adaptive
from better_robot.optim.strategies.base import DampingStrategy
from better_robot.optim.strategies.constant import Constant
from better_robot.optim.strategies.trust_region import TrustRegion


@pytest.mark.parametrize("cls", [Cholesky, LSTSQ, CG])
def test_linear_solver_protocol(cls) -> None:
    assert isinstance(cls(), LinearSolver)


@pytest.mark.parametrize("cls", [Adaptive, Constant, TrustRegion])
def test_damping_strategy_protocol(cls) -> None:
    assert isinstance(cls(), DampingStrategy)


@pytest.mark.parametrize("cls", [L2, Huber, Cauchy, Tukey])
def test_robust_kernel_protocol(cls) -> None:
    assert isinstance(cls(), RobustKernel)


def test_kernel_rho_method_exists() -> None:
    """All kernels expose ``rho(s)`` returning the loss value at a squared norm."""
    import torch
    s = torch.tensor([0.1, 1.0, 10.0])
    for k in (L2(), Huber(), Cauchy(), Tukey()):
        rho = k.rho(s)
        assert rho.shape == s.shape
        assert torch.isfinite(rho).all()
