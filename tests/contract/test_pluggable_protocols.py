"""Conformance tests for the retained pluggable Protocols.

* ``LinearSolver`` — every concrete solver in ``optim.solvers`` satisfies
  ``isinstance(obj, LinearSolver)`` (``runtime_checkable`` protocol).
* ``RobustKernel`` — every concrete kernel satisfies the same contract.

See ``docs/conventions/extension.md §§4–6``.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.kernels.base import RobustKernel
from better_robot.optim.kernels.cauchy import Cauchy
from better_robot.optim.kernels.huber import Huber
from better_robot.optim.kernels.l2 import L2
from better_robot.optim.kernels.tukey import Tukey
from better_robot.optim.solvers.base import LinearSolver
from better_robot.optim.solvers.cholesky import Cholesky
from better_robot.optim.solvers.lstsq import LSTSQ


@pytest.mark.parametrize("cls", [Cholesky, LSTSQ])
def test_linear_solver_protocol(cls) -> None:
    assert isinstance(cls(), LinearSolver)


@pytest.mark.parametrize("cls", [L2, Huber, Cauchy, Tukey])
def test_robust_kernel_protocol(cls) -> None:
    assert isinstance(cls(), RobustKernel)


def test_kernel_rho_method_exists() -> None:
    """All kernels expose ``rho(s)`` returning the loss value at a squared norm."""
    s = torch.tensor([0.1, 1.0, 10.0])
    for k in (L2(), Huber(), Cauchy(), Tukey()):
        rho = k.rho(s)
        assert rho.shape == s.shape
        assert torch.isfinite(rho).all()
