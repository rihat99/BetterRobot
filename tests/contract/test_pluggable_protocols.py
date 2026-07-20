"""Conformance tests for the retained pluggable Protocols.

* ``LinearSolver`` — every concrete solver in ``optim.solvers`` satisfies
  ``isinstance(obj, LinearSolver)`` (``runtime_checkable`` protocol).
* ``RobustKernel`` — every concrete kernel satisfies the same contract.

See ``docs/conventions/extension.md §§4–6``.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.optim import (
    BandedCholesky,
    Cauchy,
    Cholesky,
    GemanMcClure,
    Huber,
    InformativeLinearSolver,
    L2,
    LinearSolver,
    RobustKernel,
    Tukey,
)


@pytest.mark.parametrize("cls", [Cholesky, BandedCholesky])
def test_linear_solver_protocol(cls) -> None:
    assert isinstance(cls(), LinearSolver)
    assert isinstance(cls(), InformativeLinearSolver)


@pytest.mark.parametrize("cls", [L2, Huber, Cauchy, Tukey, GemanMcClure])
def test_robust_kernel_protocol(cls) -> None:
    assert isinstance(cls(), RobustKernel)


def test_kernel_rho_method_exists() -> None:
    """All kernels expose ``rho(s)`` returning the loss value at a squared norm."""
    s = torch.tensor([0.1, 1.0, 10.0])
    for k in (L2(), Huber(), Cauchy(), Tukey(), GemanMcClure()):
        rho = k.rho(s)
        assert rho.shape == s.shape
        assert torch.isfinite(rho).all()
