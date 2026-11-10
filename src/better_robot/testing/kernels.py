"""The contract of a robust kernel, checked on one instance: shapes, ``rho(0) = 0``, ``rho`` never decreasing, and
``weight`` equal to twice the derivative of ``rho``, the factor the solvers assume."""

from __future__ import annotations

import torch

from ..optim import RobustKernel

_BATCHES = ((), (5,), (2, 3))


def check_robust_kernel(kernel: RobustKernel, *, dtype: torch.dtype, generator: torch.Generator) -> None:
    """Raise ``AssertionError`` unless ``kernel`` honours the `RobustKernel` contract in ``dtype`` on the CPU.

    Checks: ``rho`` and ``weight`` keep the dtype and return ``(B...)`` for squared norms of the batch shapes ``()``,
    ``(5,)`` and ``(2, 3)``; and, on ``s = 0`` and 64 squared norms drawn log-uniformly from ``[1e-4, 1e4]`` by
    ``generator``, both are finite, ``rho(0) == 0``, ``rho`` never decreases in ``s``, and ``weight(s)`` equals
    ``2 d rho / ds`` measured by autograd (the plain loss ``s / 2`` has weight 1). The tolerance is ``1e-5`` in
    float32 and ``1e-10`` in float64, relative to values above 1.
    """
    for batch in _BATCHES:
        ones = torch.ones(batch, dtype=dtype)
        for name, values in (("rho", kernel.rho(ones)), ("weight", kernel.weight(ones))):
            if values.dtype != dtype or values.shape != ones.shape:
                raise AssertionError(f"{name}{batch} is {tuple(values.shape)} {values.dtype}, expected {batch} {dtype}")

    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    exponents = 8.0 * torch.rand(64, generator=generator, dtype=dtype) - 4.0
    squared_norm = torch.cat((torch.zeros(1, dtype=dtype), 10.0**exponents)).sort().values.requires_grad_()
    rho = kernel.rho(squared_norm)
    (slope,) = torch.autograd.grad(rho.sum(), squared_norm)
    rho, weight = rho.detach(), kernel.weight(squared_norm.detach())
    finite = bool(rho.isfinite().all() and weight.isfinite().all())
    if not finite:
        raise AssertionError("rho or weight is not finite on squared norms in [0, 1e4]")
    at_zero = rho[0].item()
    if abs(at_zero) > tolerance:
        raise AssertionError(f"rho(0) is {at_zero:.3e}, expected 0")
    scale = rho.abs().clamp(min=1.0)
    decrease = ((rho[:-1] - rho[1:]) / scale[1:]).max().item()
    if decrease > tolerance:
        raise AssertionError(f"rho decreases by {decrease:.3e} between two growing squared norms")
    expected = 2.0 * slope
    error = ((weight - expected).abs() / expected.abs().clamp(min=1.0)).max().item()
    if error > tolerance:
        raise AssertionError(f"weight(s) == 2 d rho / ds: max error {error:.3e} > {tolerance:.1e}")


__all__ = ["check_robust_kernel"]
