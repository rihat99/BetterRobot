"""Thin ``torch.optim`` adapter for named product-manifold problems."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import math
from numbers import Real
from typing import NamedTuple, TypeAlias

import torch

from .problem import Problem, Weight
from .variables import Values, detach_values

OptimizerFactory: TypeAlias = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


class FirstOrderResult(NamedTuple):
    step: torch.Tensor
    converged: torch.Tensor
    cost: torch.Tensor


def _gradient_norm(buffers: Mapping[str, torch.Tensor], zero_gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    norms = []
    for name, buffer in buffers.items():
        if not buffer.numel():
            continue
        if buffer.grad is None:
            buffer.grad = zero_gradients[name]
        norms.append(buffer.grad.detach().abs().amax(dim=-1))
    return torch.stack(norms, dim=-1).amax(dim=-1)


def run_first_order(
    values: Mapping[str, torch.Tensor],
    problem: Problem,
    optimizer_factory: OptimizerFactory,
    *,
    max_iter: int,
    tolerance: float,
    weights: Mapping[str, Weight] | None = None,
) -> tuple[Values, FirstOrderResult]:
    """Run a persistent ``torch.optim`` instance in rebased tangent charts."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative int")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(float(tolerance))
        or tolerance < 0.0
    ):
        raise ValueError("tolerance must be a finite non-negative number")
    if not problem.residuals or problem.tangent_dim_total <= 0:
        raise ValueError("first-order optimization requires a residual and a free tangent coordinate")

    batch_shape = problem._validate_values(values)
    current = detach_values(dict(values))
    buffers = {
        spec.name: torch.nn.Parameter(current[spec.name].new_zeros(*batch_shape, spec.free_dim))
        for spec in problem.vars
    }
    zero_gradients = {name: torch.zeros_like(buffer) for name, buffer in buffers.items()}
    optimizer = optimizer_factory(buffers.values())
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_factory must return a torch.optim.Optimizer")

    exemplar = current[problem.vars[0].name]
    step = exemplar.new_zeros(batch_shape, dtype=torch.int64)
    converged = exemplar.new_zeros(batch_shape, dtype=torch.bool)
    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        cost = problem.objective(problem.retract(current, buffers), weights=weights)
        (cost * ~converged).sum().backward()
        converged = converged | (_gradient_norm(buffers, zero_gradients) <= tolerance)
        active = ~converged
        if bool(converged.all()):  # bench-ok: eager driver early-exit boundary
            break

        optimizer.step()
        mask = active.unsqueeze(-1)
        tangent = {name: buffer.detach() * mask for name, buffer in buffers.items()}
        current = detach_values(problem.retract(current, tangent))
        step = step + active.to(dtype=step.dtype)
        with torch.no_grad():
            for buffer in buffers.values():
                buffer.zero_()

    optimizer.zero_grad(set_to_none=True)
    cost = problem.objective(problem.retract(current, buffers), weights=weights)
    cost.sum().backward()
    converged = converged | (_gradient_norm(buffers, zero_gradients) <= tolerance)
    return detach_values(current), FirstOrderResult(step.detach(), converged.detach(), cost.detach())


__all__ = ["FirstOrderResult", "OptimizerFactory", "run_first_order"]
