"""Small tensor utilities shared by the named-block solvers."""

from __future__ import annotations

import torch

from .problem import Problem
from .variables import Values


def _batch_shape(values: Values, problem: Problem) -> tuple[int, ...]:
    """Infer leading solve-batch axes from the first variable block."""
    spec = problem.vars[0]
    value = values[spec.name]
    return tuple(value.shape[: value.ndim - len(spec.shape)])


def _blend_values(mask: torch.Tensor, yes: Values, no: Values) -> Values:
    """Select complete named values per batch element."""
    return {
        name: torch.where(
            mask.reshape((*mask.shape, *((1,) * (yes[name].ndim - mask.ndim)))),
            yes[name],
            no[name],
        )
        for name in yes
    }
