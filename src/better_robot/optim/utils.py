"""Small tensor utilities shared by LM and implicit differentiation."""

from __future__ import annotations

from typing import TypeAlias

import torch

from .problem import Problem

_TensorValues: TypeAlias = dict[str, torch.Tensor]


def _batch_shape(values: _TensorValues, problem: Problem) -> tuple[int, ...]:
    """Infer leading solve-batch axes from the first variable block."""
    variable = problem.vars[0]
    return variable.batch_shape_of(values[variable.name])


def _blend_values(mask: torch.Tensor, yes: _TensorValues, no: _TensorValues) -> _TensorValues:
    """Select complete named values per batch element."""
    return {
        name: torch.where(
            mask.reshape((*mask.shape, *((1,) * (yes[name].ndim - mask.ndim)))),
            yes[name],
            no[name],
        )
        for name in yes
    }


def _state_coordinates(
    values: _TensorValues,
    problem: Problem,
    state_index: torch.Tensor,
) -> torch.Tensor:
    """Gather ambient coordinates corresponding to tangent-space bounds."""
    batch_shape = _batch_shape(values, problem)
    flat = torch.cat(
        tuple(values[variable.name].reshape(*batch_shape, -1) for variable in problem.vars),
        dim=-1,
    )
    gathered = flat.index_select(-1, state_index.clamp(min=0))
    return torch.where(state_index >= 0, gathered, torch.zeros_like(gathered))
