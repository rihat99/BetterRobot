"""Tangent-space autograd primitives shared by gradients and Jacobians."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from .variables import Values, VarSpec


def perturb_values(
    specs: tuple[VarSpec, ...], values: Mapping[str, torch.Tensor], deltas: Mapping[str, torch.Tensor]
) -> Values:
    """Retract each named value by its reduced tangent delta."""
    return {spec.name: spec.retract(values[spec.name], deltas[spec.name]) for spec in specs}


def tangent_grad(
    f: Callable[[Values], torch.Tensor],
    specs: tuple[VarSpec, ...],
    values: Mapping[str, torch.Tensor],
    *,
    create_graph: bool = False,
    graph_inputs: Sequence[torch.Tensor] = (),
) -> Values:
    """Differentiate ``f(values ⊕ delta)`` at zero in reduced tangents."""
    batch_shapes = tuple(spec.batch_shape(values[spec.name]) for spec in specs)
    if batch_shapes and any(shape != batch_shapes[0] for shape in batch_shapes[1:]):
        raise ValueError(f"All tangent_grad values must share a batch shape, got {batch_shapes}")
    _value, gradient = _tangent_value_and_grad(
        f,
        specs,
        values,
        batch_shape=batch_shapes[0] if batch_shapes else (),
        create_graph=create_graph,
        graph_inputs=graph_inputs,
    )
    return gradient


def _tangent_value_and_grad(
    f: Callable[[Values], torch.Tensor],
    specs: tuple[VarSpec, ...],
    values: Mapping[str, torch.Tensor],
    *,
    batch_shape: tuple[int, ...],
    create_graph: bool = False,
    graph_inputs: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, Values]:
    deltas: Values = {}
    active: list[torch.Tensor] = []
    active_names: list[str] = []
    for spec in specs:
        delta = values[spec.name].new_zeros(*batch_shape, spec.free_dim)
        if spec.free_dim:
            delta.requires_grad_(True)
            active.append(delta)
            active_names.append(spec.name)
        deltas[spec.name] = delta

    perturbed = {spec.name: spec.retract(values[spec.name], deltas[spec.name]) for spec in specs}
    output = f(perturbed)
    if output.numel() == 0:
        raise ValueError("tangent_grad requires a non-empty tensor output")
    computed = (
        torch.autograd.grad(output.sum(), active, create_graph=create_graph, allow_unused=True)
        if active and output.requires_grad
        else ()
    )
    by_name = dict(zip(active_names, computed))
    anchors = [value.sum() * 0.0 for value in (*values.values(), *graph_inputs) if create_graph and value.requires_grad]
    graph_anchor = sum(anchors[1:], anchors[0]) if anchors else None

    result: Values = {}
    for spec in specs:
        delta = deltas[spec.name]
        grad = by_name.get(spec.name)
        value = torch.zeros_like(delta) if grad is None else grad
        if graph_anchor is not None:
            value = value + graph_anchor
        result[spec.name] = value
    return output if create_graph else output.detach(), result
