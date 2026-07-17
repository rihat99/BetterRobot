"""Tangent-space autograd primitives shared by gradients and Jacobians."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from .variables import Values, VarSpec


def perturb_values(
    specs: tuple[VarSpec, ...],
    values: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
) -> Values:
    """Evaluate the product-manifold retraction ``values ⊕ deltas``."""
    return {spec.name: spec.retract(values[spec.name], deltas[spec.name]) for spec in specs}


def tangent_grad(
    f: Callable[[Values], torch.Tensor],
    specs: tuple[VarSpec, ...],
    values: Mapping[str, torch.Tensor],
    *,
    create_graph: bool = False,
    graph_inputs: Sequence[torch.Tensor] = (),
) -> Values:
    """Differentiate ``f(values ⊕ delta)`` at ``delta = 0`` per free block.

    Tensor outputs are summed, yielding independent per-batch gradients when
    the function is batch-separable. Fixed coordinates never enter autograd;
    returned tensors therefore use each block's reduced tangent dimension.
    """
    batch_shapes = tuple(spec.batch_shape(values[spec.name]) for spec in specs)
    if batch_shapes and any(shape != batch_shapes[0] for shape in batch_shapes[1:]):
        raise ValueError(f"All tangent_grad values must share a batch shape, got {batch_shapes}")
    _value, gradient = _tangent_value_and_grad_prevalidated(
        f,
        specs,
        values,
        batch_shape=batch_shapes[0] if batch_shapes else (),
        create_graph=create_graph,
        graph_inputs=graph_inputs,
    )
    return gradient


def _tangent_value_and_grad_prevalidated(
    f: Callable[[Values], torch.Tensor],
    specs: tuple[VarSpec, ...],
    values: Mapping[str, torch.Tensor],
    *,
    batch_shape: tuple[int, ...],
    create_graph: bool = False,
    graph_inputs: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, Values]:
    """Return ``f`` and its reduced tangent gradient without public validation.

    Solver entry points validate values once, then use this fixed-layout helper
    from their update loop. Retraction therefore goes through each block's
    prevalidated reduced-tangent path instead of repeating finite/manifold/bound
    host checks.
    """
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

    perturbed = {
        spec.name: spec._retract_prevalidated(
            values[spec.name],
            deltas[spec.name],
            batch_shape=batch_shape,
        )
        for spec in specs
    }
    output = f(perturbed)
    if output.numel() == 0:
        raise ValueError("tangent_grad requires a non-empty tensor output")
    computed = (
        torch.autograd.grad(
            output.sum(),
            active,
            create_graph=create_graph,
            allow_unused=True,
        )
        if active and output.requires_grad
        else ()
    )
    by_name = dict(zip(active_names, computed))
    graph_anchor = None
    if create_graph:
        anchors = [value.sum() * 0.0 for value in (*values.values(), *graph_inputs) if value.requires_grad]
        if anchors:
            graph_anchor = sum(anchors[1:], anchors[0])

    result: Values = {}
    for spec in specs:
        delta = deltas[spec.name]
        grad = by_name.get(spec.name)
        value = torch.zeros_like(delta) if grad is None else grad
        if graph_anchor is not None:
            value = value + graph_anchor
        result[spec.name] = value
    value = output if create_graph else output.detach()
    return value, result
