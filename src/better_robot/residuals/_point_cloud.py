"""Shared fixed-shape point-cloud nearest-correspondence helpers.

The padded point-cloud convention is ``(points, validity_mask)`` with shapes
``(..., count, coordinates)`` and ``(..., count)``.  Correspondence indices
are selected from detached distances, then the continuous delta and distance
are reconstructed from the original tensors so gradients flow through the
matched points but never through ``argmin``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _NearestCorrespondence:
    """Fixed-shape nearest-neighbour result for every padded query row."""

    delta: torch.Tensor
    distance: torch.Tensor
    index: torch.Tensor
    valid: torch.Tensor


def _validate_clouds(
    query: torch.Tensor,
    reference: torch.Tensor,
    query_validity: torch.Tensor,
    reference_validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(query, torch.Tensor) or not query.is_floating_point() or query.ndim < 2:
        raise TypeError("query points must be a floating tensor with shape (..., count, coordinates)")
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point() or reference.ndim < 2:
        raise TypeError("reference points must be a floating tensor with shape (..., count, coordinates)")
    if query.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "query and reference points must have the same coordinate dimension, "
            f"got {query.shape[-1]} and {reference.shape[-1]}"
        )
    if query.dtype != reference.dtype or query.device != reference.device:
        raise ValueError(
            "query and reference points must share dtype/device, "
            f"got {query.dtype}/{query.device} and {reference.dtype}/{reference.device}"
        )
    if reference.shape[-2] == 0:
        raise ValueError("the padded reference point axis must contain at least one slot")
    for label, mask, count, device in (
        ("query_validity", query_validity, query.shape[-2], query.device),
        ("reference_validity", reference_validity, reference.shape[-2], reference.device),
    ):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim < 1:
            raise TypeError(f"{label} must be a bool tensor with shape (..., {count})")
        if mask.shape[-1] != count:
            raise ValueError(f"{label} must end in ({count},), got {tuple(mask.shape)}")
        if mask.device != device:
            raise ValueError(f"{label} must be on device {device}, got {mask.device}")

    try:
        prefix = torch.broadcast_shapes(
            tuple(query.shape[:-2]),
            tuple(reference.shape[:-2]),
            tuple(query_validity.shape[:-1]),
            tuple(reference_validity.shape[:-1]),
        )
    except RuntimeError as exc:
        raise ValueError("point clouds and validity masks have incompatible leading shapes") from exc

    coordinates = query.shape[-1]
    query = query.expand(*prefix, query.shape[-2], coordinates)
    reference = reference.expand(*prefix, reference.shape[-2], coordinates)
    query_validity = query_validity.expand(*prefix, query.shape[-2])
    reference_validity = reference_validity.expand(*prefix, reference.shape[-2])
    return query, reference, query_validity, reference_validity


def _gather_rows(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``table[..., index, :]`` with broadcasted leading axes."""
    if table.ndim < 2 or index.ndim < 1:
        raise ValueError("table/index must have shapes (..., rows, width) and (..., queries)")
    if table.device != index.device:
        raise ValueError("table and index must be on the same device")
    try:
        prefix = torch.broadcast_shapes(tuple(table.shape[:-2]), tuple(index.shape[:-1]))
    except RuntimeError as exc:
        raise ValueError("table and index have incompatible leading shapes") from exc
    table = table.expand(*prefix, table.shape[-2], table.shape[-1])
    index = index.expand(*prefix, index.shape[-1])
    gather_index = index[..., None].expand(*index.shape, table.shape[-1])
    return table.gather(-2, gather_index)


def _detached_nearest(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_validity: torch.Tensor,
    reference_validity: torch.Tensor,
    chunk_size: int,
) -> _NearestCorrespondence:
    """Match padded query rows while keeping the correspondence non-differentiable."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    query, reference, query_validity, reference_validity = _validate_clouds(
        query,
        reference,
        query_validity,
        reference_validity,
    )
    count = query.shape[-2]
    has_reference = reference_validity.any(dim=-1, keepdim=True)
    deltas: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    validities: list[torch.Tensor] = []

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        query_chunk = query[..., start:stop, :]
        valid_chunk = query_validity[..., start:stop] & has_reference

        # The detached score tensor owns all discrete choices.  Continuous
        # deltas are rebuilt below from the original graph-carrying tensors.
        detached_delta = query_chunk.detach().unsqueeze(-2) - reference.detach().unsqueeze(-3)
        score = detached_delta.square().sum(dim=-1)
        score = score.masked_fill(~reference_validity.unsqueeze(-2), torch.inf)
        index = score.argmin(dim=-1)
        selected = _gather_rows(reference, index)
        delta = query_chunk - selected
        distance = torch.linalg.vector_norm(delta, dim=-1)

        deltas.append(torch.where(valid_chunk.unsqueeze(-1), delta, torch.zeros_like(delta)))
        distances.append(torch.where(valid_chunk, distance, torch.zeros_like(distance)))
        indices.append(index)
        validities.append(valid_chunk)

    return _NearestCorrespondence(
        delta=torch.cat(deltas, dim=-2),
        distance=torch.cat(distances, dim=-1),
        index=torch.cat(indices, dim=-1),
        valid=torch.cat(validities, dim=-1),
    )


__all__: list[str] = []
