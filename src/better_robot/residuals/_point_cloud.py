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

from .._validation import check_tensor


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
    query = check_tensor("query", query, floating=True)
    reference = check_tensor("reference", reference, floating=True, dtype=query.dtype, device=query.device)
    if query.ndim < 2 or reference.ndim < 2:
        raise ValueError(
            "query and reference must have shape (..., count, coordinates), "
            f"got {tuple(query.shape)} and {tuple(reference.shape)}"
        )
    if query.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "query and reference points must have the same coordinate dimension, "
            f"got {query.shape[-1]} and {reference.shape[-1]}"
        )
    if reference.shape[-2] == 0:
        raise ValueError("the padded reference point axis must contain at least one slot")
    query_validity = check_tensor(
        "query_validity", query_validity, shape=(query.shape[-2],), dtype=torch.bool, device=query.device
    )
    reference_validity = check_tensor(
        "reference_validity",
        reference_validity,
        shape=(reference.shape[-2],),
        dtype=torch.bool,
        device=query.device,
    )
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

        # Detached scores own discrete choices; deltas retain the original graph.
        detached_query = torch.where(
            query_validity[..., start:stop].unsqueeze(-1),
            query_chunk.detach(),
            torch.zeros_like(query_chunk),
        )
        detached_reference = torch.where(
            reference_validity.unsqueeze(-1),
            reference.detach(),
            torch.zeros_like(reference),
        )
        detached_delta = detached_query.unsqueeze(-2) - detached_reference.unsqueeze(-3)
        score = detached_delta.square().sum(dim=-1)
        score = score.masked_fill(~reference_validity.unsqueeze(-2), torch.inf)
        index = score.argmin(dim=-1)
        selected = _gather_rows(reference, index)
        safe_query = torch.where(
            valid_chunk.unsqueeze(-1),
            query_chunk,
            torch.zeros_like(query_chunk),
        )
        safe_selected = torch.where(
            valid_chunk.unsqueeze(-1),
            selected,
            torch.zeros_like(selected),
        )
        delta = safe_query - safe_selected
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
