"""Batch alignment of public tensor inputs with the model's value batch ``I...``.

Every public boundary checks each tensor input once with the one validator, `lie._checks.check_tensor`
(contract 06), then aligns it here with `batch_shape` / `expand_batch`. Raw passes below trust the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..exceptions import ShapeError
from ..lie._checks import broadcast_batch

if TYPE_CHECKING:
    from .model import Model


def batch_shape(model: Model, name: str, value: torch.Tensor, event: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast of ``value``'s leading axes (before ``event``) with the model's value batch ``I...``."""
    batch = tuple(value.shape[: value.ndim - len(event)])
    for field in ("joint_placements", "body_inertias", "frame_placements"):
        values = getattr(model, field)
        value_batch = tuple(values.shape[:-2])
        try:
            batch = broadcast_batch(batch, value_batch)
        except ShapeError as error:
            raise ShapeError(f"cannot broadcast {name} batch {batch} with {field} batch {value_batch}") from error
    return batch


def expand_batch(name: str, value: torch.Tensor, batch: tuple[int, ...], event: tuple[int, ...]) -> torch.Tensor:
    """``value`` broadcast to ``(*batch, *event)`` as a view."""
    try:
        return torch.broadcast_to(value, (*batch, *event))
    except RuntimeError as error:
        raise ShapeError(f"cannot broadcast {name} {tuple(value.shape)} to batch {batch}") from error


__all__ = ["batch_shape", "expand_batch"]
