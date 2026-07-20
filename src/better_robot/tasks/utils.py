"""Shared private helpers for task facades."""

from __future__ import annotations

import torch

from ..optim import OptimizerInfo


def _hemisphere_align(
    q: torch.Tensor,
    coordinate_slices: tuple[slice, ...],
) -> torch.Tensor:
    """Align quaternion coordinate slices to the preceding time sample."""
    aligned = q.clone()
    for coordinate_slice in coordinate_slices:
        start = 0 if coordinate_slice.start is None else coordinate_slice.start
        stop = q.shape[-1] if coordinate_slice.stop is None else coordinate_slice.stop
        if stop - start != 4:
            continue
        quaternion = aligned[..., coordinate_slice]
        adjacent_dot = (quaternion[..., 1:, :] * quaternion[..., :-1, :]).sum(dim=-1)
        step_sign = torch.where(
            adjacent_dot < 0.0,
            -torch.ones_like(adjacent_dot),
            torch.ones_like(adjacent_dot),
        )
        first = torch.ones_like(quaternion[..., :1, 0])
        signs = torch.cat((first, step_sign), dim=-1).cumprod(dim=-1).unsqueeze(-1)
        aligned[..., coordinate_slice] = quaternion * signs
    return aligned


def _public_diagnostics(
    infos: tuple[OptimizerInfo, ...],
) -> tuple[int | torch.Tensor, bool | torch.Tensor, int | torch.Tensor]:
    """Return summed iterations and final public convergence diagnostics."""
    iterations = sum(
        (info.iterations for info in infos),
        torch.zeros_like(infos[0].iterations),
    )
    converged = infos[-1].converged
    status = infos[-1].status
    if iterations.ndim == 0:
        return int(iterations), bool(converged), int(status)
    return iterations, converged, status
