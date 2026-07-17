"""Manifold-aware kernel smoothing for pose trajectories."""

from __future__ import annotations

from typing import Literal

import torch

from ..lie import se3, so3
from .trajectory import Trajectory


def _hemisphere_align(q: torch.Tensor, *, start: int) -> torch.Tensor:
    quaternion = q[..., start : start + 4]
    adjacent_dot = (quaternion[..., 1:, :] * quaternion[..., :-1, :]).sum(dim=-1)
    step_sign = torch.where(
        adjacent_dot < 0,
        -torch.ones_like(adjacent_dot),
        torch.ones_like(adjacent_dot),
    )
    first = torch.ones_like(quaternion[..., :1, 0])
    signs = torch.cat((first, step_sign), dim=-1).cumprod(dim=-1).unsqueeze(-1)
    aligned_quaternion = quaternion * signs
    if start == 0:
        return aligned_quaternion
    return torch.cat((q[..., :start], aligned_quaternion), dim=-1)


def smooth_trajectory(
    trajectory: Trajectory,
    kernel: torch.Tensor,
    *,
    kind: Literal["auto", "so3", "se3"] = "auto",
) -> Trajectory:
    """Smooth a quaternion or SE3 :class:`Trajectory` on the manifold.

    ``kernel`` is a one-dimensional, odd-length sequence of non-negative
    weights (for example a box or sampled Gaussian kernel).  Boundary
    samples are replicated.  ``kind="so3"`` requires ``q.shape[-1] == 4``;
    ``kind="se3"`` requires the library pose layout ``[tx, ty, tz, qx, qy,
    qz, qw]``.  ``"auto"`` selects from those two feature sizes.

    The batch and time dimensions are vectorized.  Each window uses the
    consumer-compatible iterative weighted mean built from :func:`so3.slerp`
    or :func:`se3.sclerp`; only the small kernel dimension is traversed.
    Quaternion signs are first aligned along time to respect the scalar-last
    hemisphere-continuity contract.  ``t``, ``v``, ``a``, and ``tau`` are
    preserved; only ``q`` is smoothed.
    """
    q = trajectory.q
    if kind == "auto":
        if q.shape[-1] == 4:
            kind = "so3"
        elif q.shape[-1] == 7:
            kind = "se3"
        else:
            raise ValueError(
                f"kind='auto' requires quaternion (..., 4) or SE3 (..., 7) samples; got q.shape={tuple(q.shape)}"
            )
    elif kind not in ("so3", "se3"):
        raise ValueError(f"kind must be 'auto', 'so3', or 'se3'; got {kind!r}")
    expected_dim = 4 if kind == "so3" else 7
    if q.shape[-1] != expected_dim:
        raise ValueError(f"kind={kind!r} requires q.shape[-1] == {expected_dim}; got {q.shape[-1]}")

    kernel = torch.as_tensor(kernel, dtype=q.dtype, device=q.device)
    if kernel.ndim != 1 or kernel.numel() == 0 or kernel.numel() % 2 == 0:
        raise ValueError(f"kernel must be a non-empty odd-length 1D tensor; got {tuple(kernel.shape)}")
    if bool((kernel < 0).any()):
        raise ValueError("kernel weights must be non-negative")
    if bool(kernel.sum() <= 0):
        raise ValueError("kernel weights must have a positive sum")
    weights = kernel / kernel.sum()

    if q.shape[-2] == 0:
        raise ValueError("cannot smooth an empty trajectory")
    aligned = _hemisphere_align(q, start=0 if kind == "so3" else 3)
    num_knots = q.shape[-2]
    radius = kernel.numel() // 2
    offsets = torch.arange(-radius, radius + 1, device=q.device)
    centers = torch.arange(num_knots, device=q.device).unsqueeze(-1)
    indices = (centers + offsets).clamp(0, num_knots - 1)
    windows = aligned[..., indices, :]

    mean = windows[..., 0, :]
    accumulated = weights[0]
    interpolate = so3.slerp if kind == "so3" else se3.sclerp
    for index in range(1, weights.numel()):
        next_weight = weights[index]
        next_total = accumulated + next_weight
        alpha = next_weight / next_total.clamp_min(torch.finfo(q.dtype).tiny)
        candidate = interpolate(mean, windows[..., index, :], alpha)
        mean = torch.where(next_weight > 0.0, candidate, mean)
        accumulated = next_total

    identical = (windows == windows[..., :1, :]).all(dim=(-2, -1), keepdim=False)
    # Preserve bit-identical forward values without replacing the weighted
    # mean's gradient by the first replicated sample's gradient.
    exact_forward = windows[..., 0, :].detach() + (mean - mean.detach())
    smoothed_q = torch.where(identical.unsqueeze(-1), exact_forward, mean)
    return Trajectory(
        t=trajectory.t,
        q=smoothed_q,
        v=trajectory.v,
        a=trajectory.a,
        tau=trajectory.tau,
        extras=dict(trajectory.extras),
        metadata=dict(trajectory.metadata),
        model_id=trajectory.model_id,
    )


__all__ = ["smooth_trajectory"]
