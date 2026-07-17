"""Differentiable point-set alignment."""

from __future__ import annotations

import torch


def _validate_point_sets(source: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("source and target must be torch.Tensor values")
    if source.ndim < 2 or target.ndim < 2 or source.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError(
            f"source and target must have shape (..., N, 3); got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[-2] != target.shape[-2]:
        raise ValueError(f"source and target need the same N; got {source.shape[-2]} and {target.shape[-2]}")
    if source.shape[-2] < 3:
        raise ValueError(f"at least 3 points are required; got {source.shape[-2]}")
    if not source.is_floating_point() or not target.is_floating_point():
        raise TypeError("source and target must be floating torch.Tensor values")
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("source and target must share dtype and device")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("source and target must contain only finite values")


def _normalized_fit_weights(
    source: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    weight_shape = source.shape[:-1]
    if weights is None:
        fit_weights = source.new_ones(weight_shape)
    else:
        if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
            raise TypeError("weights must be a floating torch.Tensor or None")
        if weights.dtype != source.dtype or weights.device != source.device:
            raise ValueError("weights must share source dtype and device")
        try:
            fit_weights = torch.broadcast_to(weights, weight_shape)
        except RuntimeError as exc:
            raise ValueError(f"weights must broadcast to {tuple(weight_shape)}; got {tuple(weights.shape)}") from exc

    if not bool(torch.isfinite(fit_weights).all()) or bool((fit_weights < 0).any()):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = fit_weights.sum(dim=-1, keepdim=True)
    if bool((weight_sum <= 0).any()):
        raise ValueError("weights must have a positive sum")
    return fit_weights / weight_sum


def umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    estimate_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a proper 3D similarity transform by weighted Umeyama alignment.

    Finds ``target ~= scale * (rotation @ source) + translation`` for point
    sets shaped ``(..., N, 3)``.  Leading batch dimensions of ``source`` and
    ``target`` broadcast; ``weights`` may be ``(N,)`` or ``(..., N)`` and
    broadcasts over the same batch.  Weights must be non-negative with a
    positive sum.  Set ``estimate_scale=False`` for a rigid (6-DoF) fit.

    Returns ``(scale, rotation, translation)`` with shapes ``(...)``,
    ``(..., 3, 3)``, and ``(..., 3)``.  The SVD reflection correction
    guarantees ``det(rotation) = +1``.
    """
    _validate_point_sets(source, target)
    source, target = torch.broadcast_tensors(source, target)
    normalized_weights = _normalized_fit_weights(source, weights)

    source_mean = (normalized_weights.unsqueeze(-1) * source).sum(dim=-2)
    target_mean = (normalized_weights.unsqueeze(-1) * target).sum(dim=-2)
    source_centered = source - source_mean.unsqueeze(-2)
    target_centered = target - target_mean.unsqueeze(-2)

    covariance = torch.einsum(
        "...n,...ni,...nj->...ij",
        normalized_weights,
        target_centered,
        source_centered,
    )
    u, singular_values, vh = torch.linalg.svd(covariance)
    uncorrected_det = torch.linalg.det(u @ vh)
    final_sign = torch.where(
        uncorrected_det < 0,
        -torch.ones_like(uncorrected_det),
        torch.ones_like(uncorrected_det),
    )
    correction = torch.cat(
        (torch.ones_like(singular_values[..., :2]), final_sign.unsqueeze(-1)),
        dim=-1,
    )
    rotation = (u * correction.unsqueeze(-2)) @ vh

    source_variance = (normalized_weights * source_centered.square().sum(dim=-1)).sum(dim=-1)
    if bool((source_variance <= torch.finfo(source.dtype).eps).any()):
        raise ValueError("source point set has zero weighted variance")
    if estimate_scale:
        scale = (singular_values * correction).sum(dim=-1) / source_variance
    else:
        scale = torch.ones_like(source_variance)
    rotated_mean = torch.einsum("...ij,...j->...i", rotation, source_mean)
    translation = target_mean - scale.unsqueeze(-1) * rotated_mean
    return scale, rotation, translation


__all__ = ["umeyama"]
