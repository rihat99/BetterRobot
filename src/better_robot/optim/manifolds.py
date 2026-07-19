"""State-coordinate bounds shared by optimization variables."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Bounds:
    """Validated lower and upper box limits in full state coordinates."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.lower, torch.Tensor) or not isinstance(self.upper, torch.Tensor):
            raise TypeError("Bounds lower and upper must be torch.Tensor values")
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"Bounds lower/upper shapes must match, got {tuple(self.lower.shape)} and {tuple(self.upper.shape)}"
            )
        if self.lower.dtype != self.upper.dtype or self.lower.device != self.upper.device:
            raise ValueError("Bounds lower and upper must have the same dtype and device")
        if not self.lower.is_floating_point():
            raise TypeError("Bounds lower and upper must use a floating dtype")
        if bool(torch.isnan(self.lower).any()) or bool(torch.isnan(self.upper).any()):
            raise ValueError("Bounds lower and upper must not contain NaN")
        if bool((self.lower > self.upper).any()):
            raise ValueError("Bounds lower must be <= upper at every state coordinate")


__all__ = ["Bounds"]
