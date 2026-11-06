"""``Trajectory`` — a time-indexed sequence of robot configurations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..exceptions import ShapeError


@dataclass(frozen=True)
class Trajectory:
    """A sampled robot motion: configurations ``q`` and optional velocities ``v`` at ``times``.

    Shapes: ``q (T, nq)``, ``v (T, nv)`` or ``None``, ``times (T,)`` in seconds.
    Shapes are checked at construction; values are not.
    """

    q: torch.Tensor  #: ``(T, nq)`` configurations.
    times: torch.Tensor  #: ``(T,)`` sample times in seconds.
    v: torch.Tensor | None = None  #: ``(T, nv)`` generalized velocities, or ``None``.

    def __post_init__(self) -> None:
        if self.q.ndim != 2:
            raise ShapeError(f"q has shape {tuple(self.q.shape)}; expected (T, nq)")
        num_steps = self.q.shape[0]
        if tuple(self.times.shape) != (num_steps,):
            raise ShapeError(f"times has shape {tuple(self.times.shape)}; expected ({num_steps},)")
        if num_steps > 1 and not bool((self.times[1:] > self.times[:-1]).all()):
            raise ValueError("times must be strictly increasing")
        if self.v is not None and (self.v.ndim != 2 or self.v.shape[0] != num_steps):
            raise ShapeError(f"v has shape {tuple(self.v.shape)}; expected ({num_steps}, nv)")

    def __len__(self) -> int:
        """Number of samples ``T``."""
        return self.q.shape[0]

    def at(self, step: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(q (nq,), v (nv,) or None)`` at integer sample index ``step``."""
        return self.q[step], None if self.v is None else self.v[step]
