"""``Trajectory`` — first-class temporal type with shapes ``(B, T, nq)``.

See ``docs/08_TASKS.md §2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_model.data import Data
    from ..data_model.model import Model


@dataclass
class Trajectory:
    """Temporal batch of configurations, velocities, accelerations, controls.

    Shapes: ``t (B, T)``, ``q (B, T, nq)``, ``v (B, T, nv)``, ``a (B, T, nv)``,
    ``u (B, T, nu)``.
    """

    t: torch.Tensor
    q: torch.Tensor
    v: torch.Tensor | None = None
    a: torch.Tensor | None = None
    u: torch.Tensor | None = None
    model_id: int = -1

    @property
    def batch_size(self) -> int:
        return int(self.q.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.q.shape[1])

    def slice(self, t_start: float, t_end: float) -> "Trajectory":
        """Return a sub-trajectory over the given time window.

        See docs/08_TASKS.md §2.
        """
        raise NotImplementedError("see docs/08_TASKS.md §2")

    def resample(self, new_t: torch.Tensor) -> "Trajectory":
        """Return a trajectory resampled onto ``new_t``.

        See docs/08_TASKS.md §2.
        """
        raise NotImplementedError("see docs/08_TASKS.md §2")

    def to_data(self, model: "Model") -> "Data":
        """Materialise a batched ``Data`` from the trajectory.

        See docs/08_TASKS.md §2.
        """
        raise NotImplementedError("see docs/08_TASKS.md §2")
