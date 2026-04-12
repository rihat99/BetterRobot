"""Broadcasting helpers for residuals / kinematics.

See ``docs/10_BATCHING_AND_BACKENDS.md``.
"""

from __future__ import annotations

import torch


def broadcast_to_batch(t: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    """Broadcast ``t`` to the given leading batch shape.

    See docs/10_BATCHING_AND_BACKENDS.md.
    """
    raise NotImplementedError("see docs/10_BATCHING_AND_BACKENDS.md")
