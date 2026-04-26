"""Leading-batch helpers.

See ``docs/design/10_BATCHING_AND_BACKENDS.md``.
"""

from __future__ import annotations

import torch


def batch_shape(t: torch.Tensor, feature_dims: int) -> tuple[int, ...]:
    """Return ``t.shape[:-feature_dims]`` as a tuple.

    See docs/design/10_BATCHING_AND_BACKENDS.md.
    """
    return tuple(t.shape[:-feature_dims]) if feature_dims > 0 else tuple(t.shape)


def flatten_batch(t: torch.Tensor, feature_dims: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten leading batch dims into one. Returns ``(flat, original_shape)``.

    See docs/design/10_BATCHING_AND_BACKENDS.md.
    """
    raise NotImplementedError("see docs/design/10_BATCHING_AND_BACKENDS.md")
