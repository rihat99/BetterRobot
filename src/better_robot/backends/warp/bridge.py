"""``TorchArray`` / ``WarpBridge`` — translate between ``torch.Tensor`` and
``wp.array`` buffers.

Skeleton only. Real implementation lands in phase 7 (see
``docs/concepts/batching_and_backends.md §7``).
"""

from __future__ import annotations


class WarpBridge:
    """Zero-copy bridge between PyTorch tensors and Warp arrays.

    See docs/concepts/batching_and_backends.md §7.
    """

    def to_warp(self, t):
        raise NotImplementedError("see docs/concepts/batching_and_backends.md §7")

    def to_torch(self, a):
        raise NotImplementedError("see docs/concepts/batching_and_backends.md §7")
