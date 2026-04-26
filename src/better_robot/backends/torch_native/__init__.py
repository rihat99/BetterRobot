"""``torch_native`` backend — pure-PyTorch implementations of the
``Backend`` Protocols. The default backend.

See ``docs/design/10_BATCHING_AND_BACKENDS.md §7``.
"""

from __future__ import annotations

from .dynamics_ops import TorchNativeDynamicsOps
from .kinematics_ops import TorchNativeKinematicsOps
from .lie_ops import TorchNativeLieOps


class TorchNativeBackend:
    """The default ``Backend`` — bundles the three torch-native ops bundles."""

    name: str = "torch_native"

    def __init__(self) -> None:
        self.lie = TorchNativeLieOps()
        self.kinematics = TorchNativeKinematicsOps()
        self.dynamics = TorchNativeDynamicsOps()


# Module-level singleton — what `default_backend()` returns by default.
BACKEND = TorchNativeBackend()


__all__ = ["TorchNativeBackend", "BACKEND"]
