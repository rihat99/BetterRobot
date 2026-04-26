"""Primitive ops used by the torch-native backend.

Currently a trivial wrapper layer — real kernels live in ``kinematics/``,
``dynamics/``, ``collision/``. This module exists so the Warp backend has a
mirror location to drop replacement kernels into.

See ``docs/concepts/batching_and_backends.md §7``.
"""

from __future__ import annotations
