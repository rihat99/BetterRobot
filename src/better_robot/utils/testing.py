"""Shared test utilities.

``assert_close_manifold``, ``fk_regression``, and other helpers used across
``tests_v2/``.

See ``docs/design/01_ARCHITECTURE.md``.
"""

from __future__ import annotations

import torch


def assert_close_manifold(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert two SE3 / SO3 tensors are close on the manifold.

    See docs/design/01_ARCHITECTURE.md.
    """
    raise NotImplementedError("see docs/design/01_ARCHITECTURE.md — utils/testing")
