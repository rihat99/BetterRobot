"""Torch-native ``LieOps`` — the only Lie backend after P10-D.

Forwards every method to :mod:`better_robot.lie._torch_native_backend`,
the pure-PyTorch SE3/SO3 implementation. The legacy PyPose backend was
removed in P10-D; the ``BR_LIE_BACKEND`` env var that previously chose
between them is no longer consulted.

The lazy import keeps ``backends/`` at the bottom of the dependency DAG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _backend():
    from ...lie import _torch_native_backend as _tn
    return _tn


class TorchNativeLieOps:
    """Concrete :class:`~better_robot.backends.protocol.LieOps`."""

    # ── SE3 ──────────────────────────────────────────────────────────

    def se3_compose(self, a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_compose(a, b)

    def se3_inverse(self, t: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_inverse(t)

    def se3_log(self, t: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_log(t)

    def se3_exp(self, v: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_exp(v)

    def se3_act(self, t: "torch.Tensor", p: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_act(t, p)

    def se3_adjoint(self, t: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_adjoint(t)

    def se3_adjoint_inv(self, t: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_adjoint_inv(t)

    def se3_normalize(self, t: "torch.Tensor") -> "torch.Tensor":
        return _backend().se3_normalize(t)

    # ── SO3 ──────────────────────────────────────────────────────────

    def so3_compose(self, a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_compose(a, b)

    def so3_inverse(self, q: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_inverse(q)

    def so3_log(self, q: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_log(q)

    def so3_exp(self, w: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_exp(w)

    def so3_act(self, q: "torch.Tensor", p: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_act(q, p)

    def so3_to_matrix(self, q: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_to_matrix(q)

    def so3_from_matrix(self, R: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_from_matrix(R)

    def so3_normalize(self, q: "torch.Tensor") -> "torch.Tensor":
        return _backend().so3_normalize(q)


__all__ = ["TorchNativeLieOps"]
