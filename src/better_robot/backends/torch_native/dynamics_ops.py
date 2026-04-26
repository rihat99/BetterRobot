"""Torch-native ``DynamicsOps`` — forwards to the existing dynamics skeletons.

Most paths still raise ``NotImplementedError`` until P11 lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ...data_model.data import Data
    from ...data_model.model import Model


class TorchNativeDynamicsOps:
    """Concrete :class:`~better_robot.backends.protocol.DynamicsOps`."""

    def rnea(
        self,
        model: "Model",
        data: "Data",
        q: "torch.Tensor",
        v: "torch.Tensor",
        a: "torch.Tensor",
    ) -> "torch.Tensor":
        from ...dynamics.rnea import rnea as _rnea
        return _rnea(model, data, q, v, a)

    def aba(
        self,
        model: "Model",
        data: "Data",
        q: "torch.Tensor",
        v: "torch.Tensor",
        tau: "torch.Tensor",
    ) -> "torch.Tensor":
        from ...dynamics.aba import aba as _aba
        return _aba(model, data, q, v, tau)

    def crba(
        self, model: "Model", data: "Data", q: "torch.Tensor",
    ) -> "torch.Tensor":
        from ...dynamics.crba import crba as _crba
        return _crba(model, data, q)

    def center_of_mass(
        self,
        model: "Model",
        data: "Data",
        q: "torch.Tensor",
        v: "torch.Tensor | None" = None,
        a: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        from ...dynamics.centroidal import center_of_mass as _com
        return _com(model, data, q, v, a)


__all__ = ["TorchNativeDynamicsOps"]
