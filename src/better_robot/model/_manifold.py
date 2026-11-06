"""Configuration-manifold retraction and tangent: ``integrate`` and ``difference`` over the public coordinates.

The public coordinates are grouped by `Manifold`: each group is one batched call of the manifold's own map over the
gathered coordinate columns, so every joint kind on a manifold shares one formula (record 0013). Mimic targets have
no public coordinates and take no part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ._checks import check_tensor
from .manifolds import Manifold

if TYPE_CHECKING:
    from .model import Model


@dataclass(frozen=True)
class ManifoldPlan:
    """Coordinate columns of each manifold on one device: ``(manifold, q columns (n, nq), v columns (n, nv))``."""

    device: torch.device
    groups: tuple[tuple[Manifold, torch.Tensor, torch.Tensor], ...]

    def columns(self, manifold: Manifold) -> tuple[torch.Tensor, torch.Tensor]:
        """``(q columns (n, manifold.nq), v columns (n, manifold.nv))`` long; ``n = 0`` when no joint lives there."""
        for candidate, q_columns, v_columns in self.groups:
            if candidate is manifold:
                return q_columns, v_columns
        empty = torch.zeros(0, dtype=torch.long, device=self.device)
        return empty.reshape(0, manifold.nq), empty.reshape(0, manifold.nv)


def build_plan(model: Model, dtype: torch.dtype, device: torch.device) -> ManifoldPlan:
    """Group the public coordinates by manifold (``dtype`` is unused: the columns are integers)."""
    columns: dict[Manifold, tuple[list[list[int]], list[list[int]]]] = {}
    for joint_id, joint in enumerate(model.joints):
        if model.nqs[joint_id] == 0:
            continue
        manifold = joint.manifold
        q_columns, v_columns = columns.setdefault(manifold, ([], []))
        q_start, v_start = model.idx_qs[joint_id], model.idx_vs[joint_id]
        for block in range(joint.nq // manifold.nq):
            q_columns.append([q_start + block * manifold.nq + k for k in range(manifold.nq)])
            v_columns.append([v_start + block * manifold.nv + k for k in range(manifold.nv)])
    return ManifoldPlan(
        device=device,
        groups=tuple(
            (
                manifold,
                torch.tensor(q_columns, dtype=torch.long, device=device).reshape(-1, manifold.nq),
                torch.tensor(v_columns, dtype=torch.long, device=device).reshape(-1, manifold.nv),
            )
            for manifold, (q_columns, v_columns) in columns.items()
        ),
    )


def manifold_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> ManifoldPlan:
    """The model's `ManifoldPlan` on ``device`` (built once per model and device)."""
    return model._plan("manifold", build_plan, dtype=dtype, device=device)


def _pair(model: Model, name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor, width_a: int, width_b: int):
    check_tensor(name_a, a, shape=(width_a,), dtype=model.dtype, device=model.device)
    check_tensor(name_b, b, shape=(width_b,), dtype=model.dtype, device=model.device)
    batch = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    return a.expand(*batch, width_a), b.expand(*batch, width_b), batch


def integrate(model: Model, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q ⊕ v`` for ``q: (B..., nq)`` and ``v: (B..., nv)``; returns ``(B..., nq)``."""
    q, v, _ = _pair(model, "q", q, "v", v, model.nq, model.nv)
    result = q.clone()
    for manifold, q_columns, v_columns in manifold_plan(model, dtype=q.dtype, device=q.device).groups:
        values = manifold.integrate(q[..., q_columns], v[..., v_columns])
        result = result.index_copy(-1, q_columns.reshape(-1), values.flatten(start_dim=-2))
    return result


def difference(model: Model, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """``q1 ⊖ q0`` for ``q0, q1: (B..., nq)``; returns ``(B..., nv)``."""
    q0, q1, batch = _pair(model, "q0", q0, "q1", q1, model.nq, model.nq)
    result = q0.new_zeros(*batch, model.nv)
    for manifold, q_columns, v_columns in manifold_plan(model, dtype=q0.dtype, device=q0.device).groups:
        values = manifold.difference(q0[..., q_columns], q1[..., q_columns])
        result = result.index_copy(-1, v_columns.reshape(-1), values.flatten(start_dim=-2))
    return result


__all__ = ["ManifoldPlan", "build_plan", "difference", "integrate", "manifold_plan"]
