"""Public (reduced) coordinates of a model: the mimic expansion and the configuration manifold.

The public ``q`` and ``v`` omit mimic targets; the recursions run over every joint. ``full_configuration`` and
``full_tangent`` expand, ``reduced_columns`` and ``reduced_mass_matrix`` contract with the same affine map, which
is a pure function of the model's static mimic fields.

``integrate`` and ``difference`` are the retraction and the tangent of the public coordinates. The coordinates are
grouped by `JointManifold`: each group is one batched call of the manifold's own map over the gathered coordinate
columns, so every joint kind on a manifold shares one formula (record 0013). Mimic targets have no public
coordinates and take no part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ..exceptions import ShapeError
from ..lie._checks import broadcast_batch, check_tensor
from ..model.manifolds import EUCLIDEAN, JointManifold
from ..model.model import Model


@final
@dataclass(frozen=True)
class MimicPlan:
    """Affine map between public and full coordinates: ``q_full = q @ q_expansion.mT + q_offset``."""

    q_expansion: torch.Tensor  #: ``(nq_full, nq)``
    q_offset: torch.Tensor  #: ``(nq_full,)``
    v_expansion: torch.Tensor  #: ``(nv_full, nv)``


def build_mimic_plan(model: Model, dtype: torch.dtype, device: torch.device) -> MimicPlan:
    """The expansion matrices from ``mimic_source_ids``, ``mimic_multipliers`` and ``mimic_offsets``."""
    q_entries: list[tuple[int, int, float]] = []  # (full row, public column, weight)
    v_entries: list[tuple[int, int, float]] = []
    q_offset = [0.0] * model.nq_full
    for joint_id in range(model.njoints):
        full_q, full_v = model.idx_qs_full[joint_id], model.idx_vs_full[joint_id]
        source = model.mimic_source_ids[joint_id]
        if source != joint_id:
            q_entries.append((full_q, model.idx_qs[source], model.mimic_multipliers[joint_id]))
            v_entries.append((full_v, model.idx_vs[source], model.mimic_multipliers[joint_id]))
            q_offset[full_q] = model.mimic_offsets[joint_id]
            continue
        q_entries += [(full_q + k, model.idx_qs[joint_id] + k, 1.0) for k in range(model.nqs_full[joint_id])]
        v_entries += [(full_v + k, model.idx_vs[joint_id] + k, 1.0) for k in range(model.nvs_full[joint_id])]

    def matrix(entries: list[tuple[int, int, float]], shape: tuple[int, int]) -> torch.Tensor:
        result = torch.zeros(shape, dtype=dtype, device=device)
        rows, columns, weights = zip(*entries) if entries else ((), (), ())
        result[list(rows), list(columns)] = torch.tensor(weights, dtype=dtype, device=device)
        return result

    return MimicPlan(
        q_expansion=matrix(q_entries, (model.nq_full, model.nq)),
        q_offset=torch.tensor(q_offset, dtype=dtype, device=device),
        v_expansion=matrix(v_entries, (model.nv_full, model.nv)),
    )


def mimic_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> MimicPlan:
    """The model's `MimicPlan` in ``dtype`` on ``device`` (built once per model, dtype and device)."""
    return model._plan("mimic", build_mimic_plan, dtype=dtype, device=device)


def full_configuration(model: Model, q: torch.Tensor) -> torch.Tensor:
    """``q: (B..., nq)`` with the mimic targets filled in, ``(B..., nq_full)``; ``q`` itself without mimics.

    Raises ``TypeError``, ``ShapeError``, ``DtypeMismatchError`` or ``DeviceMismatchError`` for a ``q`` that is not a
    tensor of the model's dtype and device ending in ``nq``.
    """
    check_tensor("q", q, shape=(model.nq,), dtype=model.dtype, device=model.device)
    return full_configuration_raw(model, q)


def full_configuration_raw(model: Model, q: torch.Tensor) -> torch.Tensor:
    """`full_configuration` without input checks, for a checked ``q: (B..., nq) → (B..., nq_full)``."""
    if model.nq_full == model.nq:
        return q
    plan = mimic_plan(model, dtype=q.dtype, device=q.device)
    return q @ plan.q_expansion.mT + plan.q_offset


def full_tangent(model: Model, v: torch.Tensor) -> torch.Tensor:
    """``v: (B..., nv)`` (a velocity or acceleration) over every joint, ``(B..., nv_full)``."""
    if model.nv_full == model.nv:
        return v
    return v @ mimic_plan(model, dtype=v.dtype, device=v.device).v_expansion.mT


def reduced_columns(model: Model, full: torch.Tensor) -> torch.Tensor:
    """Full-tangent columns ``(B..., nv_full)`` (a generalized force, Jacobian rows) summed onto the public ``nv``."""
    if model.nv_full == model.nv:
        return full
    return full @ mimic_plan(model, dtype=full.dtype, device=full.device).v_expansion


def reduced_mass_matrix(model: Model, mass_full: torch.Tensor) -> torch.Tensor:
    """``(B..., nv_full, nv_full)`` projected onto the public tangent, ``G.T @ M @ G``."""
    if model.nv_full == model.nv:
        return mass_full
    expansion = mimic_plan(model, dtype=mass_full.dtype, device=mass_full.device).v_expansion
    return expansion.mT @ mass_full @ expansion


@final
@dataclass(frozen=True)
class ManifoldPlan:
    """Coordinate columns of each manifold on one device: ``(manifold, q columns (n, nq), v columns (n, nv))``."""

    device: torch.device
    groups: tuple[tuple[JointManifold, torch.Tensor, torch.Tensor], ...]

    def columns(self, manifold: JointManifold) -> tuple[torch.Tensor, torch.Tensor]:
        """``(q columns (n, manifold.nq), v columns (n, manifold.nv))`` long; ``n = 0`` when no joint lives there."""
        for candidate, q_columns, v_columns in self.groups:
            if candidate is manifold:
                return q_columns, v_columns
        empty = torch.zeros(0, dtype=torch.long, device=self.device)
        return empty.reshape(0, manifold.nq), empty.reshape(0, manifold.nv)


def build_manifold_plan(model: Model, dtype: torch.dtype, device: torch.device) -> ManifoldPlan:
    """Group the public coordinates by manifold (``dtype`` is unused: the columns are integers)."""
    columns: dict[JointManifold, tuple[list[list[int]], list[list[int]]]] = {}
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
    return model._plan("manifold", build_manifold_plan, dtype=dtype, device=device)


def euclidean_columns(model: Model) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q columns (n,), v columns (n,))``, long on the model's device: the public coordinates whose joints declare
    the ``EUCLIDEAN`` manifold, where ``integrate`` adds ``v`` to ``q``; the grouping ``integrate`` itself uses."""
    q_columns, v_columns = manifold_plan(model, dtype=model.dtype, device=model.device).columns(EUCLIDEAN)
    return q_columns.reshape(-1), v_columns.reshape(-1)


def _pair(model: Model, name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor, width_a: int, width_b: int):
    check_tensor(name_a, a, shape=(width_a,), dtype=model.dtype, device=model.device)
    check_tensor(name_b, b, shape=(width_b,), dtype=model.dtype, device=model.device)
    try:
        batch = broadcast_batch(a.shape[:-1], b.shape[:-1])
    except ShapeError as error:
        raise ShapeError(
            f"{name_a} batch {tuple(a.shape[:-1])} and {name_b} batch {tuple(b.shape[:-1])} do not broadcast"
        ) from error
    return a.expand(*batch, width_a), b.expand(*batch, width_b), batch


def integrate(model: Model, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q ⊕ v``: step ``q: (B..., nq)`` along ``v: (B..., nv)`` on the configuration manifold; ``(B..., nq)``."""
    q, v, _ = _pair(model, "q", q, "v", v, model.nq, model.nv)
    result = q.clone()
    for manifold, q_columns, v_columns in manifold_plan(model, dtype=q.dtype, device=q.device).groups:
        values = manifold.integrate(q[..., q_columns], v[..., v_columns])
        result = result.index_copy(-1, q_columns.reshape(-1), values.flatten(start_dim=-2))
    return result


def difference(model: Model, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """``q1 ⊖ q0``: the tangent ``(B..., nv)`` from ``q0`` to ``q1``, both ``(B..., nq)``."""
    q0, q1, batch = _pair(model, "q0", q0, "q1", q1, model.nq, model.nq)
    result = q0.new_zeros(*batch, model.nv)
    for manifold, q_columns, v_columns in manifold_plan(model, dtype=q0.dtype, device=q0.device).groups:
        values = manifold.difference(q0[..., q_columns], q1[..., q_columns])
        result = result.index_copy(-1, v_columns.reshape(-1), values.flatten(start_dim=-2))
    return result


__all__ = [
    "ManifoldPlan",
    "MimicPlan",
    "difference",
    "euclidean_columns",
    "full_configuration",
    "full_tangent",
    "integrate",
    "manifold_plan",
    "mimic_plan",
    "reduced_columns",
    "reduced_mass_matrix",
]
