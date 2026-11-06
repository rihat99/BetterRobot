"""Public (reduced) and full coordinates of a model with mimic joints.

The public ``q`` and ``v`` omit mimic targets; the recursions run over every joint. ``full_configuration`` and
``full_tangent`` expand, ``reduced_columns`` and ``reduced_mass_matrix`` contract with the same affine map, which
is a pure function of the model's static mimic fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.model import Model


@dataclass(frozen=True)
class MimicPlan:
    """Affine map between public and full coordinates: ``q_full = q @ q_expansion.mT + q_offset``."""

    q_expansion: torch.Tensor  #: ``(nq_full, nq)``
    q_offset: torch.Tensor  #: ``(nq_full,)``
    v_expansion: torch.Tensor  #: ``(nv_full, nv)``


def build_plan(model: Model, dtype: torch.dtype, device: torch.device) -> MimicPlan:
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
    return model._plan("mimic", build_plan, dtype=dtype, device=device)


def full_configuration(model: Model, q: torch.Tensor) -> torch.Tensor:
    """``q: (B..., nq)`` with the mimic targets filled in, ``(B..., nq_full)``; ``q`` itself without mimics."""
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


__all__ = ["MimicPlan", "full_configuration", "full_tangent", "mimic_plan", "reduced_columns", "reduced_mass_matrix"]
