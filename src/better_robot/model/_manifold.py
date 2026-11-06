"""Configuration-manifold retraction and tangent: ``integrate`` and ``difference`` over the public coordinates.

Joints are grouped by manifold (Euclidean, spherical, free flyer, unbounded revolute, planar); each group is one
batched Lie-group formula over gathered coordinate columns. A joint whose public width differs from its own
(a mimic target root) or whose class is not in a group falls back to the joint's own ``integrate``/``difference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..lie import se3, so3
from ._checks import check_tensor
from .joint_models import (
    JointFreeFlyer,
    JointHelical,
    JointPlanar,
    JointPrismaticUnaligned,
    JointPX,
    JointPY,
    JointPZ,
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
    JointRX,
    JointRY,
    JointRZ,
    JointSpherical,
    JointTranslation,
)

if TYPE_CHECKING:
    from .model import Model

_EUCLIDEAN = frozenset(
    {
        JointRX,
        JointRY,
        JointRZ,
        JointRevoluteUnaligned,
        JointPX,
        JointPY,
        JointPZ,
        JointPrismaticUnaligned,
        JointTranslation,
        JointHelical,
    }
)
_GROUPS = {
    "spherical": JointSpherical,
    "free_flyer": JointFreeFlyer,
    "unbounded": JointRevoluteUnbounded,
    "planar": JointPlanar,
}
_WIDTHS = {"spherical": (4, 3), "free_flyer": (7, 6), "unbounded": (2, 1), "planar": (4, 3)}


@dataclass(frozen=True)
class ManifoldPlan:
    """Coordinate columns of each manifold group on one device; ``q_*`` index ``q``, ``v_*`` index ``v``."""

    euclidean_q: torch.Tensor  #: ``(n_e,)`` long
    euclidean_v: torch.Tensor  #: ``(n_e,)`` long
    spherical_q: torch.Tensor  #: ``(n_s, 4)`` long
    spherical_v: torch.Tensor  #: ``(n_s, 3)`` long
    free_flyer_q: torch.Tensor  #: ``(n_f, 7)`` long
    free_flyer_v: torch.Tensor  #: ``(n_f, 6)`` long
    unbounded_q: torch.Tensor  #: ``(n_u, 2)`` long
    unbounded_v: torch.Tensor  #: ``(n_u, 1)`` long
    planar_q: torch.Tensor  #: ``(n_p, 4)`` long
    planar_v: torch.Tensor  #: ``(n_p, 3)`` long
    fallback: tuple[tuple[int, torch.Tensor, torch.Tensor], ...]  #: ``(joint id, q columns, v columns)`` per joint


def build_plan(model: Model, dtype: torch.dtype, device: torch.device) -> ManifoldPlan:
    """Group the public coordinates by manifold (``dtype`` is unused: the columns are integers)."""
    groups: dict[str, list[int]] = {"euclidean": [], "spherical": [], "free_flyer": [], "unbounded": [], "planar": []}
    fallback: list[int] = []
    for joint_id, joint in enumerate(model.joints):
        nq, nv = model.nqs[joint_id], model.nvs[joint_id]
        if nq == 0 and nv == 0:
            continue
        if nq != joint.nq or nv != joint.nv:
            fallback.append(joint_id)
        elif type(joint) in _EUCLIDEAN:
            groups["euclidean"].append(joint_id)
        else:
            group = next((name for name, cls in _GROUPS.items() if type(joint) is cls), None)
            (groups[group] if group else fallback).append(joint_id)

    def columns(joint_ids: list[int], starts: tuple[int, ...], widths: tuple[int, ...]) -> torch.Tensor:
        flat = [starts[j] + k for j in joint_ids for k in range(widths[j])]
        return torch.tensor(flat, dtype=torch.long, device=device)

    def matrix(name: str, starts: tuple[int, ...], width: int) -> torch.Tensor:
        return columns(groups[name], starts, (width,) * model.njoints).reshape(len(groups[name]), width)

    return ManifoldPlan(
        euclidean_q=columns(groups["euclidean"], model.idx_qs, model.nqs),
        euclidean_v=columns(groups["euclidean"], model.idx_vs, model.nvs),
        spherical_q=matrix("spherical", model.idx_qs, 4),
        spherical_v=matrix("spherical", model.idx_vs, 3),
        free_flyer_q=matrix("free_flyer", model.idx_qs, 7),
        free_flyer_v=matrix("free_flyer", model.idx_vs, 6),
        unbounded_q=matrix("unbounded", model.idx_qs, 2),
        unbounded_v=matrix("unbounded", model.idx_vs, 1),
        planar_q=matrix("planar", model.idx_qs, 4),
        planar_v=matrix("planar", model.idx_vs, 3),
        fallback=tuple(
            (j, columns([j], model.idx_qs, model.nqs), columns([j], model.idx_vs, model.nvs)) for j in fallback
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
    if model.nq == 0:
        return result
    plan = manifold_plan(model, dtype=q.dtype, device=q.device)
    if plan.euclidean_q.numel():
        result = result.index_copy(-1, plan.euclidean_q, q[..., plan.euclidean_q] + v[..., plan.euclidean_v])
    if plan.spherical_q.numel():
        values = so3.normalize(so3.compose(q[..., plan.spherical_q], so3.exp(v[..., plan.spherical_v])))
        result = result.index_copy(-1, plan.spherical_q.reshape(-1), values.flatten(start_dim=-2))
    if plan.free_flyer_q.numel():
        values = se3.normalize(se3.compose(q[..., plan.free_flyer_q], se3.exp(v[..., plan.free_flyer_v])))
        result = result.index_copy(-1, plan.free_flyer_q.reshape(-1), values.flatten(start_dim=-2))
    if plan.unbounded_q.numel():
        group = q[..., plan.unbounded_q]
        theta = torch.atan2(group[..., 1], group[..., 0]) + v[..., plan.unbounded_v][..., 0]
        values = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        result = result.index_copy(-1, plan.unbounded_q.reshape(-1), values.flatten(start_dim=-2))
    if plan.planar_q.numel():
        group, step = q[..., plan.planar_q], v[..., plan.planar_v]
        theta = torch.atan2(group[..., 3], group[..., 2]) + step[..., 2]
        values = torch.stack(
            (group[..., 0] + step[..., 0], group[..., 1] + step[..., 1], torch.cos(theta), torch.sin(theta)), dim=-1
        )
        result = result.index_copy(-1, plan.planar_q.reshape(-1), values.flatten(start_dim=-2))
    for joint_id, q_columns, v_columns in plan.fallback:
        values = model.joints[joint_id].integrate(q[..., q_columns], v[..., v_columns])
        result = result.index_copy(-1, q_columns, values)
    return result


def difference(model: Model, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """``q1 ⊖ q0`` for ``q0, q1: (B..., nq)``; returns ``(B..., nv)``."""
    q0, q1, batch = _pair(model, "q0", q0, "q1", q1, model.nq, model.nq)
    result = q0.new_zeros(*batch, model.nv)
    if model.nv == 0:
        return result
    plan = manifold_plan(model, dtype=q0.dtype, device=q0.device)
    if plan.euclidean_v.numel():
        result = result.index_copy(-1, plan.euclidean_v, q1[..., plan.euclidean_q] - q0[..., plan.euclidean_q])
    if plan.spherical_q.numel():
        delta = so3.compose(so3.inverse(q0[..., plan.spherical_q]), q1[..., plan.spherical_q])
        values = so3.log(so3.normalize(delta))
        result = result.index_copy(-1, plan.spherical_v.reshape(-1), values.flatten(start_dim=-2))
    if plan.free_flyer_q.numel():
        values = se3.log(se3.compose(se3.inverse(q0[..., plan.free_flyer_q]), q1[..., plan.free_flyer_q]))
        result = result.index_copy(-1, plan.free_flyer_v.reshape(-1), values.flatten(start_dim=-2))
    if plan.unbounded_q.numel():
        group0, group1 = q0[..., plan.unbounded_q], q1[..., plan.unbounded_q]
        theta0 = torch.atan2(group0[..., 1], group0[..., 0])
        theta1 = torch.atan2(group1[..., 1], group1[..., 0])
        result = result.index_copy(-1, plan.unbounded_v.reshape(-1), (theta1 - theta0).unsqueeze(-1).flatten(-2))
    if plan.planar_q.numel():
        group0, group1 = q0[..., plan.planar_q], q1[..., plan.planar_q]
        theta0 = torch.atan2(group0[..., 3], group0[..., 2])
        theta1 = torch.atan2(group1[..., 3], group1[..., 2])
        values = torch.stack(
            (group1[..., 0] - group0[..., 0], group1[..., 1] - group0[..., 1], theta1 - theta0), dim=-1
        )
        result = result.index_copy(-1, plan.planar_v.reshape(-1), values.flatten(start_dim=-2))
    for joint_id, q_columns, v_columns in plan.fallback:
        values = model.joints[joint_id].difference(q0[..., q_columns], q1[..., q_columns])
        result = result.index_copy(-1, v_columns, values)
    return result


__all__ = ["ManifoldPlan", "build_plan", "difference", "integrate", "manifold_plan"]
