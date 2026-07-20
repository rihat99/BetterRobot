"""Internal manifold operations for :class:`~better_robot.data_model.Model`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..exceptions import DeviceMismatchError, ShapeError
from ..lie import se3, so3

if TYPE_CHECKING:
    from .model import Model


def _validate_manifold_tensor(
    model: Model,
    name: str,
    value: torch.Tensor,
    trailing_width: int,
) -> None:
    if value.ndim < 1 or value.shape[-1] != trailing_width:
        raise ShapeError(f"{name} has shape {tuple(value.shape)}; expected trailing dimension {trailing_width}")
    model_device = model.structure.idx_qs_tensor.device
    if value.device != model_device:
        raise DeviceMismatchError(
            f"{name}.device={value.device} != model.device={model_device}; move the tensor or call model.to(...) first"
        )


def integrate(model: Model, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the universal manifold retraction ``q ⊕ v``."""

    _validate_manifold_tensor(model, "q", q, model.nq)
    _validate_manifold_tensor(model, "v", v, model.nv)
    batch_shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    result_dtype = torch.promote_types(q.dtype, v.dtype)
    q_broadcast = q.to(dtype=result_dtype).expand(*batch_shape, model.nq)
    v_broadcast = v.to(dtype=result_dtype).expand(*batch_shape, model.nv)
    if model.nq == 0:
        return q_broadcast.clone()

    result = q_broadcast.clone()
    structure = model.structure

    q_indices = structure.manifold_euclidean_q_indices
    if q_indices.numel():
        values = q_broadcast[..., q_indices] + v_broadcast[..., structure.manifold_euclidean_v_indices]
        result = result.index_copy(-1, q_indices, values)

    q_indices = structure.manifold_spherical_q_indices
    if q_indices.numel():
        q_group = q_broadcast[..., q_indices]
        v_group = v_broadcast[..., structure.manifold_spherical_v_indices]
        values = so3.normalize(so3.compose(q_group, so3.exp(v_group)))
        result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

    q_indices = structure.manifold_free_flyer_q_indices
    if q_indices.numel():
        q_group = q_broadcast[..., q_indices]
        v_group = v_broadcast[..., structure.manifold_free_flyer_v_indices]
        values = se3.normalize(se3.compose(q_group, se3.exp(v_group)))
        result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

    q_indices = structure.manifold_unbounded_q_indices
    if q_indices.numel():
        q_group = q_broadcast[..., q_indices]
        v_group = v_broadcast[..., structure.manifold_unbounded_v_indices]
        theta = torch.atan2(q_group[..., 1], q_group[..., 0]) + v_group[..., 0]
        values = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

    q_indices = structure.manifold_planar_q_indices
    if q_indices.numel():
        q_group = q_broadcast[..., q_indices]
        v_group = v_broadcast[..., structure.manifold_planar_v_indices]
        theta = torch.atan2(q_group[..., 3], q_group[..., 2]) + v_group[..., 2]
        values = torch.stack(
            (
                q_group[..., 0] + v_group[..., 0],
                q_group[..., 1] + v_group[..., 1],
                torch.cos(theta),
                torch.sin(theta),
            ),
            dim=-1,
        )
        result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

    for fallback_index, joint_id in enumerate(structure.manifold_fallback_joint_ids):
        q_start = structure.manifold_fallback_q_offsets[fallback_index]
        q_stop = structure.manifold_fallback_q_offsets[fallback_index + 1]
        v_start = structure.manifold_fallback_v_offsets[fallback_index]
        v_stop = structure.manifold_fallback_v_offsets[fallback_index + 1]
        q_indices = structure.manifold_fallback_q_indices[q_start:q_stop]
        v_indices = structure.manifold_fallback_v_indices[v_start:v_stop]
        values = model.joint_models[joint_id].integrate(
            q_broadcast[..., q_indices],
            v_broadcast[..., v_indices],
        )
        result = result.index_copy(-1, q_indices, values)

    return result


def difference(model: Model, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:  # noqa: PLR0915
    """Compute the universal tangent ``q1 ⊖ q0``."""

    _validate_manifold_tensor(model, "q0", q0, model.nq)
    _validate_manifold_tensor(model, "q1", q1, model.nq)
    batch_shape = torch.broadcast_shapes(q0.shape[:-1], q1.shape[:-1])
    result_dtype = torch.promote_types(q0.dtype, q1.dtype)
    q0_broadcast = q0.to(dtype=result_dtype).expand(*batch_shape, model.nq)
    q1_broadcast = q1.to(dtype=result_dtype).expand(*batch_shape, model.nq)
    if model.nv == 0:
        return q0_broadcast.new_zeros(*batch_shape, model.nv)

    result = q0_broadcast.new_zeros(*batch_shape, model.nv)
    structure = model.structure

    v_indices = structure.manifold_euclidean_v_indices
    if v_indices.numel():
        values = (
            q1_broadcast[..., structure.manifold_euclidean_q_indices]
            - q0_broadcast[..., structure.manifold_euclidean_q_indices]
        )
        result = result.index_copy(-1, v_indices, values)

    q_indices = structure.manifold_spherical_q_indices
    if q_indices.numel():
        q0_group = q0_broadcast[..., q_indices]
        q1_group = q1_broadcast[..., q_indices]
        delta = so3.compose(so3.inverse(q0_group), q1_group)
        values = so3.log(so3.normalize(delta))
        v_indices = structure.manifold_spherical_v_indices
        result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

    q_indices = structure.manifold_free_flyer_q_indices
    if q_indices.numel():
        q0_group = q0_broadcast[..., q_indices]
        q1_group = q1_broadcast[..., q_indices]
        values = se3.log(se3.compose(se3.inverse(q0_group), q1_group))
        v_indices = structure.manifold_free_flyer_v_indices
        result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

    q_indices = structure.manifold_unbounded_q_indices
    if q_indices.numel():
        q0_group = q0_broadcast[..., q_indices]
        q1_group = q1_broadcast[..., q_indices]
        theta0 = torch.atan2(q0_group[..., 1], q0_group[..., 0])
        theta1 = torch.atan2(q1_group[..., 1], q1_group[..., 0])
        values = (theta1 - theta0).unsqueeze(-1)
        v_indices = structure.manifold_unbounded_v_indices
        result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

    q_indices = structure.manifold_planar_q_indices
    if q_indices.numel():
        q0_group = q0_broadcast[..., q_indices]
        q1_group = q1_broadcast[..., q_indices]
        theta0 = torch.atan2(q0_group[..., 3], q0_group[..., 2])
        theta1 = torch.atan2(q1_group[..., 3], q1_group[..., 2])
        values = torch.stack(
            (
                q1_group[..., 0] - q0_group[..., 0],
                q1_group[..., 1] - q0_group[..., 1],
                theta1 - theta0,
            ),
            dim=-1,
        )
        v_indices = structure.manifold_planar_v_indices
        result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

    for fallback_index, joint_id in enumerate(structure.manifold_fallback_joint_ids):
        q_start = structure.manifold_fallback_q_offsets[fallback_index]
        q_stop = structure.manifold_fallback_q_offsets[fallback_index + 1]
        v_start = structure.manifold_fallback_v_offsets[fallback_index]
        v_stop = structure.manifold_fallback_v_offsets[fallback_index + 1]
        q_indices = structure.manifold_fallback_q_indices[q_start:q_stop]
        v_indices = structure.manifold_fallback_v_indices[v_start:v_stop]
        values = model.joint_models[joint_id].difference(
            q0_broadcast[..., q_indices],
            q1_broadcast[..., q_indices],
        )
        result = result.index_copy(-1, v_indices, values)

    return result


def random_configuration(model: Model, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw a random valid configuration for ``model``."""

    parts: list[torch.Tensor] = []
    for joint_id in range(model.njoints):
        joint_model = model.joint_models[joint_id]
        joint_nq = model.nqs[joint_id]
        if joint_nq == 0:
            continue
        q_index = model.idx_qs[joint_id]
        lower = model.lower_pos_limit[q_index : q_index + joint_nq]
        upper = model.upper_pos_limit[q_index : q_index + joint_nq]
        parts.append(joint_model.random_configuration(generator, lower, upper))
    if not parts:
        return torch.zeros(model.nq)
    return torch.cat(parts, dim=-1)


__all__: list[str] = []
