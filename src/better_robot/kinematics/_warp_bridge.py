"""Functional torch↔Warp bridge for the opt-in fused FK prototype.

The custom op owns fresh torch outputs and launches Warp on torch's current
CUDA stream.  Its registered autograd formula recomputes the VJP through the
torch oracle; a paired backward-op schema is reserved for a future explicit
VJP but is not the active path.  This is intentionally conservative: Warp's
generated adjoint is not correct for a dynamic topological loop, while the
torch recompute path is first- and second-order differentiable and reduces
shared model-value gradients through the execution-batch index maps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import warp as wp

from ..data_model.execution_batch import flatten_execution_batch
from ..data_model.model_structure import JOINT_KIND_CODES, ModelStructure
from ..data_model.model_values import ModelValues
from ..lie import se3, so3


wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/betterrobot-warp-cache")
wp.init()

from ._warp_kernels import fk_frames_f32, fk_frames_f64  # noqa: E402


def require_warp() -> None:
    """Initialize the optional Warp lane or propagate its import/runtime error."""

    wp.init()


def _torch_joint_transform(  # noqa: PLR0911
    kind: int,
    q_slice: torch.Tensor,
    axis: torch.Tensor,
    pitch: torch.Tensor,
) -> torch.Tensor:
    if kind in (0, 1, 17):
        return se3.identity(
            batch_shape=tuple(q_slice.shape[:-1]),
            dtype=q_slice.dtype,
            device=q_slice.device,
        )
    if kind in (2, 3, 4, 5):
        return se3.from_axis_angle(axis, q_slice[..., 0])
    if kind == 6:
        return se3.from_axis_angle(axis, torch.atan2(q_slice[..., 1], q_slice[..., 0]))
    if kind in (7, 8, 9, 10):
        return se3.from_translation(axis, q_slice[..., 0])
    if kind == 11:
        zeros = q_slice.new_zeros((*q_slice.shape[:-1], 3))
        return torch.cat((zeros, so3.normalize(q_slice)), dim=-1)
    if kind == 12:
        return se3.normalize(q_slice)
    if kind == 13:
        x, y, cosine, sine = q_slice.unbind(dim=-1)
        half_cosine = torch.sqrt(((1.0 + cosine) * 0.5).clamp_min(0.0))
        half_sine = torch.sqrt(((1.0 - cosine) * 0.5).clamp_min(0.0)) * torch.sign(sine)
        zero = torch.zeros_like(x)
        return torch.stack((x, y, zero, zero, zero, half_sine, half_cosine), dim=-1)
    if kind == 14:
        zeros = q_slice.new_zeros((*q_slice.shape[:-1], 3))
        ones = q_slice.new_ones((*q_slice.shape[:-1], 1))
        return torch.cat((q_slice, zeros, ones), dim=-1)
    if kind == 15:
        angle = q_slice[..., 0]
        return se3.compose(
            se3.from_translation(axis, pitch * angle),
            se3.from_axis_angle(axis, angle),
        )
    raise RuntimeError(f"Warp FK torch-VJP received unsupported joint kind code {kind}")


def _torch_fk_from_tables(
    q: torch.Tensor,
    joint_placements: torch.Tensor,
    frame_placements: torch.Tensor,
    q_map: torch.Tensor,
    value_map: torch.Tensor,
    parents: torch.Tensor,
    topo_order: torch.Tensor,
    kinds: torch.Tensor,
    nqs: torch.Tensor,
    idx_qs: torch.Tensor,
    axes: torch.Tensor,
    pitches: torch.Tensor,
    frame_parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_exec = q.index_select(0, q_map.to(torch.int64))
    placements_exec = joint_placements.index_select(0, value_map.to(torch.int64))
    frames_exec = frame_placements.index_select(0, value_map.to(torch.int64))
    parents_host = parents.detach().cpu()  # bench-ok: immutable prototype topology
    topo_host = topo_order.detach().cpu()  # bench-ok: immutable prototype topology
    kinds_host = kinds.detach().cpu()  # bench-ok: immutable prototype topology
    nqs_host = nqs.detach().cpu()  # bench-ok: immutable prototype topology
    idx_host = idx_qs.detach().cpu()  # bench-ok: immutable prototype topology
    parents_static = tuple(int(value) for value in parents_host.tolist())
    topo_static = tuple(int(value) for value in topo_host.tolist())
    kinds_static = tuple(int(value) for value in kinds_host.tolist())
    nqs_static = tuple(int(value) for value in nqs_host.tolist())
    idx_static = tuple(int(value) for value in idx_host.tolist())

    world: list[torch.Tensor | None] = [None] * len(parents_static)
    local: list[torch.Tensor | None] = [None] * len(parents_static)
    for joint_index in topo_static:
        nq_i = nqs_static[joint_index]
        iq = idx_static[joint_index]
        delta = _torch_joint_transform(
            kinds_static[joint_index],
            q_exec[..., iq : iq + nq_i],
            axes[joint_index],
            pitches[joint_index],
        )
        local_pose = se3.compose(placements_exec[..., joint_index, :], delta)
        local[joint_index] = local_pose
        parent = parents_static[joint_index]
        world[joint_index] = (
            local_pose if parent < 0 else se3.compose(world[parent], local_pose)  # type: ignore[arg-type]
        )
    local_tensor = torch.stack(local, dim=-2)  # type: ignore[arg-type]
    world_tensor = torch.stack(world, dim=-2)  # type: ignore[arg-type]
    frame_tensor = se3.compose(world_tensor.index_select(-2, frame_parents.to(torch.int64)), frames_exec)
    return world_tensor, local_tensor, frame_tensor


def _vjp(
    q: torch.Tensor,
    joint_placements: torch.Tensor,
    frame_placements: torch.Tensor,
    q_map: torch.Tensor,
    value_map: torch.Tensor,
    parents: torch.Tensor,
    topo_order: torch.Tensor,
    kinds: torch.Tensor,
    nqs: torch.Tensor,
    idx_qs: torch.Tensor,
    axes: torch.Tensor,
    pitches: torch.Tensor,
    frame_parents: torch.Tensor,
    grad_world: torch.Tensor,
    grad_local: torch.Tensor,
    grad_frames: torch.Tensor,
    *,
    create_graph: bool,
    detach_inputs: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        original = (q, joint_placements, frame_placements)
        working = tuple(
            value.detach().requires_grad_(True) if detach_inputs or not value.requires_grad else value
            for value in original
        )
        outputs = _torch_fk_from_tables(
            working[0],
            working[1],
            working[2],
            q_map,
            value_map,
            parents,
            topo_order,
            kinds,
            nqs,
            idx_qs,
            axes,
            pitches,
            frame_parents,
        )
        gradients = torch.autograd.grad(
            outputs,
            working,
            (grad_world, grad_local, grad_frames),
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient for value, gradient in zip(original, gradients)
    )  # type: ignore[return-value]


@torch.library.custom_op("better_robot::warp_fk_backward", mutates_args=())
def _warp_fk_backward(
    q: torch.Tensor,
    joint_placements: torch.Tensor,
    frame_placements: torch.Tensor,
    q_map: torch.Tensor,
    value_map: torch.Tensor,
    parents: torch.Tensor,
    topo_order: torch.Tensor,
    kinds: torch.Tensor,
    nqs: torch.Tensor,
    idx_qs: torch.Tensor,
    axes: torch.Tensor,
    pitches: torch.Tensor,
    frame_parents: torch.Tensor,
    grad_world: torch.Tensor,
    grad_local: torch.Tensor,
    grad_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _vjp(
        q,
        joint_placements,
        frame_placements,
        q_map,
        value_map,
        parents,
        topo_order,
        kinds,
        nqs,
        idx_qs,
        axes,
        pitches,
        frame_parents,
        grad_world,
        grad_local,
        grad_frames,
        create_graph=False,
        detach_inputs=True,
    )


@_warp_fk_backward.register_fake
def _warp_fk_backward_fake(
    q,
    joint_placements,
    frame_placements,
    q_map,
    value_map,
    parents,
    topo_order,
    kinds,
    nqs,
    idx_qs,
    axes,
    pitches,
    frame_parents,
    grad_world,
    grad_local,
    grad_frames,
):
    return torch.empty_like(q), torch.empty_like(joint_placements), torch.empty_like(frame_placements)


@torch.library.custom_op("better_robot::warp_fk_forward", mutates_args=())
def _warp_fk_forward(
    q: torch.Tensor,
    joint_placements: torch.Tensor,
    frame_placements: torch.Tensor,
    q_map: torch.Tensor,
    value_map: torch.Tensor,
    parents: torch.Tensor,
    topo_order: torch.Tensor,
    kinds: torch.Tensor,
    nqs: torch.Tensor,
    idx_qs: torch.Tensor,
    axes: torch.Tensor,
    pitches: torch.Tensor,
    frame_parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    execution_size = q_map.shape[0]
    njoints = parents.shape[0]
    nframes = frame_parents.shape[0]
    world = q.new_empty((execution_size, njoints, 7))
    local = q.new_empty((execution_size, njoints, 7))
    frames = q.new_empty((execution_size, nframes, 7))
    transform_dtype = wp.transformf if q.dtype == torch.float32 else wp.transformd
    axis_dtype = wp.vec3f if q.dtype == torch.float32 else wp.vec3d
    scalar_dtype = wp.float32 if q.dtype == torch.float32 else wp.float64
    kernel = fk_frames_f32 if q.dtype == torch.float32 else fk_frames_f64

    device = wp.device_from_torch(q.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(q.device)) if q.is_cuda else None
    inputs = [
        wp.from_torch(q, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(joint_placements, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(frame_placements, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(q_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(value_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(parents, dtype=wp.int32, requires_grad=False),
        wp.from_torch(topo_order, dtype=wp.int32, requires_grad=False),
        wp.from_torch(kinds, dtype=wp.int8, requires_grad=False),
        wp.from_torch(idx_qs, dtype=wp.int32, requires_grad=False),
        wp.from_torch(axes, dtype=axis_dtype, requires_grad=False),
        wp.from_torch(pitches, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(frame_parents, dtype=wp.int32, requires_grad=False),
        njoints,
        nframes,
    ]
    outputs = [
        wp.from_torch(local, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(world, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(frames, dtype=transform_dtype, requires_grad=False),
    ]
    wp.launch(
        kernel,
        dim=execution_size,
        inputs=inputs,
        outputs=outputs,
        device=device,
        stream=stream,
        record_tape=False,
    )
    return world, local, frames


@_warp_fk_forward.register_fake
def _warp_fk_forward_fake(
    q,
    joint_placements,
    frame_placements,
    q_map,
    value_map,
    parents,
    topo_order,
    kinds,
    nqs,
    idx_qs,
    axes,
    pitches,
    frame_parents,
):
    execution_size = q_map.shape[0]
    return (
        q.new_empty((execution_size, parents.shape[0], 7)),
        q.new_empty((execution_size, parents.shape[0], 7)),
        q.new_empty((execution_size, frame_parents.shape[0], 7)),
    )


def _setup_fk_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _fk_backward(ctx, grad_world, grad_local, grad_frames):
    saved = ctx.saved_tensors
    grad_world = (
        torch.zeros_like(saved[0].new_empty((saved[3].shape[0], saved[5].shape[0], 7)))
        if grad_world is None
        else grad_world
    )
    grad_local = torch.zeros_like(grad_world) if grad_local is None else grad_local
    grad_frames = saved[0].new_zeros((saved[3].shape[0], saved[12].shape[0], 7)) if grad_frames is None else grad_frames
    if torch.is_grad_enabled():
        gradients = _vjp(
            *saved,
            grad_world,
            grad_local,
            grad_frames,
            create_graph=True,
            detach_inputs=False,
        )
    else:
        # A torch custom-op implementation runs below the Autograd dispatch
        # key, so a recompute VJP cannot be evaluated inside the paired op.
        # Keep recomputation in the registered formula; the paired op remains
        # the shape/schema prototype for a future hand-written Warp VJP.
        gradients = _vjp(
            *saved,
            grad_world,
            grad_local,
            grad_frames,
            create_graph=False,
            detach_inputs=True,
        )
    return (*gradients, *(None for _ in range(len(saved) - 3)))


_warp_fk_forward.register_autograd(_fk_backward, setup_context=_setup_fk_context)


@dataclass(frozen=True)
class WarpFKResult:
    world: torch.Tensor
    local: torch.Tensor
    frames: torch.Tensor


def try_warp_forward_kinematics(  # noqa: PLR0911
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> WarpFKResult | None:
    """Run the opt-in prototype, or return ``None`` for torch fallback."""

    values.validate(structure)
    if q.dtype not in (torch.float32, torch.float64):
        return None
    if any(code < 0 or code == JOINT_KIND_CODES["composite"] for code in structure.joint_kind_codes):
        return None
    if q.stride(-1) != 1 or values.joint_placements.stride(-1) != 1 or values.frame_placements.stride(-1) != 1:
        return None
    if values.joint_placements.shape[:-2] != values.frame_placements.shape[:-2]:
        return None

    placements = values.joint_placements.to(dtype=q.dtype)
    frame_placements = values.frame_placements.to(dtype=q.dtype)
    execution = flatten_execution_batch(
        q,
        (placements, frame_placements, values.body_inertias),
        value_event_ndims=(2, 2, 2),
    )
    if not torch.equal(execution.values[0].batch_indices, execution.values[1].batch_indices):
        return None
    q_unique = execution.q.tensor
    placement_unique = execution.values[0].tensor
    frame_unique = execution.values[1].tensor
    if (
        q_unique.data_ptr() != q.data_ptr()
        or placement_unique.data_ptr() != placements.data_ptr()
        or frame_unique.data_ptr() != frame_placements.data_ptr()
    ):
        return None

    device = q.device
    world, local, frames = _warp_fk_forward(
        q_unique,
        placement_unique,
        frame_unique,
        execution.q.batch_indices.to(device=device, dtype=torch.int32),
        execution.values[0].batch_indices.to(device=device, dtype=torch.int32),
        structure.parents_tensor.to(device=device),
        structure.topo_order_tensor.to(device=device),
        structure.joint_kind_tensor.to(device=device),
        structure.nqs_tensor.to(device=device),
        structure.idx_qs_tensor.to(device=device),
        structure.joint_axes.to(device=device, dtype=q.dtype),
        structure.joint_pitches.to(device=device, dtype=q.dtype),
        structure.frame_parent_joints.to(device=device),
    )
    return WarpFKResult(
        execution.unflatten(world),
        execution.unflatten(local),
        execution.unflatten(frames),
    )


__all__ = ["WarpFKResult", "require_warp", "try_warp_forward_kinematics"]
