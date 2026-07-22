"""Functional torch↔Warp bridge for the CUDA-validated opt-in fused FK lane.

The custom op owns fresh torch outputs and launches Warp on torch's current
CUDA stream. The active autograd wrapper recomputes the VJP through the raw
torch FK pass. This is intentionally conservative: Warp's generated adjoint is
not correct for a dynamic topological loop, while the torch recompute path is
first- and second-order differentiable and reduces shared model-value gradients
through the execution-batch index maps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch
import warp as wp

from ..data_model.execution_batch import flatten_execution_batch
from ..data_model.model_structure import JOINT_KIND_CODES, ModelStructure
from ..data_model.model_values import ModelValues


wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/betterrobot-warp-cache")
wp.init()

from ._warp_kernels import fk_frames_f32, fk_frames_f64  # noqa: E402
from .forward import (  # noqa: E402
    _warn_warp_fallback,
    forward_kinematics_raw,
    frame_placements_raw,
)


_SUPPORTED_JOINT_KIND_CODES = frozenset(range(JOINT_KIND_CODES["helical"] + 1))


def _vjp(
    q: torch.Tensor,
    joint_placements: torch.Tensor,
    frame_placements: torch.Tensor,
    q_map: torch.Tensor,
    value_map: torch.Tensor,
    frame_map: torch.Tensor,
    structure: ModelStructure,
    values: ModelValues,
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
        q_exec = working[0].index_select(0, q_map.to(torch.int64))
        placements_exec = working[1].index_select(0, value_map.to(torch.int64))
        frames_exec = working[2].index_select(0, frame_map.to(torch.int64))
        body_inertias = values.body_inertias
        if body_inertias.ndim > 2:
            body_inertias = body_inertias[(0,) * (body_inertias.ndim - 2)]
        recompute_values = replace(
            values,
            joint_placements=placements_exec,
            frame_placements=frames_exec,
            body_inertias=body_inertias,
        )
        fk_result = forward_kinematics_raw(structure, recompute_values, q_exec)
        world = fk_result.joint_pose_world
        local = fk_result.joint_pose_local
        frames = frame_placements_raw(
            structure,
            recompute_values,
            world,
        ).frame_pose_world
        gradients = torch.autograd.grad(
            (world, local, frames),
            working,
            (grad_world, grad_local, grad_frames),
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient for value, gradient in zip(original, gradients)
    )  # type: ignore[return-value]


@torch.library.custom_op("better_robot::warp_fk_forward", mutates_args=())
def _warp_fk_forward(
    q: torch.Tensor,
    joint_placements: torch.Tensor,
    frame_placements: torch.Tensor,
    q_map: torch.Tensor,
    value_map: torch.Tensor,
    frame_map: torch.Tensor,
    parents: torch.Tensor,
    topo_order: torch.Tensor,
    kinds: torch.Tensor,
    idx_qs: torch.Tensor,
    idx_qs_full: torch.Tensor,
    mimic_sources: torch.Tensor,
    q_expansion: torch.Tensor,
    q_offsets: torch.Tensor,
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
        wp.from_torch(frame_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(parents, dtype=wp.int32, requires_grad=False),
        wp.from_torch(topo_order, dtype=wp.int32, requires_grad=False),
        wp.from_torch(kinds, dtype=wp.int8, requires_grad=False),
        wp.from_torch(idx_qs, dtype=wp.int32, requires_grad=False),
        wp.from_torch(idx_qs_full, dtype=wp.int32, requires_grad=False),
        wp.from_torch(mimic_sources, dtype=wp.int32, requires_grad=False),
        wp.from_torch(q_expansion, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(q_offsets, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(axes, dtype=axis_dtype, requires_grad=False),
        wp.from_torch(pitches, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(frame_parents, dtype=wp.int32, requires_grad=False),
        njoints,
        nframes,
        q.shape[1],
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
    frame_map,
    parents,
    topo_order,
    kinds,
    idx_qs,
    idx_qs_full,
    mimic_sources,
    q_expansion,
    q_offsets,
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


class _WarpFKFunction(torch.autograd.Function):
    """Autograd owner for the Warp forward and raw-torch recompute VJP."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        joint_placements: torch.Tensor,
        frame_placements: torch.Tensor,
        q_map: torch.Tensor,
        value_map: torch.Tensor,
        frame_map: torch.Tensor,
        structure: ModelStructure,
        values: ModelValues,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.structure = structure
        ctx.values = values
        ctx.save_for_backward(q, joint_placements, frame_placements, q_map, value_map, frame_map)
        device = q.device
        return _warp_fk_forward(
            q,
            joint_placements,
            frame_placements,
            q_map,
            value_map,
            frame_map,
            structure.parents_tensor.to(device=device),
            structure.topo_order_tensor.to(device=device),
            structure.joint_kind_tensor.to(device=device),
            structure.idx_qs_tensor.to(device=device),
            structure.idx_qs_full_tensor.to(device=device),
            structure.mimic_source_tensor.to(device=device),
            structure.q_expansion.to(device=device, dtype=q.dtype),
            structure.q_offset.to(device=device, dtype=q.dtype),
            structure.joint_axes.to(device=device, dtype=q.dtype),
            structure.joint_pitches.to(device=device, dtype=q.dtype),
            structure.frame_parent_joints.to(device=device),
        )

    @staticmethod
    def backward(ctx, grad_world, grad_local, grad_frames):
        q, joint_placements, frame_placements, q_map, value_map, frame_map = ctx.saved_tensors
        grad_world = q.new_zeros((q_map.shape[0], ctx.structure.njoints, 7)) if grad_world is None else grad_world
        grad_local = torch.zeros_like(grad_world) if grad_local is None else grad_local
        grad_frames = q.new_zeros((q_map.shape[0], ctx.structure.nframes, 7)) if grad_frames is None else grad_frames
        create_graph = torch.is_grad_enabled()
        gradients = _vjp(
            q,
            joint_placements,
            frame_placements,
            q_map,
            value_map,
            frame_map,
            ctx.structure,
            ctx.values,
            grad_world,
            grad_local,
            grad_frames,
            create_graph=create_graph,
            detach_inputs=not create_graph,
        )
        return (*gradients, None, None, None, None, None)


@dataclass(frozen=True)
class WarpFKResult:
    world: torch.Tensor
    local: torch.Tensor
    frames: torch.Tensor


def _capture_is_active(*tensors: torch.Tensor) -> bool:
    return any(tensor.is_cuda for tensor in tensors) and torch.cuda.is_current_stream_capturing()


def _raise_capture_layout_error(name: str, *capture_inputs: torch.Tensor) -> None:
    """Reject a layout fallback while the current CUDA stream is recording."""
    if _capture_is_active(*capture_inputs):
        raise RuntimeError(
            f"better_robot: input {name!r} has an unsupported layout for the Warp FK lane "
            "while CUDA graph capture is active; silent Torch fallback is disabled during "
            "capture. Make the input contiguous before capture, or disable graph capture."
        )


def _decline(reason_key: str, reason: str, *capture_inputs: torch.Tensor, remedy: str) -> None:
    """Warn once before a Torch fallback, or forbid it during graph recording."""
    if _capture_is_active(*capture_inputs):
        raise RuntimeError(
            "better_robot: Warp FK cannot silently fall back to the Torch lane while CUDA "
            f"graph capture is active. {reason}. {remedy}, or disable graph capture."
        )
    _warn_warp_fallback(reason_key, reason)


def try_warp_forward_kinematics(  # noqa: PLR0911
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> WarpFKResult | None:
    """Run the opt-in Warp lane, or return ``None`` for torch fallback."""

    if q.dtype not in (torch.float32, torch.float64):
        _decline(
            f"dtype:{q.dtype}",
            f"q has unsupported dtype {q.dtype}",
            q,
            values.joint_placements,
            values.frame_placements,
            remedy="Convert q to float32 or float64 before capture",
        )
        return None
    if any(code not in _SUPPORTED_JOINT_KIND_CODES for code in structure.joint_kind_codes):
        _decline(
            "joint-kind",
            "the model contains a joint kind unsupported by the Warp ABI",
            q,
            values.joint_placements,
            values.frame_placements,
            remedy=(
                "Use only fixed, revolute, continuous, prismatic, spherical, "
                "floating, planar, translation, or helical joints"
            ),
        )
        return None
    layout_inputs = (
        ("q", q),
        ("joint_placements", values.joint_placements),
        ("frame_placements", values.frame_placements),
    )
    unsupported_layout = tuple((name, tensor) for name, tensor in layout_inputs if tensor.stride(-1) != 1)
    if unsupported_layout:
        capture_inputs = (q, values.joint_placements, values.frame_placements)
        for name, _ in unsupported_layout:
            _raise_capture_layout_error(name, *capture_inputs)
        names = tuple(name for name, _ in unsupported_layout)
        _warn_warp_fallback(
            f"layout:{names}",
            f"inputs {names} do not have unit stride on their trailing axis",
        )
        return None
    joint_batch = tuple(values.joint_placements.shape[:-2])
    frame_batch = tuple(values.frame_placements.shape[:-2])
    try:
        torch.broadcast_shapes(joint_batch, frame_batch)
    except RuntimeError:
        _decline(
            "placement-batch-shape",
            "joint and frame placement batch shapes do not broadcast",
            q,
            values.joint_placements,
            values.frame_placements,
            remedy="Give joint and frame placements broadcast-compatible batch shapes before capture",
        )
        return None

    placements = values.joint_placements.to(dtype=q.dtype)
    frame_placements = values.frame_placements.to(dtype=q.dtype)
    execution = flatten_execution_batch(
        q,
        (placements, frame_placements, values.body_inertias),
        value_event_ndims=(2, 2, 2),
    )
    # Joint and frame placements keep independent execution-to-input maps, so a
    # batched shape table pairs cleanly with an unbatched frame table. The
    # ``materialized`` check below compares data pointers rather than tensor
    # values: ``torch.equal`` would synchronize the host during graph capture.
    q_unique = execution.q.tensor
    placement_unique = execution.values[0].tensor
    frame_unique = execution.values[1].tensor
    materialized = tuple(
        name
        for name, flattened, original in (
            ("q", q_unique, q),
            ("joint_placements", placement_unique, placements),
            ("frame_placements", frame_unique, frame_placements),
        )
        if flattened.data_ptr() != original.data_ptr()
    )
    if materialized:
        _decline(
            f"materialized:{materialized}",
            f"flattening would materialize inputs {materialized}",
            q,
            values.joint_placements,
            values.frame_placements,
            remedy="Make those batch dimensions reshape-compatible before capture",
        )
        return None

    q_map = execution.q.batch_indices.to(device=q.device, dtype=torch.int32)
    value_map = execution.values[0].batch_indices.to(device=q.device, dtype=torch.int32)
    frame_map = execution.values[1].batch_indices.to(device=q.device, dtype=torch.int32)
    world, local, frames = _WarpFKFunction.apply(
        q_unique,
        placement_unique,
        frame_unique,
        q_map,
        value_map,
        frame_map,
        structure,
        values,
    )
    return WarpFKResult(
        execution.unflatten(world),
        execution.unflatten(local),
        execution.unflatten(frames),
    )


__all__ = ["WarpFKResult", "try_warp_forward_kinematics"]
