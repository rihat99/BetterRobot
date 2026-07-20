"""Torch↔Warp bridge for the opt-in fused inverse-dynamics lane."""

from __future__ import annotations

import os
from dataclasses import replace

import torch
import warp as wp

from ..data_model.execution_batch import flatten_execution_batch
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ._warp_kernels import rnea_f32, rnea_f64
from .rnea import RNEAResult, _warn_warp_rnea_fallback, rnea_raw


wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/betterrobot-warp-cache")
wp.init()

# Codes 0–15 are the concrete joint kinds implemented by the frozen kernel
# dispatch. Composite, the metadata-only mimic sentinel, unknown negative
# codes, and future appended kinds must decline instead of becoming fixed.
_SUPPORTED_JOINT_KIND_CODES = frozenset(range(16))


def _vjp(
    q: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    fext: torch.Tensor,
    joint_placements: torch.Tensor,
    body_inertias: torch.Tensor,
    gravity: torch.Tensor,
    q_map: torch.Tensor,
    velocity_map: torch.Tensor,
    acceleration_map: torch.Tensor,
    fext_map: torch.Tensor,
    placement_map: torch.Tensor,
    inertia_map: torch.Tensor,
    gravity_map: torch.Tensor,
    structure: ModelStructure,
    values: ModelValues,
    gradients: tuple[torch.Tensor, ...],
    *,
    has_fext: bool,
    create_graph: bool,
    detach_inputs: bool,
) -> tuple[torch.Tensor, ...]:
    """Recompute the RNEA VJP through the canonical Torch implementation."""
    with torch.enable_grad():
        original = (
            q,
            velocity,
            acceleration,
            fext,
            joint_placements,
            body_inertias,
            gravity,
        )
        working = tuple(
            value.detach().requires_grad_(True) if detach_inputs or not value.requires_grad else value
            for value in original
        )
        q_exec = working[0].index_select(0, q_map.to(torch.int64))
        velocity_exec = working[1].index_select(0, velocity_map.to(torch.int64))
        acceleration_exec = working[2].index_select(0, acceleration_map.to(torch.int64))
        fext_exec = working[3].index_select(0, fext_map.to(torch.int64))
        placements_exec = working[4].index_select(0, placement_map.to(torch.int64))
        inertias_exec = working[5].index_select(0, inertia_map.to(torch.int64))
        gravity_exec = working[6].index_select(0, gravity_map.to(torch.int64))
        frame_placements = values.frame_placements
        if frame_placements.ndim > 2:
            frame_placements = frame_placements[(0,) * (frame_placements.ndim - 2)]
        recompute_values = replace(
            values,
            joint_placements=placements_exec,
            body_inertias=inertias_exec,
            frame_placements=frame_placements,
            gravity=gravity_exec,
        )
        result = rnea_raw(
            structure,
            recompute_values,
            q_exec,
            velocity_exec,
            acceleration_exec,
            fext=fext_exec if has_fext else None,
        )
        outputs = (
            result.tau,
            result.joint_pose_world,
            result.joint_pose_local,
            result.joint_velocity_local,
            result.joint_acceleration_local,
            result.joint_forces,
        )
        input_gradients = torch.autograd.grad(
            outputs,
            working,
            gradients,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(original, input_gradients, strict=True)
    )


@torch.library.custom_op("better_robot::warp_rnea_forward", mutates_args=())
def _warp_rnea_forward(  # noqa: PLR0913 - explicit frozen device ABI
    q: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    fext: torch.Tensor,
    joint_placements: torch.Tensor,
    body_inertias: torch.Tensor,
    gravity: torch.Tensor,
    q_map: torch.Tensor,
    velocity_map: torch.Tensor,
    acceleration_map: torch.Tensor,
    fext_map: torch.Tensor,
    placement_map: torch.Tensor,
    inertia_map: torch.Tensor,
    gravity_map: torch.Tensor,
    parents: torch.Tensor,
    topo_order: torch.Tensor,
    kinds: torch.Tensor,
    idx_qs: torch.Tensor,
    idx_qs_full: torch.Tensor,
    idx_vs: torch.Tensor,
    idx_vs_full: torch.Tensor,
    nvs_full: torch.Tensor,
    axes: torch.Tensor,
    pitches: torch.Tensor,
    motion_subspaces: torch.Tensor,
    mimic_sources: torch.Tensor,
    q_expansion: torch.Tensor,
    q_offsets: torch.Tensor,
    v_expansion: torch.Tensor,
    has_fext: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    execution_size = q_map.shape[0]
    njoints = parents.shape[0]
    nv = velocity.shape[1]
    tau = q.new_empty((execution_size, nv))
    world = q.new_empty((execution_size, njoints, 7))
    local = q.new_empty((execution_size, njoints, 7))
    body_velocity = q.new_empty((execution_size, njoints, 6))
    body_acceleration = q.new_empty((execution_size, njoints, 6))
    body_force = q.new_empty((execution_size, njoints, 6))

    is_float32 = q.dtype == torch.float32
    scalar_dtype = wp.float32 if is_float32 else wp.float64
    transform_dtype = wp.transformf if is_float32 else wp.transformd
    axis_dtype = wp.vec3f if is_float32 else wp.vec3d
    kernel = rnea_f32 if is_float32 else rnea_f64
    device = wp.device_from_torch(q.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(q.device))

    inputs = [
        wp.from_torch(q, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(velocity, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(acceleration, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(fext, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(joint_placements, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(body_inertias, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(gravity, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(q_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(velocity_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(acceleration_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(fext_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(placement_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(inertia_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(gravity_map, dtype=wp.int32, requires_grad=False),
        wp.from_torch(parents, dtype=wp.int32, requires_grad=False),
        wp.from_torch(topo_order, dtype=wp.int32, requires_grad=False),
        wp.from_torch(kinds, dtype=wp.int8, requires_grad=False),
        wp.from_torch(idx_qs, dtype=wp.int32, requires_grad=False),
        wp.from_torch(idx_qs_full, dtype=wp.int32, requires_grad=False),
        wp.from_torch(idx_vs, dtype=wp.int32, requires_grad=False),
        wp.from_torch(idx_vs_full, dtype=wp.int32, requires_grad=False),
        wp.from_torch(nvs_full, dtype=wp.int32, requires_grad=False),
        wp.from_torch(axes, dtype=axis_dtype, requires_grad=False),
        wp.from_torch(pitches, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(motion_subspaces, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(mimic_sources, dtype=wp.int32, requires_grad=False),
        wp.from_torch(q_expansion, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(q_offsets, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(v_expansion, dtype=scalar_dtype, requires_grad=False),
        q.shape[1],
        nv,
        njoints,
        has_fext,
    ]
    outputs = [
        wp.from_torch(tau, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(local, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(world, dtype=transform_dtype, requires_grad=False),
        wp.from_torch(body_velocity, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(body_acceleration, dtype=scalar_dtype, requires_grad=False),
        wp.from_torch(body_force, dtype=scalar_dtype, requires_grad=False),
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
    return tau, world, local, body_velocity, body_acceleration, body_force


@_warp_rnea_forward.register_fake
def _warp_rnea_forward_fake(  # noqa: PLR0913 - mirrors the custom-op schema
    q,
    velocity,
    acceleration,
    fext,
    joint_placements,
    body_inertias,
    gravity,
    q_map,
    velocity_map,
    acceleration_map,
    fext_map,
    placement_map,
    inertia_map,
    gravity_map,
    parents,
    topo_order,
    kinds,
    idx_qs,
    idx_qs_full,
    idx_vs,
    idx_vs_full,
    nvs_full,
    axes,
    pitches,
    motion_subspaces,
    mimic_sources,
    q_expansion,
    q_offsets,
    v_expansion,
    has_fext,
):
    execution_size = q_map.shape[0]
    njoints = parents.shape[0]
    return (
        q.new_empty((execution_size, velocity.shape[1])),
        q.new_empty((execution_size, njoints, 7)),
        q.new_empty((execution_size, njoints, 7)),
        q.new_empty((execution_size, njoints, 6)),
        q.new_empty((execution_size, njoints, 6)),
        q.new_empty((execution_size, njoints, 6)),
    )


class _WarpRNEAFunction(torch.autograd.Function):
    """Own the Warp forward and canonical Torch recompute VJP."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        fext: torch.Tensor,
        joint_placements: torch.Tensor,
        body_inertias: torch.Tensor,
        gravity: torch.Tensor,
        q_map: torch.Tensor,
        velocity_map: torch.Tensor,
        acceleration_map: torch.Tensor,
        fext_map: torch.Tensor,
        placement_map: torch.Tensor,
        inertia_map: torch.Tensor,
        gravity_map: torch.Tensor,
        structure: ModelStructure,
        values: ModelValues,
        has_fext: bool,
    ) -> tuple[torch.Tensor, ...]:
        ctx.structure = structure
        ctx.values = values
        ctx.has_fext = has_fext
        ctx.save_for_backward(
            q,
            velocity,
            acceleration,
            fext,
            joint_placements,
            body_inertias,
            gravity,
            q_map,
            velocity_map,
            acceleration_map,
            fext_map,
            placement_map,
            inertia_map,
            gravity_map,
        )
        device = q.device
        return _warp_rnea_forward(
            q,
            velocity,
            acceleration,
            fext,
            joint_placements,
            body_inertias,
            gravity,
            q_map,
            velocity_map,
            acceleration_map,
            fext_map,
            placement_map,
            inertia_map,
            gravity_map,
            structure.parents_tensor.to(device=device),
            structure.topo_order_tensor.to(device=device),
            structure.joint_kind_tensor.to(device=device),
            structure.idx_qs_tensor.to(device=device),
            structure.idx_qs_full_tensor.to(device=device),
            structure.idx_vs_tensor.to(device=device),
            structure.idx_vs_full_tensor.to(device=device),
            structure.nvs_full_tensor.to(device=device),
            structure.joint_axes.to(device=device, dtype=q.dtype),
            structure.joint_pitches.to(device=device, dtype=q.dtype),
            structure.joint_motion_subspaces.to(device=device, dtype=q.dtype),
            structure.mimic_source_tensor.to(device=device),
            structure.q_expansion.to(device=device, dtype=q.dtype),
            structure.q_offset.to(device=device, dtype=q.dtype),
            structure.v_expansion.to(device=device, dtype=q.dtype),
            has_fext,
        )

    @staticmethod
    def backward(ctx, *gradients):
        (
            q,
            velocity,
            acceleration,
            fext,
            joint_placements,
            body_inertias,
            gravity,
            q_map,
            velocity_map,
            acceleration_map,
            fext_map,
            placement_map,
            inertia_map,
            gravity_map,
        ) = ctx.saved_tensors
        output_shapes = (
            (q_map.shape[0], velocity.shape[1]),
            (q_map.shape[0], ctx.structure.njoints, 7),
            (q_map.shape[0], ctx.structure.njoints, 7),
            (q_map.shape[0], ctx.structure.njoints, 6),
            (q_map.shape[0], ctx.structure.njoints, 6),
            (q_map.shape[0], ctx.structure.njoints, 6),
        )
        prepared_gradients = tuple(
            q.new_zeros(shape) if gradient is None else gradient
            for shape, gradient in zip(output_shapes, gradients, strict=True)
        )
        create_graph = torch.is_grad_enabled()
        input_gradients = _vjp(
            q,
            velocity,
            acceleration,
            fext,
            joint_placements,
            body_inertias,
            gravity,
            q_map,
            velocity_map,
            acceleration_map,
            fext_map,
            placement_map,
            inertia_map,
            gravity_map,
            ctx.structure,
            ctx.values,
            prepared_gradients,
            has_fext=ctx.has_fext,
            create_graph=create_graph,
            detach_inputs=not create_graph,
        )
        return (
            *input_gradients,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _capture_is_active(*tensors: torch.Tensor) -> bool:
    return any(tensor.is_cuda for tensor in tensors) and torch.cuda.is_current_stream_capturing()


def _decline(
    reason_key: str,
    reason: str,
    *capture_inputs: torch.Tensor,
) -> None:
    if _capture_is_active(*capture_inputs):
        raise RuntimeError(
            "better_robot: Warp RNEA cannot fall back to the Torch lane while CUDA "
            f"graph capture is active because {reason}. Fix the input before capture "
            "or disable graph capture."
        )
    _warn_warp_rnea_fallback(reason_key, reason)


def try_warp_rnea(  # noqa: PLR0911
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> RNEAResult | None:
    """Run fused inverse dynamics, or return ``None`` after an honest decline."""
    capture_inputs = (
        q,
        velocity,
        acceleration,
        values.joint_placements,
        values.body_inertias,
        values.gravity,
        *(tuple() if fext is None else (fext,)),
    )
    if not q.is_cuda:
        _decline("device", "the inputs are not CUDA tensors", *capture_inputs)
        return None
    if q.dtype not in (torch.float32, torch.float64):
        _decline("dtype", f"q has unsupported dtype {q.dtype}", *capture_inputs)
        return None
    if any(code not in _SUPPORTED_JOINT_KIND_CODES for code in structure.joint_kind_codes):
        _decline(
            "joint-kind",
            "the model contains a joint kind unsupported by the Warp ABI",
            *capture_inputs,
        )
        return None
    tensor_inputs = (
        ("q", q),
        ("velocity", velocity),
        ("acceleration", acceleration),
        ("joint_placements", values.joint_placements),
        ("body_inertias", values.body_inertias),
        ("gravity", values.gravity),
        *((tuple()) if fext is None else (("fext", fext),)),
    )
    mismatched = tuple(name for name, tensor in tensor_inputs if tensor.device != q.device or tensor.dtype != q.dtype)
    if mismatched:
        _decline(
            f"dtype-device:{mismatched}",
            f"inputs {mismatched} do not share q's dtype and CUDA device",
            *capture_inputs,
        )
        return None
    unsupported_layout = tuple(name for name, tensor in tensor_inputs if tensor.stride(-1) != 1)
    if unsupported_layout:
        _decline(
            f"layout:{unsupported_layout}",
            f"inputs {unsupported_layout} do not have unit trailing stride",
            *capture_inputs,
        )
        return None

    if q.shape[-1:] != (structure.nq,) or velocity.shape[-1:] != (structure.nv,):
        _decline(
            "query-shape",
            "q or velocity has an unsupported trailing shape",
            *capture_inputs,
        )
        return None
    if acceleration.shape[-1:] != (structure.nv,):
        _decline(
            "acceleration-shape",
            "acceleration has an unsupported trailing shape",
            *capture_inputs,
        )
        return None
    if fext is not None and fext.shape[-2:] != (structure.njoints, 6):
        _decline("fext-shape", "fext has an unsupported trailing shape", *capture_inputs)
        return None

    canonical_batch = values._execution_batch_shape(q)
    fext_input = q.new_zeros((structure.njoints, 6)) if fext is None else fext
    execution = flatten_execution_batch(
        q,
        (
            velocity,
            acceleration,
            fext_input,
            values.joint_placements,
            values.body_inertias,
            values.frame_placements,
            values.gravity,
        ),
        value_event_ndims=(1, 1, 2, 2, 2, 2, 1),
    )
    if execution.batch_shape != canonical_batch:
        _decline(
            "query-batch-shape",
            "velocity, acceleration, fext, or gravity would enlarge the model execution batch",
            *capture_inputs,
        )
        return None

    q_unique = execution.q.tensor
    velocity_unique = execution.values[0].tensor
    acceleration_unique = execution.values[1].tensor
    fext_unique = execution.values[2].tensor
    placement_unique = execution.values[3].tensor
    inertia_unique = execution.values[4].tensor
    gravity_unique = execution.values[6].tensor
    materialized = tuple(
        name
        for name, flattened, original in (
            ("q", q_unique, q),
            ("velocity", velocity_unique, velocity),
            ("acceleration", acceleration_unique, acceleration),
            ("fext", fext_unique, fext_input),
            ("joint_placements", placement_unique, values.joint_placements),
            ("body_inertias", inertia_unique, values.body_inertias),
            ("gravity", gravity_unique, values.gravity),
        )
        if flattened.data_ptr() != original.data_ptr()
    )
    if materialized:
        _decline(
            f"materialized:{materialized}",
            f"flattening would materialize inputs {materialized}",
            *capture_inputs,
        )
        return None

    device = q.device
    q_map = execution.q.batch_indices.to(device=device, dtype=torch.int32)
    velocity_map = execution.values[0].batch_indices.to(device=device, dtype=torch.int32)
    acceleration_map = execution.values[1].batch_indices.to(device=device, dtype=torch.int32)
    fext_map = execution.values[2].batch_indices.to(device=device, dtype=torch.int32)
    placement_map = execution.values[3].batch_indices.to(device=device, dtype=torch.int32)
    inertia_map = execution.values[4].batch_indices.to(device=device, dtype=torch.int32)
    gravity_map = execution.values[6].batch_indices.to(device=device, dtype=torch.int32)
    flat_outputs = _WarpRNEAFunction.apply(
        q_unique,
        velocity_unique,
        acceleration_unique,
        fext_unique,
        placement_unique,
        inertia_unique,
        gravity_unique,
        q_map,
        velocity_map,
        acceleration_map,
        fext_map,
        placement_map,
        inertia_map,
        gravity_map,
        structure,
        values,
        fext is not None,
    )
    outputs = tuple(execution.unflatten(output) for output in flat_outputs)
    return RNEAResult(*outputs)


__all__ = ["try_warp_rnea"]
