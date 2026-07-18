"""CUDA-validated opt-in Warp FK kernels.

One thread owns one execution-batch element and performs a serial
topological sweep. These kernels are forward-only: their
PyTorch custom-op wrapper supplies the VJP by recomputing the torch lane,
avoiding Warp's unsafe generated adjoint for dynamic loops.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _compose_f32(a: wp.transformf, b: wp.transformf) -> wp.transformf:
    """Compose transforms with the exact polynomial used by the torch lane.

    Warp's built-in transform multiplication is equivalent on unit
    quaternions, but its off-manifold derivative is not the derivative of
    ``better_robot.lie._impl.se3_compose``.  Placement tensors are public,
    differentiable values, so the fused lane must preserve that derivative as
    well as the on-manifold forward result.
    """
    a_t = wp.transform_get_translation(a)
    a_q = wp.transform_get_rotation(a)
    b_t = wp.transform_get_translation(b)
    b_q = wp.transform_get_rotation(b)

    a_xyz = wp.vec3f(a_q[0], a_q[1], a_q[2])
    rotated = b_t + wp.float32(2.0) * wp.cross(
        a_xyz, wp.cross(a_xyz, b_t) + a_q[3] * b_t
    )
    c_t = a_t + rotated
    c_q = wp.quatf(
        a_q[3] * b_q[0] + a_q[0] * b_q[3] + a_q[1] * b_q[2] - a_q[2] * b_q[1],
        a_q[3] * b_q[1] - a_q[0] * b_q[2] + a_q[1] * b_q[3] + a_q[2] * b_q[0],
        a_q[3] * b_q[2] + a_q[0] * b_q[1] - a_q[1] * b_q[0] + a_q[2] * b_q[3],
        a_q[3] * b_q[3] - a_q[0] * b_q[0] - a_q[1] * b_q[1] - a_q[2] * b_q[2],
    )
    return wp.transformf(c_t, c_q)


@wp.func
def _compose_f64(a: wp.transformd, b: wp.transformd) -> wp.transformd:
    """Float64 counterpart of :func:`_compose_f32`."""
    a_t = wp.transform_get_translation(a)
    a_q = wp.transform_get_rotation(a)
    b_t = wp.transform_get_translation(b)
    b_q = wp.transform_get_rotation(b)

    a_xyz = wp.vec3d(a_q[0], a_q[1], a_q[2])
    rotated = b_t + wp.float64(2.0) * wp.cross(
        a_xyz, wp.cross(a_xyz, b_t) + a_q[3] * b_t
    )
    c_t = a_t + rotated
    c_q = wp.quatd(
        a_q[3] * b_q[0] + a_q[0] * b_q[3] + a_q[1] * b_q[2] - a_q[2] * b_q[1],
        a_q[3] * b_q[1] - a_q[0] * b_q[2] + a_q[1] * b_q[3] + a_q[2] * b_q[0],
        a_q[3] * b_q[2] + a_q[0] * b_q[1] - a_q[1] * b_q[0] + a_q[2] * b_q[3],
        a_q[3] * b_q[3] - a_q[0] * b_q[0] - a_q[1] * b_q[1] - a_q[2] * b_q[2],
    )
    return wp.transformd(c_t, c_q)


@wp.func
def _joint_transform_f32(  # noqa: PLR0911
    kind: wp.int8,
    q_row: int,
    q_index: int,
    q: wp.array2d(dtype=wp.float32),
    axis: wp.vec3f,
    pitch: wp.float32,
) -> wp.transformf:
    zero = wp.vec3f(0.0, 0.0, 0.0)
    identity = wp.quatf(0.0, 0.0, 0.0, 1.0)
    if kind == 2 or kind == 3 or kind == 4 or kind == 5:  # noqa: PLR1714 -- Warp does not support Python set membership
        return wp.transformf(zero, wp.quat_from_axis_angle(axis, q[q_row, q_index]))
    if kind == 6:
        angle = wp.atan2(q[q_row, q_index + 1], q[q_row, q_index])
        return wp.transformf(zero, wp.quat_from_axis_angle(axis, angle))
    if kind == 7 or kind == 8 or kind == 9 or kind == 10:  # noqa: PLR1714 -- Warp does not support Python set membership
        return wp.transformf(axis * q[q_row, q_index], identity)
    if kind == 11:
        rotation = wp.normalize(
            wp.quatf(
                q[q_row, q_index],
                q[q_row, q_index + 1],
                q[q_row, q_index + 2],
                q[q_row, q_index + 3],
            )
        )
        return wp.transformf(zero, rotation)
    if kind == 12:
        translation = wp.vec3f(
            q[q_row, q_index],
            q[q_row, q_index + 1],
            q[q_row, q_index + 2],
        )
        rotation = wp.normalize(
            wp.quatf(
                q[q_row, q_index + 3],
                q[q_row, q_index + 4],
                q[q_row, q_index + 5],
                q[q_row, q_index + 6],
            )
        )
        return wp.transformf(translation, rotation)
    if kind == 13:
        x = q[q_row, q_index]
        y = q[q_row, q_index + 1]
        cosine = q[q_row, q_index + 2]
        sine = q[q_row, q_index + 3]
        half_cosine = wp.sqrt(wp.max((1.0 + cosine) * 0.5, 0.0))
        half_sine = wp.sqrt(wp.max((1.0 - cosine) * 0.5, 0.0)) * wp.sign(sine)
        return wp.transformf(
            wp.vec3f(x, y, 0.0), wp.quatf(0.0, 0.0, half_sine, half_cosine)
        )
    if kind == 14:
        return wp.transformf(
            wp.vec3f(
                q[q_row, q_index],
                q[q_row, q_index + 1],
                q[q_row, q_index + 2],
            ),
            identity,
        )
    if kind == 15:
        angle = q[q_row, q_index]
        rotation_pose = wp.transformf(zero, wp.quat_from_axis_angle(axis, angle))
        translation_pose = wp.transformf(axis * (pitch * angle), identity)
        return _compose_f32(translation_pose, rotation_pose)
    return wp.transformf(zero, identity)


@wp.func
def _joint_transform_f64(  # noqa: PLR0911
    kind: wp.int8,
    q_row: int,
    q_index: int,
    q: wp.array2d(dtype=wp.float64),
    axis: wp.vec3d,
    pitch: wp.float64,
) -> wp.transformd:
    zero = wp.vec3d(0.0, 0.0, 0.0)
    identity = wp.quatd(0.0, 0.0, 0.0, 1.0)
    if kind == 2 or kind == 3 or kind == 4 or kind == 5:  # noqa: PLR1714 -- Warp does not support Python set membership
        return wp.transformd(zero, wp.quat_from_axis_angle(axis, q[q_row, q_index]))
    if kind == 6:
        angle = wp.atan2(q[q_row, q_index + 1], q[q_row, q_index])
        return wp.transformd(zero, wp.quat_from_axis_angle(axis, angle))
    if kind == 7 or kind == 8 or kind == 9 or kind == 10:  # noqa: PLR1714 -- Warp does not support Python set membership
        return wp.transformd(axis * q[q_row, q_index], identity)
    if kind == 11:
        rotation = wp.normalize(
            wp.quatd(
                q[q_row, q_index],
                q[q_row, q_index + 1],
                q[q_row, q_index + 2],
                q[q_row, q_index + 3],
            )
        )
        return wp.transformd(zero, rotation)
    if kind == 12:
        translation = wp.vec3d(
            q[q_row, q_index],
            q[q_row, q_index + 1],
            q[q_row, q_index + 2],
        )
        rotation = wp.normalize(
            wp.quatd(
                q[q_row, q_index + 3],
                q[q_row, q_index + 4],
                q[q_row, q_index + 5],
                q[q_row, q_index + 6],
            )
        )
        return wp.transformd(translation, rotation)
    if kind == 13:
        x = q[q_row, q_index]
        y = q[q_row, q_index + 1]
        cosine = q[q_row, q_index + 2]
        sine = q[q_row, q_index + 3]
        half_cosine = wp.sqrt(wp.max((wp.float64(1.0) + cosine) * wp.float64(0.5), wp.float64(0.0)))
        half_sine = wp.sqrt(wp.max((wp.float64(1.0) - cosine) * wp.float64(0.5), wp.float64(0.0))) * wp.sign(sine)
        return wp.transformd(
            wp.vec3d(x, y, wp.float64(0.0)),
            wp.quatd(wp.float64(0.0), wp.float64(0.0), half_sine, half_cosine),
        )
    if kind == 14:
        return wp.transformd(
            wp.vec3d(
                q[q_row, q_index],
                q[q_row, q_index + 1],
                q[q_row, q_index + 2],
            ),
            identity,
        )
    if kind == 15:
        angle = q[q_row, q_index]
        rotation_pose = wp.transformd(zero, wp.quat_from_axis_angle(axis, angle))
        translation_pose = wp.transformd(axis * (pitch * angle), identity)
        return _compose_f64(translation_pose, rotation_pose)
    return wp.transformd(zero, identity)


@wp.kernel
def fk_frames_f32(
    q: wp.array2d(dtype=wp.float32),
    joint_placements: wp.array2d(dtype=wp.transformf),
    frame_placements: wp.array2d(dtype=wp.transformf),
    q_map: wp.array(dtype=wp.int32),
    value_map: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    topo_order: wp.array(dtype=wp.int32),
    kinds: wp.array(dtype=wp.int8),
    idx_qs: wp.array(dtype=wp.int32),
    axes: wp.array(dtype=wp.vec3f),
    pitches: wp.array(dtype=wp.float32),
    frame_parents: wp.array(dtype=wp.int32),
    njoints: int,
    nframes: int,
    local_out: wp.array2d(dtype=wp.transformf),
    world_out: wp.array2d(dtype=wp.transformf),
    frame_out: wp.array2d(dtype=wp.transformf),
):
    execution_index = wp.tid()
    q_row = q_map[execution_index]
    value_row = value_map[execution_index]
    for order_index in range(njoints):
        joint_index = topo_order[order_index]
        joint_delta = _joint_transform_f32(
            kinds[joint_index],
            q_row,
            idx_qs[joint_index],
            q,
            axes[joint_index],
            pitches[joint_index],
        )
        local_pose = _compose_f32(
            joint_placements[value_row, joint_index], joint_delta
        )
        local_out[execution_index, joint_index] = local_pose
        parent = parents[joint_index]
        if parent < 0:
            world_out[execution_index, joint_index] = local_pose
        else:
            world_out[execution_index, joint_index] = _compose_f32(
                world_out[execution_index, parent], local_pose
            )
    for frame_index in range(nframes):
        parent = frame_parents[frame_index]
        frame_out[execution_index, frame_index] = _compose_f32(
            world_out[execution_index, parent],
            frame_placements[value_row, frame_index],
        )


@wp.kernel
def fk_frames_f64(
    q: wp.array2d(dtype=wp.float64),
    joint_placements: wp.array2d(dtype=wp.transformd),
    frame_placements: wp.array2d(dtype=wp.transformd),
    q_map: wp.array(dtype=wp.int32),
    value_map: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    topo_order: wp.array(dtype=wp.int32),
    kinds: wp.array(dtype=wp.int8),
    idx_qs: wp.array(dtype=wp.int32),
    axes: wp.array(dtype=wp.vec3d),
    pitches: wp.array(dtype=wp.float64),
    frame_parents: wp.array(dtype=wp.int32),
    njoints: int,
    nframes: int,
    local_out: wp.array2d(dtype=wp.transformd),
    world_out: wp.array2d(dtype=wp.transformd),
    frame_out: wp.array2d(dtype=wp.transformd),
):
    execution_index = wp.tid()
    q_row = q_map[execution_index]
    value_row = value_map[execution_index]
    for order_index in range(njoints):
        joint_index = topo_order[order_index]
        joint_delta = _joint_transform_f64(
            kinds[joint_index],
            q_row,
            idx_qs[joint_index],
            q,
            axes[joint_index],
            pitches[joint_index],
        )
        local_pose = _compose_f64(
            joint_placements[value_row, joint_index], joint_delta
        )
        local_out[execution_index, joint_index] = local_pose
        parent = parents[joint_index]
        if parent < 0:
            world_out[execution_index, joint_index] = local_pose
        else:
            world_out[execution_index, joint_index] = _compose_f64(
                world_out[execution_index, parent], local_pose
            )
    for frame_index in range(nframes):
        parent = frame_parents[frame_index]
        frame_out[execution_index, frame_index] = _compose_f64(
            world_out[execution_index, parent],
            frame_placements[value_row, frame_index],
        )


__all__ = ["fk_frames_f32", "fk_frames_f64"]
