"""Forward-only fused Warp kernels for inverse dynamics."""

from __future__ import annotations

import warp as wp

from ..kinematics._warp_kernels import (
    _compose_f32,
    _compose_f64,
    _joint_transform_reduced_f32,
    _joint_transform_reduced_f64,
)


@wp.func
def _load_spatial_f32(
    values: wp.array3d(dtype=wp.float32),
    execution_index: int,
    joint_index: int,
) -> wp.spatial_vectorf:
    """Load BetterRobot linear-first storage into Warp angular-first form."""
    return wp.spatial_vectorf(
        values[execution_index, joint_index, 3],
        values[execution_index, joint_index, 4],
        values[execution_index, joint_index, 5],
        values[execution_index, joint_index, 0],
        values[execution_index, joint_index, 1],
        values[execution_index, joint_index, 2],
    )


@wp.func
def _store_spatial_f32(
    values: wp.array3d(dtype=wp.float32),
    execution_index: int,
    joint_index: int,
    spatial: wp.spatial_vectorf,
):
    """Store a Warp angular-first vector in BetterRobot linear-first form."""
    angular = wp.spatial_top(spatial)
    linear = wp.spatial_bottom(spatial)
    values[execution_index, joint_index, 0] = linear[0]
    values[execution_index, joint_index, 1] = linear[1]
    values[execution_index, joint_index, 2] = linear[2]
    values[execution_index, joint_index, 3] = angular[0]
    values[execution_index, joint_index, 4] = angular[1]
    values[execution_index, joint_index, 5] = angular[2]


@wp.func
def _motion_subspace_f32(
    subspaces: wp.array3d(dtype=wp.float32),
    joint_index: int,
    coordinate: int,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(
        subspaces[joint_index, 3, coordinate],
        subspaces[joint_index, 4, coordinate],
        subspaces[joint_index, 5, coordinate],
        subspaces[joint_index, 0, coordinate],
        subspaces[joint_index, 1, coordinate],
        subspaces[joint_index, 2, coordinate],
    )


@wp.func
def _expanded_tangent_f32(
    value: wp.array2d(dtype=wp.float32),
    value_row: int,
    joint_index: int,
    coordinate: int,
    idx_vs: wp.array(dtype=wp.int32),
    idx_vs_full: wp.array(dtype=wp.int32),
    mimic_sources: wp.array(dtype=wp.int32),
    v_expansion: wp.array2d(dtype=wp.float32),
    nv: int,
) -> wp.float32:
    if mimic_sources[joint_index] == joint_index:
        return value[value_row, idx_vs[joint_index] + coordinate]
    full_index = idx_vs_full[joint_index] + coordinate
    expanded = wp.float32(0.0)
    for reduced_index in range(nv):
        expanded += v_expansion[full_index, reduced_index] * value[value_row, reduced_index]
    return expanded


@wp.func
def _inertia_product_f32(
    inertias: wp.array3d(dtype=wp.float32),
    value_row: int,
    joint_index: int,
    motion: wp.spatial_vectorf,
) -> wp.spatial_vectorf:
    mass = inertias[value_row, joint_index, 0]
    com = wp.vec3f(
        inertias[value_row, joint_index, 1],
        inertias[value_row, joint_index, 2],
        inertias[value_row, joint_index, 3],
    )
    angular = wp.spatial_top(motion)
    linear = wp.spatial_bottom(motion)
    force = mass * (linear - wp.cross(com, angular))
    inertia_angular = wp.vec3f(
        inertias[value_row, joint_index, 4] * angular[0]
        + inertias[value_row, joint_index, 7] * angular[1]
        + inertias[value_row, joint_index, 8] * angular[2],
        inertias[value_row, joint_index, 7] * angular[0]
        + inertias[value_row, joint_index, 5] * angular[1]
        + inertias[value_row, joint_index, 9] * angular[2],
        inertias[value_row, joint_index, 8] * angular[0]
        + inertias[value_row, joint_index, 9] * angular[1]
        + inertias[value_row, joint_index, 6] * angular[2],
    )
    torque = inertia_angular + wp.cross(com, force)
    return wp.spatial_vectorf(torque[0], torque[1], torque[2], force[0], force[1], force[2])


@wp.func
def _external_force_f32(
    fext: wp.array3d(dtype=wp.float32),
    value_row: int,
    joint_index: int,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(
        fext[value_row, joint_index, 3],
        fext[value_row, joint_index, 4],
        fext[value_row, joint_index, 5],
        fext[value_row, joint_index, 0],
        fext[value_row, joint_index, 1],
        fext[value_row, joint_index, 2],
    )


@wp.func
def _rotate_f32(rotation: wp.quatf, vector: wp.vec3f) -> wp.vec3f:
    """Rotate with the same off-manifold polynomial as Torch ``so3.act``."""
    imaginary = wp.vec3f(rotation[0], rotation[1], rotation[2])
    return vector + wp.float32(2.0) * wp.cross(
        imaginary,
        wp.cross(imaginary, vector) + rotation[3] * vector,
    )


@wp.func
def _rotate_transpose_f32(rotation: wp.quatf, vector: wp.vec3f) -> wp.vec3f:
    imaginary = wp.vec3f(rotation[0], rotation[1], rotation[2])
    return vector + wp.float32(2.0) * wp.cross(
        imaginary,
        wp.cross(imaginary, vector) - rotation[3] * vector,
    )


@wp.func
def _transport_motion_inverse_f32(
    pose: wp.transformf,
    motion: wp.spatial_vectorf,
) -> wp.spatial_vectorf:
    translation = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    angular_parent = wp.spatial_top(motion)
    linear_parent = wp.spatial_bottom(motion)
    angular_child = _rotate_transpose_f32(rotation, angular_parent)
    linear_child = _rotate_transpose_f32(
        rotation,
        linear_parent - wp.cross(translation, angular_parent),
    )
    return wp.spatial_vectorf(
        angular_child[0],
        angular_child[1],
        angular_child[2],
        linear_child[0],
        linear_child[1],
        linear_child[2],
    )


@wp.func
def _transport_force_f32(
    pose: wp.transformf,
    force: wp.spatial_vectorf,
) -> wp.spatial_vectorf:
    translation = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    torque_child = wp.spatial_top(force)
    force_child = wp.spatial_bottom(force)
    force_parent = _rotate_f32(rotation, force_child)
    torque_parent = _rotate_f32(rotation, torque_child) + wp.cross(
        translation,
        force_parent,
    )
    return wp.spatial_vectorf(
        torque_parent[0],
        torque_parent[1],
        torque_parent[2],
        force_parent[0],
        force_parent[1],
        force_parent[2],
    )


@wp.kernel
def rnea_f32(  # noqa: PLR0912, PLR0913, PLR0915 - one explicit recursive pass
    q: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    acceleration: wp.array2d(dtype=wp.float32),
    fext: wp.array3d(dtype=wp.float32),
    joint_placements: wp.array2d(dtype=wp.transformf),
    body_inertias: wp.array3d(dtype=wp.float32),
    gravity: wp.array2d(dtype=wp.float32),
    q_map: wp.array(dtype=wp.int32),
    velocity_map: wp.array(dtype=wp.int32),
    acceleration_map: wp.array(dtype=wp.int32),
    fext_map: wp.array(dtype=wp.int32),
    placement_map: wp.array(dtype=wp.int32),
    inertia_map: wp.array(dtype=wp.int32),
    gravity_map: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    topo_order: wp.array(dtype=wp.int32),
    kinds: wp.array(dtype=wp.int8),
    idx_qs: wp.array(dtype=wp.int32),
    idx_qs_full: wp.array(dtype=wp.int32),
    idx_vs: wp.array(dtype=wp.int32),
    idx_vs_full: wp.array(dtype=wp.int32),
    nvs_full: wp.array(dtype=wp.int32),
    axes: wp.array(dtype=wp.vec3f),
    pitches: wp.array(dtype=wp.float32),
    motion_subspaces: wp.array3d(dtype=wp.float32),
    mimic_sources: wp.array(dtype=wp.int32),
    q_expansion: wp.array2d(dtype=wp.float32),
    q_offsets: wp.array(dtype=wp.float32),
    v_expansion: wp.array2d(dtype=wp.float32),
    nq: int,
    nv: int,
    njoints: int,
    has_fext: bool,
    tau_out: wp.array2d(dtype=wp.float32),
    local_out: wp.array2d(dtype=wp.transformf),
    world_out: wp.array2d(dtype=wp.transformf),
    velocity_out: wp.array3d(dtype=wp.float32),
    acceleration_out: wp.array3d(dtype=wp.float32),
    force_out: wp.array3d(dtype=wp.float32),
):
    execution_index = wp.tid()
    q_row = q_map[execution_index]
    velocity_row = velocity_map[execution_index]
    acceleration_row = acceleration_map[execution_index]
    fext_row = fext_map[execution_index]
    placement_row = placement_map[execution_index]
    inertia_row = inertia_map[execution_index]
    gravity_row = gravity_map[execution_index]
    zero = wp.spatial_vectorf()

    for reduced_index in range(nv):
        tau_out[execution_index, reduced_index] = wp.float32(0.0)

    gravity_motion = wp.spatial_vectorf(
        -gravity[gravity_row, 3],
        -gravity[gravity_row, 4],
        -gravity[gravity_row, 5],
        -gravity[gravity_row, 0],
        -gravity[gravity_row, 1],
        -gravity[gravity_row, 2],
    )
    _store_spatial_f32(velocity_out, execution_index, 0, zero)
    _store_spatial_f32(acceleration_out, execution_index, 0, gravity_motion)
    _store_spatial_f32(force_out, execution_index, 0, zero)

    for order_index in range(njoints):
        joint_index = topo_order[order_index]
        joint_delta = _joint_transform_reduced_f32(
            kinds[joint_index],
            q_row,
            joint_index,
            q,
            idx_qs,
            idx_qs_full,
            mimic_sources,
            q_expansion,
            q_offsets,
            nq,
            axes[joint_index],
            pitches[joint_index],
        )
        local_pose = _compose_f32(joint_placements[placement_row, joint_index], joint_delta)
        local_out[execution_index, joint_index] = local_pose
        parent = parents[joint_index]
        if parent < 0:
            world_out[execution_index, joint_index] = local_pose
        else:
            world_out[execution_index, joint_index] = _compose_f32(
                world_out[execution_index, parent],
                local_pose,
            )

        if joint_index == 0:
            continue

        joint_velocity = zero
        joint_acceleration = zero
        coordinate_count = nvs_full[joint_index]
        for coordinate in range(coordinate_count):
            subspace = _motion_subspace_f32(motion_subspaces, joint_index, coordinate)
            velocity_value = _expanded_tangent_f32(
                velocity,
                velocity_row,
                joint_index,
                coordinate,
                idx_vs,
                idx_vs_full,
                mimic_sources,
                v_expansion,
                nv,
            )
            acceleration_value = _expanded_tangent_f32(
                acceleration,
                acceleration_row,
                joint_index,
                coordinate,
                idx_vs,
                idx_vs_full,
                mimic_sources,
                v_expansion,
                nv,
            )
            joint_velocity += subspace * velocity_value
            joint_acceleration += subspace * acceleration_value

        parent_velocity = _transport_motion_inverse_f32(
            local_pose,
            _load_spatial_f32(velocity_out, execution_index, parent),
        )
        parent_acceleration = _transport_motion_inverse_f32(
            local_pose,
            _load_spatial_f32(acceleration_out, execution_index, parent),
        )
        body_velocity = parent_velocity + joint_velocity
        body_acceleration = parent_acceleration + wp.spatial_cross(body_velocity, joint_velocity) + joint_acceleration
        momentum = _inertia_product_f32(
            body_inertias,
            inertia_row,
            joint_index,
            body_velocity,
        )
        body_force = _inertia_product_f32(
            body_inertias,
            inertia_row,
            joint_index,
            body_acceleration,
        ) + wp.spatial_cross_dual(body_velocity, momentum)
        if has_fext:
            body_force -= _external_force_f32(fext, fext_row, joint_index)
        _store_spatial_f32(velocity_out, execution_index, joint_index, body_velocity)
        _store_spatial_f32(acceleration_out, execution_index, joint_index, body_acceleration)
        _store_spatial_f32(force_out, execution_index, joint_index, body_force)

    for reverse_index in range(njoints):
        joint_index = topo_order[njoints - reverse_index - 1]
        if joint_index == 0:
            continue
        body_force = _load_spatial_f32(force_out, execution_index, joint_index)
        coordinate_count = nvs_full[joint_index]
        for coordinate in range(coordinate_count):
            subspace = _motion_subspace_f32(motion_subspaces, joint_index, coordinate)
            full_index = idx_vs_full[joint_index] + coordinate
            projected = wp.spatial_dot(subspace, body_force)
            if mimic_sources[joint_index] == joint_index:
                reduced_index = idx_vs[joint_index] + coordinate
                tau_out[execution_index, reduced_index] += projected
            else:
                for reduced_index in range(nv):
                    tau_out[execution_index, reduced_index] += v_expansion[full_index, reduced_index] * projected

        parent = parents[joint_index]
        if parent >= 0:
            parent_force = _load_spatial_f32(force_out, execution_index, parent)
            parent_force += _transport_force_f32(
                local_out[execution_index, joint_index],
                body_force,
            )
            _store_spatial_f32(force_out, execution_index, parent, parent_force)


@wp.func
def _load_spatial_f64(
    values: wp.array3d(dtype=wp.float64),
    execution_index: int,
    joint_index: int,
) -> wp.spatial_vectord:
    return wp.spatial_vectord(
        values[execution_index, joint_index, 3],
        values[execution_index, joint_index, 4],
        values[execution_index, joint_index, 5],
        values[execution_index, joint_index, 0],
        values[execution_index, joint_index, 1],
        values[execution_index, joint_index, 2],
    )


@wp.func
def _store_spatial_f64(
    values: wp.array3d(dtype=wp.float64),
    execution_index: int,
    joint_index: int,
    spatial: wp.spatial_vectord,
):
    angular = wp.spatial_top(spatial)
    linear = wp.spatial_bottom(spatial)
    values[execution_index, joint_index, 0] = linear[0]
    values[execution_index, joint_index, 1] = linear[1]
    values[execution_index, joint_index, 2] = linear[2]
    values[execution_index, joint_index, 3] = angular[0]
    values[execution_index, joint_index, 4] = angular[1]
    values[execution_index, joint_index, 5] = angular[2]


@wp.func
def _motion_subspace_f64(
    subspaces: wp.array3d(dtype=wp.float64),
    joint_index: int,
    coordinate: int,
) -> wp.spatial_vectord:
    return wp.spatial_vectord(
        subspaces[joint_index, 3, coordinate],
        subspaces[joint_index, 4, coordinate],
        subspaces[joint_index, 5, coordinate],
        subspaces[joint_index, 0, coordinate],
        subspaces[joint_index, 1, coordinate],
        subspaces[joint_index, 2, coordinate],
    )


@wp.func
def _expanded_tangent_f64(
    value: wp.array2d(dtype=wp.float64),
    value_row: int,
    joint_index: int,
    coordinate: int,
    idx_vs: wp.array(dtype=wp.int32),
    idx_vs_full: wp.array(dtype=wp.int32),
    mimic_sources: wp.array(dtype=wp.int32),
    v_expansion: wp.array2d(dtype=wp.float64),
    nv: int,
) -> wp.float64:
    if mimic_sources[joint_index] == joint_index:
        return value[value_row, idx_vs[joint_index] + coordinate]
    full_index = idx_vs_full[joint_index] + coordinate
    expanded = wp.float64(0.0)
    for reduced_index in range(nv):
        expanded += v_expansion[full_index, reduced_index] * value[value_row, reduced_index]
    return expanded


@wp.func
def _inertia_product_f64(
    inertias: wp.array3d(dtype=wp.float64),
    value_row: int,
    joint_index: int,
    motion: wp.spatial_vectord,
) -> wp.spatial_vectord:
    mass = inertias[value_row, joint_index, 0]
    com = wp.vec3d(
        inertias[value_row, joint_index, 1],
        inertias[value_row, joint_index, 2],
        inertias[value_row, joint_index, 3],
    )
    angular = wp.spatial_top(motion)
    linear = wp.spatial_bottom(motion)
    force = mass * (linear - wp.cross(com, angular))
    inertia_angular = wp.vec3d(
        inertias[value_row, joint_index, 4] * angular[0]
        + inertias[value_row, joint_index, 7] * angular[1]
        + inertias[value_row, joint_index, 8] * angular[2],
        inertias[value_row, joint_index, 7] * angular[0]
        + inertias[value_row, joint_index, 5] * angular[1]
        + inertias[value_row, joint_index, 9] * angular[2],
        inertias[value_row, joint_index, 8] * angular[0]
        + inertias[value_row, joint_index, 9] * angular[1]
        + inertias[value_row, joint_index, 6] * angular[2],
    )
    torque = inertia_angular + wp.cross(com, force)
    return wp.spatial_vectord(torque[0], torque[1], torque[2], force[0], force[1], force[2])


@wp.func
def _external_force_f64(
    fext: wp.array3d(dtype=wp.float64),
    value_row: int,
    joint_index: int,
) -> wp.spatial_vectord:
    return wp.spatial_vectord(
        fext[value_row, joint_index, 3],
        fext[value_row, joint_index, 4],
        fext[value_row, joint_index, 5],
        fext[value_row, joint_index, 0],
        fext[value_row, joint_index, 1],
        fext[value_row, joint_index, 2],
    )


@wp.func
def _rotate_f64(rotation: wp.quatd, vector: wp.vec3d) -> wp.vec3d:
    """Float64 counterpart of :func:`_rotate_f32`."""
    imaginary = wp.vec3d(rotation[0], rotation[1], rotation[2])
    return vector + wp.float64(2.0) * wp.cross(
        imaginary,
        wp.cross(imaginary, vector) + rotation[3] * vector,
    )


@wp.func
def _rotate_transpose_f64(rotation: wp.quatd, vector: wp.vec3d) -> wp.vec3d:
    imaginary = wp.vec3d(rotation[0], rotation[1], rotation[2])
    return vector + wp.float64(2.0) * wp.cross(
        imaginary,
        wp.cross(imaginary, vector) - rotation[3] * vector,
    )


@wp.func
def _transport_motion_inverse_f64(
    pose: wp.transformd,
    motion: wp.spatial_vectord,
) -> wp.spatial_vectord:
    translation = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    angular_parent = wp.spatial_top(motion)
    linear_parent = wp.spatial_bottom(motion)
    angular_child = _rotate_transpose_f64(rotation, angular_parent)
    linear_child = _rotate_transpose_f64(
        rotation,
        linear_parent - wp.cross(translation, angular_parent),
    )
    return wp.spatial_vectord(
        angular_child[0],
        angular_child[1],
        angular_child[2],
        linear_child[0],
        linear_child[1],
        linear_child[2],
    )


@wp.func
def _transport_force_f64(
    pose: wp.transformd,
    force: wp.spatial_vectord,
) -> wp.spatial_vectord:
    translation = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    torque_child = wp.spatial_top(force)
    force_child = wp.spatial_bottom(force)
    force_parent = _rotate_f64(rotation, force_child)
    torque_parent = _rotate_f64(rotation, torque_child) + wp.cross(
        translation,
        force_parent,
    )
    return wp.spatial_vectord(
        torque_parent[0],
        torque_parent[1],
        torque_parent[2],
        force_parent[0],
        force_parent[1],
        force_parent[2],
    )


@wp.kernel
def rnea_f64(  # noqa: PLR0912, PLR0913, PLR0915 - one explicit recursive pass
    q: wp.array2d(dtype=wp.float64),
    velocity: wp.array2d(dtype=wp.float64),
    acceleration: wp.array2d(dtype=wp.float64),
    fext: wp.array3d(dtype=wp.float64),
    joint_placements: wp.array2d(dtype=wp.transformd),
    body_inertias: wp.array3d(dtype=wp.float64),
    gravity: wp.array2d(dtype=wp.float64),
    q_map: wp.array(dtype=wp.int32),
    velocity_map: wp.array(dtype=wp.int32),
    acceleration_map: wp.array(dtype=wp.int32),
    fext_map: wp.array(dtype=wp.int32),
    placement_map: wp.array(dtype=wp.int32),
    inertia_map: wp.array(dtype=wp.int32),
    gravity_map: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    topo_order: wp.array(dtype=wp.int32),
    kinds: wp.array(dtype=wp.int8),
    idx_qs: wp.array(dtype=wp.int32),
    idx_qs_full: wp.array(dtype=wp.int32),
    idx_vs: wp.array(dtype=wp.int32),
    idx_vs_full: wp.array(dtype=wp.int32),
    nvs_full: wp.array(dtype=wp.int32),
    axes: wp.array(dtype=wp.vec3d),
    pitches: wp.array(dtype=wp.float64),
    motion_subspaces: wp.array3d(dtype=wp.float64),
    mimic_sources: wp.array(dtype=wp.int32),
    q_expansion: wp.array2d(dtype=wp.float64),
    q_offsets: wp.array(dtype=wp.float64),
    v_expansion: wp.array2d(dtype=wp.float64),
    nq: int,
    nv: int,
    njoints: int,
    has_fext: bool,
    tau_out: wp.array2d(dtype=wp.float64),
    local_out: wp.array2d(dtype=wp.transformd),
    world_out: wp.array2d(dtype=wp.transformd),
    velocity_out: wp.array3d(dtype=wp.float64),
    acceleration_out: wp.array3d(dtype=wp.float64),
    force_out: wp.array3d(dtype=wp.float64),
):
    execution_index = wp.tid()
    q_row = q_map[execution_index]
    velocity_row = velocity_map[execution_index]
    acceleration_row = acceleration_map[execution_index]
    fext_row = fext_map[execution_index]
    placement_row = placement_map[execution_index]
    inertia_row = inertia_map[execution_index]
    gravity_row = gravity_map[execution_index]
    zero = wp.spatial_vectord()

    for reduced_index in range(nv):
        tau_out[execution_index, reduced_index] = wp.float64(0.0)

    gravity_motion = wp.spatial_vectord(
        -gravity[gravity_row, 3],
        -gravity[gravity_row, 4],
        -gravity[gravity_row, 5],
        -gravity[gravity_row, 0],
        -gravity[gravity_row, 1],
        -gravity[gravity_row, 2],
    )
    _store_spatial_f64(velocity_out, execution_index, 0, zero)
    _store_spatial_f64(acceleration_out, execution_index, 0, gravity_motion)
    _store_spatial_f64(force_out, execution_index, 0, zero)

    for order_index in range(njoints):
        joint_index = topo_order[order_index]
        joint_delta = _joint_transform_reduced_f64(
            kinds[joint_index],
            q_row,
            joint_index,
            q,
            idx_qs,
            idx_qs_full,
            mimic_sources,
            q_expansion,
            q_offsets,
            nq,
            axes[joint_index],
            pitches[joint_index],
        )
        local_pose = _compose_f64(joint_placements[placement_row, joint_index], joint_delta)
        local_out[execution_index, joint_index] = local_pose
        parent = parents[joint_index]
        if parent < 0:
            world_out[execution_index, joint_index] = local_pose
        else:
            world_out[execution_index, joint_index] = _compose_f64(
                world_out[execution_index, parent],
                local_pose,
            )

        if joint_index == 0:
            continue

        joint_velocity = zero
        joint_acceleration = zero
        coordinate_count = nvs_full[joint_index]
        for coordinate in range(coordinate_count):
            subspace = _motion_subspace_f64(motion_subspaces, joint_index, coordinate)
            velocity_value = _expanded_tangent_f64(
                velocity,
                velocity_row,
                joint_index,
                coordinate,
                idx_vs,
                idx_vs_full,
                mimic_sources,
                v_expansion,
                nv,
            )
            acceleration_value = _expanded_tangent_f64(
                acceleration,
                acceleration_row,
                joint_index,
                coordinate,
                idx_vs,
                idx_vs_full,
                mimic_sources,
                v_expansion,
                nv,
            )
            joint_velocity += subspace * velocity_value
            joint_acceleration += subspace * acceleration_value

        parent_velocity = _transport_motion_inverse_f64(
            local_pose,
            _load_spatial_f64(velocity_out, execution_index, parent),
        )
        parent_acceleration = _transport_motion_inverse_f64(
            local_pose,
            _load_spatial_f64(acceleration_out, execution_index, parent),
        )
        body_velocity = parent_velocity + joint_velocity
        body_acceleration = parent_acceleration + wp.spatial_cross(body_velocity, joint_velocity) + joint_acceleration
        momentum = _inertia_product_f64(
            body_inertias,
            inertia_row,
            joint_index,
            body_velocity,
        )
        body_force = _inertia_product_f64(
            body_inertias,
            inertia_row,
            joint_index,
            body_acceleration,
        ) + wp.spatial_cross_dual(body_velocity, momentum)
        if has_fext:
            body_force -= _external_force_f64(fext, fext_row, joint_index)
        _store_spatial_f64(velocity_out, execution_index, joint_index, body_velocity)
        _store_spatial_f64(acceleration_out, execution_index, joint_index, body_acceleration)
        _store_spatial_f64(force_out, execution_index, joint_index, body_force)

    for reverse_index in range(njoints):
        joint_index = topo_order[njoints - reverse_index - 1]
        if joint_index == 0:
            continue
        body_force = _load_spatial_f64(force_out, execution_index, joint_index)
        coordinate_count = nvs_full[joint_index]
        for coordinate in range(coordinate_count):
            subspace = _motion_subspace_f64(motion_subspaces, joint_index, coordinate)
            full_index = idx_vs_full[joint_index] + coordinate
            projected = wp.spatial_dot(subspace, body_force)
            if mimic_sources[joint_index] == joint_index:
                reduced_index = idx_vs[joint_index] + coordinate
                tau_out[execution_index, reduced_index] += projected
            else:
                for reduced_index in range(nv):
                    tau_out[execution_index, reduced_index] += v_expansion[full_index, reduced_index] * projected

        parent = parents[joint_index]
        if parent >= 0:
            parent_force = _load_spatial_f64(force_out, execution_index, parent)
            parent_force += _transport_force_f64(
                local_out[execution_index, joint_index],
                body_force,
            )
            _store_spatial_f64(force_out, execution_index, parent, parent_force)


__all__ = ["rnea_f32", "rnea_f64"]
