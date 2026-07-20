# kinematics/ — Forward Kinematics and Jacobians

## Entry Points

- `forward_kinematics(model, q_or_data, compute_frames=False, use_warp=False)` — selects one whole FK pass, fills `joint_pose_world` (and `frame_pose_world` if `compute_frames=True`)
- `forward_kinematics_raw(structure, values, q)` — pure Torch pass returning a named `FKResult`
- `frame_placements_raw(structure, values, joint_pose_world)` — pure frame-table pass returning a named result
- `joint_jacobians_raw(structure, q, joint_pose_world)` — sequencing-free Jacobian pass returning a named result
- `frame_jacobian_raw(structure, values, q, joint_pose_world, frame_id, reference=..., joint_jacobians=None)` — pure one-frame Jacobian extraction with optional reuse of an existing joint-Jacobian pass
- `update_frame_placements(model, data)` — fills `frame_pose_world` from existing `joint_pose_world`
- `compute_joint_jacobians(model, data)` — fills `data.joint_jacobians` for all joints
- `get_frame_jacobian(model, data, frame_id, reference=...)` — extracts `(B..., 6, nv)` for one frame
- `get_joint_jacobian(model, data, joint_id, reference=...)` — same for joints

All passes use one execution batch: the right-aligned broadcast of `q` with
the joint-placement, body-inertia, and frame-placement value tables. Callers
must add semantic singleton axes explicitly (for example `(B, 1, njoints, 7)`
alongside `(B, T, nq)`); no auto-unsqueeze is performed.

## Jacobian Reference Frames (critical)

`get_frame_jacobian` returns **LOCAL_WORLD_ALIGNED** by default:
- Linear rows: velocity of frame origin in world frame
- Angular rows: angular velocity in world frame

To convert LWA to body-frame:
```python
# CORRECT — rotate only, don't apply full adjoint:
R_ee = so3.to_matrix(T_ee[..., 3:])
J_local = torch.cat([R_ee.mT @ J_world[..., :3, :], R_ee.mT @ J_world[..., 3:, :]], dim=-2)

# WRONG — adds spurious cross-term when J is LWA:
# J_local = se3.adjoint_inv(T_ee) @ J_world
```

`compute_joint_jacobians` returns WORLD-frame Jacobian (velocity at world origin).

## Residual Jacobian Strategy

Kinematics owns no residual-Jacobian dispatcher. Object-referenced `Problem` accepts
the literal strategies `"auto"`, `"analytic"`, `"jacrev"`, `"jacfwd"`, and
`"finite_difference"`; finite differences remain an explicit debug oracle.

## FK Hot Path

The Torch lane uses shared `joint_dispatch.joint_transform` and loops over
the static `ModelStructure.topo_order` tuple, which unrolls cleanly for
`torch.compile`. `use_warp=True` opts into the CUDA-validated fused whole-pass
FK lane; it remains non-default pending owner review of backward performance.
Unsupported runtime, kind, dtype, or layout cases fall back to the Torch raw
pass with a one-shot warning naming the reason; a decline during CUDA graph
capture raises. There is no per-Lie-operation or process-global compute
selection.
