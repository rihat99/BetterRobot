# kinematics/ — Forward Kinematics and Jacobians

## Entry Points

- `forward_kinematics(model, q_or_data, compute_frames=False, use_warp=False)` — selects one whole FK pass, fills `joint_pose_world` (and `frame_pose_world` if `compute_frames=True`)
- `forward_kinematics_raw(structure, values, q)` — pure Torch pass over the `ModelStructure` / `ModelValues` seam
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

## Jacobian Strategy

```python
class JacobianStrategy(str, Enum):
    ANALYTIC = "analytic"     # call residual.jacobian(state)
    FINITE_DIFF = "finite_diff" # central FD, 2*nv + 1 evaluations
    AUTO = "auto"             # prefer analytic, fall back to finite diff
```

`residual_jacobian`'s AUTO fallback is unbatched central finite differences:
one base evaluation plus two evaluations per tangent dimension. Real
`torch.func` strategies are scheduled for the M2 residual redesign and are
not selectable today.

## FK Hot Path

The Torch lane uses shared `joint_dispatch.joint_transform` and loops over
the static `ModelStructure.topo_order` tuple, which unrolls cleanly for
`torch.compile`. `use_warp=True` opts into the fused whole-pass prototype;
unsupported runtime, kind, dtype, or layout cases intentionally fall back to
the Torch raw pass. There is no per-Lie-operation or process-global compute
selection.
