# kinematics/ — Forward Kinematics and Jacobians

## Entry Points

- `forward_kinematics(model, q_or_data, compute_frames=False)` — single topological pass, fills `oMi` (and `oMf` if `compute_frames=True`)
- `update_frame_placements(model, data)` — fills `oMf` from existing `oMi`
- `compute_joint_jacobians(model, data)` — fills `data.joint_jacobians` for all joints
- `get_frame_jacobian(model, data, frame_id, reference=...)` — extracts `(B..., 6, nv)` for one frame
- `get_joint_jacobian(model, data, joint_id, reference=...)` — same for joints

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

No `if jtype in ...` branching. All per-kind logic lives in `JointModel.joint_transform()`. The loop over `model.topo_order` is static and unrolls cleanly for `torch.compile`.
