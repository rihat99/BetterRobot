# kinematics/ — Forward Kinematics and Jacobians

## Entry Points

- `forward_kinematics(model, q_or_data, compute_frames=False)` — single topological pass, fills `oMi` (and `oMf` if `compute_frames=True`)
- `update_frame_placements(model, data)` — fills `oMf` from existing `oMi`
- `compute_joint_jacobians(model, data)` — fills `data.J` for all joints
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
    AUTODIFF = "autodiff"     # torch.func.jacrev
    FUNCTIONAL = "functional" # torch.func.jacfwd
    AUTO = "auto"             # prefer analytic, fall back to autodiff
```

`residual_jacobian`'s AUTO fallback uses central finite differences. The pure-PyTorch Lie backend has clean autograd, so `torch.autograd.functional.jacobian` is also valid; FD is kept because it's joint-kind-agnostic and matches analytic Jacobians to numerical noise.

## FK Hot Path

No `if jtype in ...` branching. All per-kind logic lives in `JointModel.joint_transform()`. The loop over `model.topo_order` is static and unrolls cleanly for `torch.compile`.
