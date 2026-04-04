# core/ — Foundation Layer

This package is the base for everything. Changes here affect all other layers.

## Files

| File | Responsibility |
|------|---------------|
| `_lie_ops.py` | All Lie group operations. **Single source of truth for SE3/se3 math.** |
| `_urdf_parser.py` | Parses `yourdfpy.URDF` into `JointInfo` + `LinkInfo` dataclasses. BFS joint ordering. |
| `_robot.py` | `Robot` class: loads from URDF, runs batched FK, exposes link indices. |

## SE3 Convention

Format: `[tx, ty, tz, qx, qy, qz, qw]` — translation first, scalar qw last (PyPose native).

Never use `.data` on a PyPose LieTensor inside a differentiable path — it detaches from autograd. Always use `.tensor()`.

## FK Implementation

`Robot.forward_kinematics(cfg)` → `(*batch, num_links, 7)`

- Processes joints in BFS topological order (`_fk_joint_order`)
- Maintains `link_world: dict[int, Tensor]` — each entry is `(*batch, 7)`
- For each joint: compose parent world pose with fixed origin, then with the joint's motion transform
- Motion transforms: `_revolute_transform(axis, angle)` and `_prismatic_transform(axis, displacement)`
- Gradient flows through `cfg → T_motion → se3_compose → link_world`
- Optional `base_pose: Tensor | None = None` (7,) SE3: when provided, `se3_apply_base` composes it with all link poses at the end. Existing callers unaffected (default `None`).

## URDF Parser Notes

- `joint.parent` / `joint.child` are strings (link names) in yourdfpy
- `joint.origin` is a `(4, 4)` numpy matrix (or `None` → identity)
- `joint.axis` is a numpy array (or `None` → `[1, 0, 0]`)
- Rotation matrix → quaternion via `pp.mat2SO3(R).tensor()` — do not implement manually

## Axes Convention for Twists

In `_urdf_parser.py`, twists are stored as `[rx, ry, rz, 0, 0, 0]` for revolute and `[0, 0, 0, tx, ty, tz]` for prismatic. In `_robot.py`, axes are extracted:
- Revolute/continuous: `joints.twists[j, :3]`
- Prismatic: `joints.twists[j, 3:]`

## `Robot.get_chain(link_idx) -> list[int]` (`_robot.py`)

Returns joint indices (actuated only) on the path from root to `link_idx`, in root→EE topological order. Uses `_fk_joint_child_link` / `_fk_joint_parent_link` for traversal. Returns `[]` for the root link. Raises `ValueError` if the link is orphaned.

Used by `pose_jacobian` to determine which joints affect a target EE.

## `adjoint_se3(T: Tensor) -> Tensor` (`_lie_ops.py`)

6×6 Adjoint matrix for SE3 pose `T = [tx, ty, tz, qx, qy, qz, qw]`:

```
Ad(T) = [[R,          skew(p) @ R],
          [zeros(3,3), R          ]]
```

Convention: PyPose se3 tangent `[tx, ty, tz, rx, ry, rz]` (translation first).

Used for the floating-base Jacobian smart trick:
```python
T_ee_local = se3_compose(se3_inverse(base), fk_world[link_idx])
J_base = diag([pos_w]*3 + [ori_w]*3) * adjoint_se3(se3_inverse(T_ee_local)) * pose_weight
```
No extra FK call needed — `T_ee_local` is derived from the already-computed world FK.

## `se3_apply_base` (`_lie_ops.py`)

Applies a base SE3 to a full set of link poses via PyPose broadcasting:

```python
se3_apply_base(base_pose, link_poses)
# base_pose: (..., 7), link_poses: (..., num_links, 7) -> (..., num_links, 7)
# pp.SE3(base_pose.unsqueeze(-2)) @ pp.SE3(link_poses)
```

Shape asserts on both inputs (`last dim == 7`) give clear diagnostics on misuse.
All Lie group operations still go through `_lie_ops.py`.
