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

## URDF Parser Notes

- `joint.parent` / `joint.child` are strings (link names) in yourdfpy
- `joint.origin` is a `(4, 4)` numpy matrix (or `None` → identity)
- `joint.axis` is a numpy array (or `None` → `[1, 0, 0]`)
- Rotation matrix → quaternion via `pp.mat2SO3(R).tensor()` — do not implement manually

## Axes Convention for Twists

In `_urdf_parser.py`, twists are stored as `[rx, ry, rz, 0, 0, 0]` for revolute and `[0, 0, 0, tx, ty, tz]` for prismatic. In `_robot.py`, axes are extracted:
- Revolute/continuous: `joints.twists[j, :3]`
- Prismatic: `joints.twists[j, 3:]`
