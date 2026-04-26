# Glossary

Quick lookup table for the vocabulary used throughout BetterRobot. The
authoritative source for the renames is
{doc}`/conventions/naming`.

## Storage conventions

| Object | Layout | Notes |
|--------|--------|-------|
| SE(3) pose | `(..., 7)` `[tx, ty, tz, qx, qy, qz, qw]` | Scalar quaternion **last**. |
| se(3) tangent | `(..., 6)` `[vx, vy, vz, wx, wy, wz]` | Linear block first. |
| Quaternion | `(..., 4)` `[qx, qy, qz, qw]` | Scalar last. |
| Spatial Jacobian | `(..., 6, nv)` rows `[v_lin (3), ω (3)]` | Linear block first. |

## Pinocchio → BetterRobot rename table

We keep Pinocchio's *algorithm* names (universal in the literature) but
replace its cryptic *storage* names with readable identifiers.

| Pinocchio | BetterRobot | Meaning |
|-----------|-------------|---------|
| `oMi` | `joint_pose_world` | World-frame pose of joint *i*. |
| `oMf` | `frame_pose_world` | World-frame pose of frame *f*. |
| `liMi` | `joint_pose_local` | Parent-frame pose of joint *i*. |
| `nle` | `bias_forces` | Non-linear effects (gravity + Coriolis). |
| `Ycrb` | `composite_inertia` | Composite-rigid-body inertia. |
| `Jcrb` / `M` | `mass_matrix` | Joint-space inertia. |
| `data.tau` | `data.joint_torques` | Generalised forces. |

Algorithm names retained verbatim: `forward_kinematics`,
`compute_joint_jacobians`, `rnea`, `aba`, `crba`, `ccrba`, `SE3`,
`Motion`, `Force`, `Inertia`.

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| **DAG** | Directed acyclic graph (the layer dependency rule). |
| **FK** | Forward kinematics. |
| **IR** | Intermediate representation (`IRModel` produced by parsers). |
| **LM** | Levenberg–Marquardt (default IK solver). |
| **LWA** | LOCAL_WORLD_ALIGNED Jacobian (the default `get_frame_jacobian` mode). |
| **RNEA / ABA / CRBA** | Recursive Newton–Euler / Articulated-body / Composite-rigid-body algorithms. |
| **SE(3) / SO(3)** | Special Euclidean / Orthogonal groups (rigid-body / rotation manifolds). |

## See also

- {doc}`/conventions/naming` — the full normative naming spec.
- {doc}`/concepts/lie_and_spatial` — why scalar-last and why linear-first.
