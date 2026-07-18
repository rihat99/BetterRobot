# Naming

Public names should tell a reader what a value means without requiring a
translation table. BetterRobot keeps the short symbols that robotics papers
use everywhere, such as `q`, `v`, `rnea`, and `SE3`. Storage fields and
ordinary functions use descriptive English.

The contract test in `tests/contract/test_naming.py` checks removed
Pinocchio-style `Data` attributes. This page defines the broader convention
that reviewers apply to new code.

## Storage fields

Use `<entity>_<quantity>_<frame>` when all three parts matter:

| Part | Examples |
|---|---|
| entity | `joint`, `frame`, `body`, `com` |
| quantity | `pose`, `velocity`, `acceleration`, `jacobian`, `momentum` |
| frame | `world`, `local`, `body` |

`joint_velocity_world` therefore means the velocity of each joint expressed
in world axes. Omit a part only when it would repeat information already fixed
by the type or API.

## Functions

| Kind | Rule | Examples |
|---|---|---|
| Published algorithm | keep the conventional lowercase acronym | `rnea`, `aba`, `crba`, `ccrba` |
| Compute | form or fill a result | `compute_joint_jacobians`, `compute_centroidal_map` |
| Get | read a cheap derived value | `get_frame_jacobian` |
| Update | fill a specific cache after an earlier pass | `update_frame_placements` |
| User task | use a plain action phrase | `forward_kinematics`, `solve_ik`, `solve_trajopt` |

Boolean names start with `is_`, `has_`, or `should_`. Avoid abbreviations
that are not standard in the field.

## Mathematical names that stay short

| Name | Meaning |
|---|---|
| `q` | generalized configuration |
| `v` | generalized velocity |
| `a` | generalized acceleration |
| `tau` | generalized force or torque |
| `nq`, `nv` | configuration and tangent dimensions |
| `SE3`, `SO3` | rigid transforms and rotations |
| `exp`, `log`, `hat`, `vee` | standard Lie-group operations |
| `Jr`, `Jl` | right and left Jacobians in equations and tight internal math |

Public functions prefer descriptive forms such as
`right_jacobian_se3`. Short equation symbols remain useful as local
variables.

## Public storage names

### Kinematics

| Pinocchio name | BetterRobot name | Shape |
|---|---|---|
| `liMi` | `joint_pose_local` | `(B..., njoints, 7)` |
| `oMi` | `joint_pose_world` | `(B..., njoints, 7)` |
| `oMf` | `frame_pose_world` | `(B..., nframes, 7)` |
| `v_joint` | `joint_velocity_local` | `(B..., njoints, 6)` |
| `ov` | `joint_velocity_world` | `(B..., njoints, 6)` |
| `a_joint` | `joint_acceleration_local` | `(B..., njoints, 6)` |
| `oa` | `joint_acceleration_world` | `(B..., njoints, 6)` |
| `J` | `joint_jacobians` | `(B..., njoints, 6, nv)` |
| `dJ` | `joint_jacobians_dot` | `(B..., njoints, 6, nv)` |

### Dynamics and centroidal quantities

| Pinocchio name | BetterRobot name | Shape |
|---|---|---|
| `M` | `mass_matrix` | `(B..., nv, nv)` |
| `C` | `coriolis_matrix` | `(B..., nv, nv)` |
| `g` | `gravity_torque` | `(B..., nv)` |
| `nle` | `bias_forces` | `(B..., nv)` |
| `Ag` | `centroidal_momentum_matrix` | `(B..., 6, nv)` |
| `hg` | `centroidal_momentum` | `(B..., 6)` |
| `com` | `com_position` | `(B..., 3)` |
| `vcom` | `com_velocity` | `(B..., 3)` |
| `acom` | `com_acceleration` | `(B..., 3)` |

`ddq` remains available because it is standard notation for generalized
acceleration.

## Model dimensions and indices

`nq`, `nv`, `njoints`, `nbodies`, and `nframes` are familiar
robotics dimensions. `idx_qs` and `idx_vs` are the per-joint start indices
for slices of `q` and `v`. Other model fields, such as
`joint_placements`, `body_inertias`, `lower_pos_limit`, and
`topo_order`, are written out.

## Shapes in annotations and docstrings

`better_robot._typing` contains readable aliases such as `ConfigTensor`,
`VelocityTensor`, `SE3Tensor`, `FramePoseStack`, and
`JointJacobian`. They help type checkers and readers; runtime shape checks
still happen at public boundaries.

Every public tensor parameter also states its event shape in the docstring:

```text
def get_frame_jacobian(
    model: Model,
    data: Data,
    frame_id: int,
) -> torch.Tensor:
    """Return a Jacobian with shape ``(B..., 6, nv)``."""
```

Use `B...` for any leading execution batch. Name units and coordinate frames
next to the shape.

## Closed choices

Use a `Literal` or enum when a public string has a fixed set of meanings.
For example, Jacobian references are `"world"`, `"local"`, and
`"local_world_aligned"`. `KinematicsLevel` records which kinematic caches
are ready.

Do not add a free-form string when a typed choice already exists.

## Review checklist

When adding a public name:

1. Prefer a descriptive noun or verb phrase.
2. Keep a short symbol only when it is common in robotics literature.
3. State shape, units, and frame for tensor values.
4. Add a new user-facing term to {doc}`/reference/glossary`.
5. Update `tests/contract/test_naming.py` if a removed storage attribute must
   stay absent.
