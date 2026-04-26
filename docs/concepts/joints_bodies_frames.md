# Joints, Bodies, and Frames

The single most consequential decision in a robotics library is what
counts as a "joint." A library that treats free-flyer base motion as
a separate concept from articulated joints ends up with two of every
algorithm — one that knows about a base pose and one that does not.
A library that treats spherical (ball) joints as a separate concept
from revolute joints ends up with two configuration layouts and a
manifold-aware integrator hidden inside half the functions.

We made joints universal. There is one `JointModel` Protocol; thirteen
concrete implementations live under `data_model/joint_models/`; every
algorithm dispatches through that Protocol and never branches on
joint kind. A floating-base robot is a robot whose first joint is
`JointFreeFlyer`. A robot with a ball shoulder is a robot with a
`JointSpherical` somewhere in the tree. The IK solver does not know
the difference; the FK loop does not know the difference. There is
exactly one place in the library where the joint kind matters — the
joint model itself.

Frames and bodies are the lighter-weight counterparts. A `Body` is a
rigid link (1:1 with joints, except joint 0). A `Frame` is a named
coordinate frame anywhere on the robot — the inertial origin of a
link, a tool tip, a camera mount, an IMU pose. Frames are
metadata-only; their actual placements live on `Data`.

## The `JointModel` Protocol

Every concrete joint implements a single Protocol — no inheritance,
no `nn.Module` overhead per joint. The instance is *stateless*; the
per-joint numerical data (placements, axes, limits) lives on `Model`.

```python
class JointModel(Protocol):
    kind: JointKind                  # "revolute_rx", "free_flyer", ...
    nq: int
    nv: int
    axis: torch.Tensor | None        # (3,) for axis-based joints

    def joint_transform(
        self,
        q_slice: torch.Tensor,       # (B..., nq)
    ) -> torch.Tensor:               # (B..., 7) SE3 pose of child in parent frame
        ...

    def joint_motion_subspace(
        self,
        q_slice: torch.Tensor,       # (B..., nq)
    ) -> torch.Tensor:               # (B..., 6, nv) S(q)
        ...

    def joint_velocity(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor:               # (B..., 6) spatial velocity
        ...

    def integrate(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor:               # (B..., nq) retraction q ⊕ v
        ...

    def difference(
        self,
        q0_slice: torch.Tensor,
        q1_slice: torch.Tensor,
    ) -> torch.Tensor:               # (B..., nv) q1 ⊖ q0
        ...

    def random_configuration(
        self,
        generator: torch.Generator | None,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor: ...

    def neutral(self) -> torch.Tensor: ...   # (nq,)

    # Dynamics hooks (default: zeros). Override for non-trivial joints.
    def joint_bias_acceleration(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor: ...                   # (B..., 6) c_J

    def joint_motion_subspace_derivative(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor: ...                   # (B..., 6, nv_j) Ṡ_J
```

Source: `src/better_robot/data_model/joint_models/base.py`.

`joint_transform` is the function the FK loop calls. `joint_motion_subspace`
is what `compute_joint_jacobians` calls. `integrate` and `difference`
are what `Model.integrate` and `Model.difference` dispatch through —
that is how a manifold-aware retraction works without the solver
knowing manifolds exist.

The dynamics hooks (`joint_bias_acceleration`,
`joint_motion_subspace_derivative`) are required for RNEA. For
revolute, prismatic, and free-flyer joints they are identically zero;
for spherical, anatomical, or coupled joints they are non-zero. The
defaults return zero so simple kinds inherit them automatically.

## The joint taxonomy

| Class | Kind | nq | nv | Notes |
|-------|------|----|----|-------|
| `JointUniverse` | `universe` | 0 | 0 | Root placeholder. Joint 0. |
| `JointFixed` | `fixed` | 0 | 0 | Rigid link to parent. |
| `JointRX` / `JointRY` / `JointRZ` | `revolute_r{x,y,z}` | 1 | 1 | Axis-aligned revolute. |
| `JointRevoluteUnaligned` | `revolute_unaligned` | 1 | 1 | Arbitrary 3-vector axis. |
| `JointRevoluteUnbounded` | `revolute_unbounded` | 2 | 1 | Continuous; (cos, sin) parameterisation. |
| `JointPX` / `JointPY` / `JointPZ` | `prismatic_p{x,y,z}` | 1 | 1 | Axis-aligned prismatic. |
| `JointPrismaticUnaligned` | `prismatic_unaligned` | 1 | 1 | Arbitrary 3-vector axis. |
| `JointSpherical` | `spherical` | 4 | 3 | SO(3) ball joint, `(qx, qy, qz, qw)`. |
| `JointFreeFlyer` | `free_flyer` | 7 | 6 | SE(3) floating base. |
| `JointTranslation` | `translation` | 3 | 3 | 3-DOF translation. |
| `JointPlanar` | `planar` | 4 | 3 | SE(2) — `(x, y, cosθ, sinθ)`. |
| `JointHelical` | `helical` | 1 | 1 | Pitch-coupled rotation + translation. |
| `JointComposite` | `composite` | Σ | Σ | Stack of sub-joints. |
| `JointMimic` | `mimic` | 0 | 0 | Zero-DOF, value derived from another joint. |

Source: `src/better_robot/data_model/joint_models/`.

Each kind is a single file. Adding a new kind — a `JointCoupled` for
a tendon-driven finger, say — is a single-file change plus a
registration entry. See {doc}`/conventions/extension` §2.

### Why `nq != nv` for some kinds

- A free-flyer has `nq=7` (translation + scalar-last quaternion) and
  `nv=6` (linear + angular twist). The configuration lives on
  SE(3) ≈ ℝ³ × SO(3); its tangent is ℝ⁶.
- A spherical joint has `nq=4` (quaternion) and `nv=3` (angular
  twist). Configuration on SO(3); tangent ℝ³.
- A continuous (unbounded) revolute has `nq=2` (cosine + sine) and
  `nv=1` (angular rate). The two-parameter representation avoids the
  `±π` wraparound.

The per-joint `nq` / `nv` makes `Model.nq != Model.nv` whenever any
joint in the tree is on a manifold, and this is what `idx_qs` and
`idx_vs` exist to track.

## The free-flyer convention — single code path

A floating-base robot is loaded with `free_flyer=True`:

```python
import better_robot as br

panda = br.load("panda.urdf")                     # fixed base; nq=7, nv=7
g1    = br.load("g1.urdf", free_flyer=True)       # floating base; nq=43, nv=42
```

For G1, `g1.joint_models[1]` is `JointFreeFlyer`. The first 7 entries
of `q` are `[tx, ty, tz, qx, qy, qz, qw]` (the base pose); the first
6 entries of `v` are `[vx, vy, vz, wx, wy, wz]` (the base twist).
The remaining entries are the articulated joints, in the same order
as if the base were fixed.

The IK solver, the FK loop, and the trajopt solver all see this as
"a normal robot whose first joint happens to be a free-flyer." There
is no `setFloatingBase()` call, no `base_pose` argument, no
`_solve_floating_*` family of solvers. The same call solves IK on
both:

```python
panda_result = br.solve_ik(panda, {"panda_hand": target})
g1_result    = br.solve_ik(g1,    {"left_hand": lh_target,
                                    "right_hand": rh_target})
```

The test that proves this works is
`tests/tasks/test_g1_floating_ik.py`: it loads G1 with `free_flyer=True`,
calls `solve_ik` for four whole-body targets, and verifies that the
result frame poses match within tolerance — using the same code path
as the Panda test.

## Mimic joints — the gather trick

URDF supports `<mimic>` joints whose configuration is a linear
function of another joint's. A naive implementation runs an
`if joint.kind == "mimic"` check inside the FK loop; that branches
the hot path and breaks `torch.compile`.

PyRoki's solution, which we adopted: store three tensors on `Model`,
one entry per joint:

```python
mimic_multiplier: torch.Tensor   # (njoints,)  1.0 for non-mimic joints
mimic_offset:     torch.Tensor   # (njoints,)  0.0 for non-mimic joints
mimic_source:     tuple[int, ...]  # source-joint index, or self-index
```

The full-q expansion is then a vectorised gather + multiply with no
Python branching:

```python
q_full[i] = mult[i] * q_active[src[i]] + off[i]
```

For non-mimic joints, `mult[i] = 1.0`, `off[i] = 0.0`, `src[i] = i`,
so the formula is the identity. For mimic joints, the values come from
the URDF `<mimic>` tag. The hot path is one tensor expression
regardless of whether the robot has mimic joints.

## `Frame` — the indirection layer

```python
@dataclass(frozen=True)
class Frame:
    name: str
    parent_joint: int
    joint_placement: torch.Tensor  # (7,) SE3 in parent joint's frame
    frame_type: FrameType          # "body" | "joint" | "fixed" | "op" | "sensor"
```

Source: `src/better_robot/data_model/frame.py`.

A frame is a named coordinate point rigidly attached to a parent
joint. It stores a fixed placement (an SE(3) in the parent joint's
frame), and that is all. The frame's *placement* in the world frame
is computed on demand by `update_frame_placements` and lives on
`data.frame_pose_world`.

Why this indirection is worth its weight:

- IK targets address frames by name (`{"panda_hand": target}`),
  not joint indices. The user never has to compute "joint 7's
  position offset by the tool tip."
- Visualisation code can query `data.frame_pose("camera_optical")`
  without inflating the joint count.
- Sensor mounting points (IMU, force-torque, motion-capture markers)
  are frames, not bodies. Adding a frame is a metadata-only change.

```python
frame_id    = model.frame_id("panda_hand_tcp")
target_pose = data.frame_pose_world[..., frame_id, :]
# Or, typed:
T_tool = data.frame_pose("panda_hand_tcp")           # SE3
```

Every body gets a default frame named `"body_<bodyname>"`, located at
the body's inertial origin. User-added frames live alongside in
`model.frames`.

## `Body` — the rigid link

```python
@dataclass(frozen=True)
class Body:
    name: str
    parent_joint: int
    visual_geom: int | None = None   # index into collision/visual geometry
```

Source: `src/better_robot/data_model/body.py`.

Bodies carry inertial properties via the `body_inertias` packed
tensor on `Model` (a `(nbodies, 10)` array of
`[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`). The struct
itself is metadata.

Bodies are 1:1 with joints in the URDF / MJCF sense: each non-universe
joint has exactly one body attached via `model.joint_placements[j]`.
Joint 0's "body" is the universe (zero inertia, no visual geometry).

## Sharp edges

- The first real joint in a fixed-base robot is `joint_models[1]`,
  not `joint_models[0]`. Joint 0 is always the universe.
- `Model.nq` is `sum(nqs)` and `Model.nv` is `sum(nvs)` — they are
  not required to be equal.
- For a free-flyer model, the first 7 entries of `q` are the base
  pose **and the quaternion is scalar-last**: `(tx, ty, tz, qx, qy,
  qz, qw)`. Code that expects scalar-first (`w, x, y, z`) is wrong
  for this library.
- A spherical joint integrates differently from three concatenated
  revolute joints. It cannot be substituted by an XYZ Euler set —
  the Euler representation has gimbal-lock singularities that an SO(3)
  parameterisation does not.
- `JointMimic` has `nq=0` / `nv=0`. The "shadow" configuration of a
  mimic joint is computed from the source via the gather trick and
  does not occupy any slot in `q`.

## Where to look next

- {doc}`kinematics` — how `forward_kinematics` calls
  `joint_transform` for every joint.
- {doc}`lie_and_spatial` — the SE(3) and SO(3) ops that
  `JointFreeFlyer.integrate` and `JointSpherical.integrate` use.
- {doc}`/conventions/extension` §2 — recipe for adding a new joint
  kind.
