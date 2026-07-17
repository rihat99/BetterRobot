# Model and Data

A robotics library has to answer two completely different questions
at completely different rates. *What is this robot?* — its kinematic
tree, its inertias, its joint limits, its frame metadata — is asked
once when the URDF loads and never again. *What does this robot
look like at this instant?* — every joint pose, every Jacobian, the
mass matrix, the centroidal momentum — is asked thousands of times
per second inside an optimiser, and most of the answers depend on a
single configuration tensor `q`.

Pinocchio's solution to that asymmetry was a strict split between a
frozen `Model` (the kinematic tree) and a per-query `Data` (the
workspace that everything else writes into). It is the one piece of
Pinocchio's architecture every modern robotics library has stolen
verbatim, because the alternative — a single mutable `Robot` object
that holds both — falls apart under three pressures: GPU sharing
(you cannot share a mutable object across CUDA streams), batching
(one workspace per query is the natural scaling unit), and autograd
(autograd graphs leak when state mutates between the forward and
backward pass).

We took the same split. `Model` is `@dataclass(frozen=True)`,
device-polymorphic via `.to()`, and shared freely across threads,
streams, and processes. `Data` is mutable, per-query, carries the
leading batch axis, and contains seventeen optional tensors that are
filled lazily as kinematics and dynamics functions need them. The
chapter that follows describes both in detail, including the cache
invariant that prevents stale Jacobians from sneaking through after
`q` changes.

## `Model` — frozen topology

```python
@dataclass(frozen=True)
class Model:
    # ──────────── counts ────────────
    njoints: int
    nbodies: int
    nframes: int
    nq: int                       # total configuration dim
    nv: int                       # total tangent dim

    # ──────────── names & indexing ────────────
    name: str
    joint_names: tuple[str, ...]
    body_names:  tuple[str, ...]
    frame_names: tuple[str, ...]
    joint_name_to_id: dict[str, int]
    body_name_to_id:  dict[str, int]
    frame_name_to_id: dict[str, int]

    # ──────────── topology ────────────
    parents:    tuple[int, ...]   # parent joint per joint, -1 for joint 0
    children:   tuple[tuple[int, ...], ...]
    subtrees:   tuple[tuple[int, ...], ...]
    supports:   tuple[tuple[int, ...], ...]
    topo_order: tuple[int, ...]

    # ──────────── joint dispatch ────────────
    joint_models: tuple[JointModel, ...]
    nqs:    tuple[int, ...]
    nvs:    tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]

    # ──────────── device tensors ────────────
    joint_placements: torch.Tensor   # (njoints, 7) parent→joint origin
    body_inertias:    torch.Tensor   # (nbodies, 10) packed inertia
    lower_pos_limit:  torch.Tensor   # (nq,)
    upper_pos_limit:  torch.Tensor   # (nq,)
    velocity_limit:   torch.Tensor
    effort_limit:     torch.Tensor
    rotor_inertia:    torch.Tensor
    armature:         torch.Tensor
    friction:         torch.Tensor
    damping:          torch.Tensor
    gravity:          torch.Tensor   # (6,) world spatial acceleration

    # ──────────── frames + neutral configurations ────────────
    frames: tuple[Frame, ...]
    reference_configurations: dict[str, torch.Tensor]
    q_neutral: torch.Tensor

    # ──────────── methods ────────────
    def to(self, device=None, dtype=None) -> "Model": ...
    def create_data(self, *, batch_shape=(), device=None, dtype=None) -> "Data": ...
    def joint_id(self, name: str) -> int: ...
    def q_permutation(self, other_joint_order: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]: ...
    def frame_id(self, name: str) -> int: ...
    def body_id (self, name: str) -> int: ...
    def get_subtree(self, joint_id: int) -> tuple[int, ...]: ...
    def get_support(self, joint_id: int) -> tuple[int, ...]: ...
    def integrate (self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...
    def difference(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor: ...
    def random_configuration(self, generator=None) -> torch.Tensor: ...
```

Source: `src/better_robot/data_model/model.py`.

A few choices are worth highlighting because they are the
load-bearing ones.

**Joint 0 is the universe.** The root "joint" is a zero-DOF
placeholder (`parents[0] == -1`). The first real joint is joint 1.
For a fixed-base Panda joint 1 is a `JointFixed`; for G1 joint 1 is
`JointFreeFlyer`. There is no separate concept of "base"; everything
is just a joint with an integer index, and the universe is the joint
whose index is 0.

**Topology is Python tuples; device tensors are PyTorch.** The
`parents`, `children`, `subtrees`, and `topo_order` fields are tuples
of integers, frozen at build time. They unroll cleanly under
`torch.compile`, which is why FK can be a single
`for j in model.topo_order` loop. The numerical buffers
(`joint_placements`, `body_inertias`, the limits) are device tensors
moved by `.to()`.

**Joint-order remapping is a gather.** When an external tensor concatenates
the same public per-joint slices in another name order, use
`Model.q_permutation` once and gather on the trailing dimension:

```python
perm_q, perm_v = model.q_permutation(external_joint_names)
q_model = q_external[..., perm_q]  # works for (nq,), (B, nq), (B, T, nq), ...
v_model = v_external[..., perm_v]
```

The method expands whole joint slices, so free-flyer `7/6`, spherical `4/3`,
and scalar joints are all handled without a per-frame Python loop. Names for
zero-DOF joints may be omitted; unknown, duplicate, or missing active names
raise `ValueError`. The external vector must use the model's public
`nqs`/`nvs` dimensions.

**`@dataclass(frozen=True)`.** No custom immutability guard, no
`_frozen` flag. Python's dataclass freeze is enough; the discipline
is enforced by the language. Every algorithm above this layer can
share a `Model` across workers without copying.

**`body_inertias` is packed.** Each body's inertial properties are
stored as a `(nbodies, 10)` tensor with layout
`[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`. Unpacking into a
typed `Inertia` happens on demand. Storing 10 floats per body in a
contiguous tensor is what lets `crba`'s composite-rigid-body
recursion gather inertias without per-body Python work.

**Mimic joints use an explicit reduced↔full map.** Public `nq`, `nv`,
`nqs`/`nvs`, and `idx_qs`/`idx_vs` omit mimic targets. Whole-body recursions
retain `nq_full`, `nv_full` and matching full slice tables. Static
`q_expansion`, `q_offset`, and `v_expansion` tensors define
`q_full = q @ q_expansion.T + q_offset` and
`v_full = v @ v_expansion.T`; the same tangent map projects Jacobians,
forces, mass matrices, and centroidal maps. Non-mimic models take an identity
fast path.

**`reference_configurations` and `q_neutral`.** A dict of named
configurations (e.g. `"half_sitting"`) plus a single canonical
neutral configuration. Solvers default to `q_neutral` as the starting
point.

### Universal `integrate` and `difference`

`ModelStructure` precomputes public q/v gather tables for five semantic
families: Euclidean joints, spherical joints, free-flyers, continuous
revolute joints, and planar joints. `Model.integrate` and
`Model.difference` gather every joint in one family, evaluate one batched
manifold operation, then scatter into the public trailing dimension. A
24-joint spherical body therefore performs one SO(3) call, not 23 Python
calls. Composite and custom `JointModel` objects retain their exact per-joint
dispatch as a compatibility fallback.

The formulas keep right/local perturbation semantics: spherical retraction is
`normalize(q · exp(v))` and free-flyer retraction is `T · exp(ξ)`. Solvers
still call one universal API and do not need to know which coordinates are
Euclidean or manifold-valued.

Leading dimensions use torch's right-aligned broadcasting. For example,
`q.shape == (B, 1, nq)` and `v.shape == (T, nv)` produce
`(B, T, nq)`. Trailing dimensions and devices are validated explicitly;
mixed floating dtypes follow torch promotion. Reduced mimic targets own zero
public width, so only their independent source enters a manifold group.

The grouped implementation is tested against the former per-joint loop with
tight dtype-aware tolerances. Contiguous grouped reductions can differ from
strided calls by a few ULPs; see
`plan/design_notes/m3_integrate_difference_vectorization.md` for parity and
CPU benchmark evidence.

## `Data` — per-query workspace

```python
@dataclass
class Data:
    # ──────────── configuration & derivatives ────────────
    q:    torch.Tensor                              # (B..., nq)
    v:    torch.Tensor | None = None                # (B..., nv)
    a:    torch.Tensor | None = None                # (B..., nv)
    tau:  torch.Tensor | None = None                # (B..., nv)

    # ──────────── kinematics ────────────
    joint_pose_local:         torch.Tensor | None = None   # (B..., njoints, 7)
    joint_pose_world:         torch.Tensor | None = None   # (B..., njoints, 7)
    frame_pose_world:         torch.Tensor | None = None   # (B..., nframes, 7)
    joint_velocity_world:     torch.Tensor | None = None   # (B..., njoints, 6)
    joint_acceleration_world: torch.Tensor | None = None   # (B..., njoints, 6)
    joint_velocity_local:     torch.Tensor | None = None
    joint_acceleration_local: torch.Tensor | None = None

    # ──────────── jacobians ────────────
    joint_jacobians:     torch.Tensor | None = None   # (B..., njoints, 6, nv)
    joint_jacobians_dot: torch.Tensor | None = None

    # ──────────── dynamics ────────────
    mass_matrix:     torch.Tensor | None = None   # (B..., nv, nv)
    coriolis_matrix: torch.Tensor | None = None
    gravity_torque:  torch.Tensor | None = None
    bias_forces:     torch.Tensor | None = None   # C(q,v)·v + g(q)
    ddq:             torch.Tensor | None = None

    # ──────────── centroidal ────────────
    centroidal_momentum_matrix: torch.Tensor | None = None
    centroidal_momentum:        torch.Tensor | None = None
    com_position:               torch.Tensor | None = None
    com_velocity:               torch.Tensor | None = None
    com_acceleration:           torch.Tensor | None = None

    # ──────────── cache bookkeeping ────────────
    _kinematics_level: KinematicsLevel = KinematicsLevel.NONE

    def reset(self) -> None: ...
    def clone(self) -> "Data": ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    def require(self, level: KinematicsLevel) -> None: ...
    def invalidate(self, level: KinematicsLevel = KinematicsLevel.NONE) -> None: ...
    def joint_pose(self, joint_id: int) -> "SE3": ...
    def frame_pose(self, name_or_id: str | int) -> "SE3": ...
```

Source: `src/better_robot/data_model/data.py`.

``Data`` is not tied to a Python ``id(model)``. Rebinding differentiable
values keeps the same ``ModelStructure``, and each public pass writes the
fully broadcast execution view into ``data.q`` before populating caches.

The design choices are:

**Optional tensors, not zero-filled buffers.** Unused cache slots
stay `None`; kinematics and dynamics functions populate them lazily.
This sidesteps the `mjwarp` 280-field god-dataclass trap where every
query allocates every field whether the caller wants it or not.

**Batch axis is load-bearing.** Every optional tensor carries `B...`
and the implementation is vectorised, not Python-looped. There is no
scalar fast path that diverges from the batched one.

**Readable identifiers.** Every field is named by what it is —
`joint_pose_world`, `frame_pose_world`, `bias_forces`,
`centroidal_momentum_matrix`. The Pinocchio cryptic shorthands
(`oMi`, `oMf`, `liMi`, `nle`, `Ag`) are not part of the public
surface; the {doc}`/conventions/naming` chapter has the full table
and rationale.

**`_kinematics_level` mirrors Pinocchio's invariant.** A
`KinematicsLevel` enum tracks how far the cache has been populated:
`NONE` (just `q`), `PLACEMENTS` (FK done), `VELOCITIES` (FK + first
derivatives), `ACCELERATIONS` (FK + first and second derivatives).
Every kinematics / dynamics entry point calls `data.require(level)`
and raises `StaleCacheError` if the cache is below.

### The cache invariant — what is enforced and what is not

`Data.__setattr__` runs on **assignments**:

```python
data = forward_kinematics(model, q)          # _kinematics_level = PLACEMENTS
data.q = new_q                               # __setattr__ fires; downstream caches invalidated
J = compute_joint_jacobians(model, data)     # raises StaleCacheError — must call FK first
```

The `__setattr__` hook handles `q`, `v`, `a`, and any other declared
input field by dropping caches above the appropriate level and
resetting `_kinematics_level`. Functions advance the level
explicitly; nothing else writes to it.

**Limitation: in-place tensor mutation is not detected.**

```python
data.q[..., 0] += 1.0    # silently stale; __setattr__ NOT called
data.q.add_(other)       # silently stale
data.q.copy_(new_q)      # silently stale
```

This is a Python limitation — tensor views and in-place ops bypass
dataclass attribute machinery. The contract this layer pins is:

> BetterRobot detects **reassignment** of `q`, `v`, `a`, and any
> public input field on `Data`, and invalidates downstream caches on
> reassignment. In-place tensor mutation of those fields is **not**
> detected; it is a documented misuse pattern.

User-facing guidance, mirrored in `Data`'s docstring:

- **Reassign**, do not mutate. `data.q = new_q`. Never
  `data.q[...] = new_q[...]`.
- If a hot path absolutely needs in-place updates, follow them with
  `data.invalidate(KinematicsLevel.NONE)` — the public method that
  resets the level explicitly.
- IK and trajopt solvers reassign; user code patterned after them
  stays in the safe zone.

The limitation is codified by
`tests/contract/test_cache_invariants.py`, which includes a
"documented limitation" case asserting that
`data.q[..., 0] += 1.0` is *not* detected. Future hardening could
store `q` behind a property returning a read-only view, but that
would add overhead and break autograd in subtle ways.

## Configuration layout — `q` and `v`

Everything lives in a flat leading-batch tensor:

```
q  shape  (B..., nq)
v  shape  (B..., nv)
```

with per-joint slices determined by `model.idx_qs[j]` and
`model.nqs[j]` (ditto `idx_vs`, `nvs`). The `forward_kinematics`
implementation iterates the topo order and slices:

```python
qj = q[..., idx_q : idx_q + nq_j]
vj = v[..., idx_v : idx_v + nv_j]
Tj = joint_models[j].joint_transform(qj)
```

No string dispatch, no type-specific branches outside the joint
modules themselves. A free-flyer base means the first 7 entries of
`q` are the base pose `[tx, ty, tz, qx, qy, qz, qw]` and the first 6
entries of `v` are the base twist `[vx, vy, vz, wx, wy, wz]`.
Everything else slots in after that.

## Sharp edges

- `Model.to(device, dtype)` returns a *new* Model — `Model` is
  frozen. Sharing one `Model` across devices (CPU for diagnostics,
  CUDA for solving) is a legitimate pattern.
- `Data` has no `.to()`. Device and dtype follow the input `q`.
- `nq != nv` for any model with a free-flyer or spherical joint
  (free-flyer: `nq=7`, `nv=6`; spherical: `nq=4`, `nv=3`). Per-joint
  dimensions are accessed via `model.nqs[j]` and `model.nvs[j]`.
- Quaternions in `q` are scalar-last `[..., qx, qy, qz, qw]`. The
  first 7 entries of `q` are not a 7-DOF rotation; they are
  translation + scalar-last quaternion.

## Where to look next

- {doc}`joints_bodies_frames` — the joint taxonomy that powers
  `Model.joint_models`, plus `Frame` and `Body`.
- {doc}`kinematics` — how `forward_kinematics` walks `topo_order`
  and fills `Data`.
- {doc}`/conventions/naming` — the rename table from Pinocchio's
  cryptic names to the readable `Data` fields above.
