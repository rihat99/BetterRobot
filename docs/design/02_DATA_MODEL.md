# 02 · Data model — `Model`, `Data`, Joints, Bodies, Frames

This document specifies the pinocchio-style **universal** data model
that the skeleton landed under `src/better_robot/data_model/`. The
static vs floating-base split is gone: a floating base is simply a
`JointFreeFlyer` root joint. A ball joint is simply a `JointSpherical`.

> **Naming.** Field names below use the readable convention from
> [13_NAMING.md](../conventions/13_NAMING.md). Pinocchio equivalents (`oMi`, `oMf`,
> `liMi`, `nle`, `Ag`, `hg`, …) — used in the original prototype before
> the skeleton landed — are not part of the public surface. The rename
> is enforced by `tests/contract/test_naming.py`.

## 1. The three objects

```
Model    — frozen topology + constants (immutable after build)
Data     — per-query mutable workspace, carries the batch axis
Frame    — name-addressable point rigidly attached to a joint
```

`Model` is built once, shared across workers/devices. `Data` is created per
query (or per batch). `Frame` is metadata that lives on `Model`; individual
frame *placements* live on `Data`.

## 2. `Model` — frozen topology

```python
# src/better_robot/data_model/model.py

import torch
from dataclasses import dataclass, field
from typing import Sequence

from ..spatial.inertia import Inertia
from .joint_models.base import JointModel
from .frame import Frame

@dataclass(frozen=True)
class Model:
    """Immutable kinematic-tree description. Pinocchio-style.

    A `Model` is the topology + constants of a mechanism. It is frozen after
    construction. All tensor buffers are device/dtype polymorphic via `.to()`.

    Joint 0 is the *universe* (fixed frame); `parents[0] == -1`. Every other
    joint has exactly one parent joint. Bodies are 1:1 with joints: `body[i]`
    is the body attached to joint `i` via `joint_placements[i]`.
    """

    # ──────────── counts ────────────
    njoints:  int                       # includes joint 0 (universe)
    nbodies:  int                       # == njoints
    nframes:  int
    nq:       int                       # total configuration dim (sum of nqs)
    nv:       int                       # total velocity dim     (sum of nvs)

    # ──────────── names & indexing ────────────
    name:        str
    joint_names: tuple[str, ...]        # length njoints
    body_names:  tuple[str, ...]        # length nbodies
    frame_names: tuple[str, ...]        # length nframes

    joint_name_to_id: dict[str, int]    # built in __post_init__
    body_name_to_id:  dict[str, int]
    frame_name_to_id: dict[str, int]

    # ──────────── topology ────────────
    parents:    tuple[int, ...]         # parent joint per joint, -1 for joint 0
    children:   tuple[tuple[int, ...], ...]
    subtrees:   tuple[tuple[int, ...], ...]
    supports:   tuple[tuple[int, ...], ...]
    topo_order: tuple[int, ...]         # depth-first traversal order (parents first);
                                         # equals tuple(range(njoints)) for URDF-loaded models,
                                         # since joint indices are assigned in DFS order

    # ──────────── joint models (dispatch tables, not python objects per joint) ────────────
    joint_models: tuple[JointModel, ...]    # length njoints, joint_models[0] == JointUniverse
    nqs:          tuple[int, ...]           # per-joint nq
    nvs:          tuple[int, ...]           # per-joint nv
    idx_qs:       tuple[int, ...]           # starting index into q for each joint
    idx_vs:       tuple[int, ...]           # starting index into v for each joint

    # ──────────── device tensors ────────────
    # Frozen buffers, but live on cpu/cuda/fp32/fp64 polymorphically.
    joint_placements: torch.Tensor          # (njoints, 7) fixed parent->joint transform (SE3)
    body_inertias:    torch.Tensor          # (nbodies, 10) mass + com + symmetric3
    lower_pos_limit:  torch.Tensor          # (nq,)
    upper_pos_limit:  torch.Tensor          # (nq,)
    velocity_limit:   torch.Tensor          # (nv,)
    effort_limit:     torch.Tensor          # (nv,)
    rotor_inertia:    torch.Tensor          # (nv,)
    armature:         torch.Tensor          # (nv,)
    friction:         torch.Tensor          # (nv,)
    damping:          torch.Tensor          # (nv,)
    gravity:          torch.Tensor          # (6,) world spatial acceleration of gravity

    # ──────────── frames ────────────
    frames: tuple[Frame, ...]               # metadata only; placements live on Data

    # ──────────── reference configurations ────────────
    reference_configurations: dict[str, torch.Tensor]  # named q's (e.g. "half_sitting")
    q_neutral: torch.Tensor                 # neutral config; midpoint for revolute/prismatic, identity for SE3/SO3

    # ──────────── methods ────────────
    def to(self, device=None, dtype=None) -> "Model": ...
    def create_data(self, *, batch_shape=(), device=None, dtype=None) -> "Data": ...
    def joint_id(self, name: str) -> int: ...
    def frame_id(self, name: str) -> int: ...
    def body_id(self,  name: str) -> int: ...
    def get_subtree(self, joint_id: int) -> tuple[int, ...]: ...
    def get_support(self, joint_id: int) -> tuple[int, ...]: ...
    def integrate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...
    def difference(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor: ...
    def random_configuration(self, generator=None) -> torch.Tensor: ...
```

### Naming notes

- `nq` vs `nv` — a free-flyer has `nq=7` (quat-xyz) but `nv=6` (twist). Every
  joint provides its own `nq_i`, `nv_i`, `idx_q_i`, `idx_v_i`. Standard
  rigid-body notation; kept as-is.
- `joint 0 == universe` — the root "joint" is a zero-DOF placeholder. The
  first *real* joint is joint 1. For a fixed-base Panda joint 1 is a
  `JointFixed`; for G1 joint 1 is a `JointFreeFlyer`.
- `body_inertias` is a `(nbodies, 10)` packed tensor: `[mass, cx, cy, cz,
  Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`. Unpacks into `Inertia` on demand.

## 3. `Data` — per-query workspace

```python
# src/better_robot/data_model/data.py

import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Data:
    """Mutable per-query workspace. Carries a leading batch (and optional time) axis.

    Shapes use `B...` to denote any number of leading batch dims (including
    (B,) for single-batch or (B, T) for trajectories).
    """

    # reference back to owning model (never mutated — for sanity checks only)
    _model_id: int

    # ──────────── configuration & derivatives ────────────
    q:    torch.Tensor                              # (B..., nq)
    v:    Optional[torch.Tensor] = None             # (B..., nv)
    a:    Optional[torch.Tensor] = None             # (B..., nv)
    tau:  Optional[torch.Tensor] = None             # (B..., nv)

    # ──────────── kinematics ────────────
    joint_pose_local:         Optional[torch.Tensor] = None   # (B..., njoints, 7)  parent-joint frame
    joint_pose_world:         Optional[torch.Tensor] = None   # (B..., njoints, 7)  world frame
    frame_pose_world:         Optional[torch.Tensor] = None   # (B..., nframes, 7)  world frame

    joint_velocity_world:     Optional[torch.Tensor] = None   # (B..., njoints, 6)
    joint_acceleration_world: Optional[torch.Tensor] = None   # (B..., njoints, 6)
    joint_velocity_local:     Optional[torch.Tensor] = None   # (B..., njoints, 6)
    joint_acceleration_local: Optional[torch.Tensor] = None   # (B..., njoints, 6)

    # ──────────── jacobians ────────────
    joint_jacobians:          Optional[torch.Tensor] = None   # (B..., njoints, 6, nv)
    joint_jacobians_dot:      Optional[torch.Tensor] = None   # (B..., njoints, 6, nv)

    # ──────────── dynamics ────────────
    mass_matrix:              Optional[torch.Tensor] = None   # (B..., nv, nv)
    coriolis_matrix:          Optional[torch.Tensor] = None   # (B..., nv, nv)
    gravity_torque:           Optional[torch.Tensor] = None   # (B..., nv)
    bias_forces:              Optional[torch.Tensor] = None   # (B..., nv)   C(q,v)·v + g(q)
    ddq:                      Optional[torch.Tensor] = None   # (B..., nv)   q̈ from aba

    # ──────────── centroidal ────────────
    centroidal_momentum_matrix: Optional[torch.Tensor] = None # (B..., 6, nv)
    centroidal_momentum:      Optional[torch.Tensor] = None   # (B..., 6)
    com_position:             Optional[torch.Tensor] = None   # (B..., 3)
    com_velocity:             Optional[torch.Tensor] = None   # (B..., 3)
    com_acceleration:         Optional[torch.Tensor] = None   # (B..., 3)

    # ──────────── cache bookkeeping ────────────
    _kinematics_level: KinematicsLevel = KinematicsLevel.NONE
    # NONE / PLACEMENTS (1) / VELOCITIES (2) / ACCELERATIONS (3)

    def reset(self) -> None: ...            # zero everything optional
    def clone(self) -> "Data": ...          # deep copy (sharing _model_id)
    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    # ──────────── cache enforcement (see §3.1) ────────────
    def require(self, level: KinematicsLevel) -> None:
        """Raise StaleCacheError if _kinematics_level < level."""
    def invalidate(self, level: KinematicsLevel = KinematicsLevel.NONE) -> None:
        """Drop downstream caches; advance/regress to `level`."""
    def joint_pose(self, joint_id: int) -> "SE3":
        """Typed accessor; raises StaleCacheError if joint_pose_world is None."""
    def frame_pose(self, name_or_id: str | int) -> "SE3":
        """Typed accessor for a named or indexed frame."""

    # `__setattr__` invalidates downstream caches on reassignment of
    # `q`, `v`, `a`. See §3.1 below for the limitation re: in-place mutation.
```

### Design choices

- **Optional tensors**, not zero-filled buffers. Unused cache slots stay
  `None`; the kinematics/dynamics functions populate them lazily. This
  sidesteps mjwarp's 280-field god-dataclass trap.
- **`_kinematics_level`** mirrors Pinocchio's "kinematic level" invariant.
  It is a `KinematicsLevel` enum (
  [13_NAMING.md §Enums](../conventions/13_NAMING.md)) and is **enforced** —
  `compute_joint_jacobians`, `update_frame_placements`, `get_frame_jacobian`,
  and `center_of_mass(... v=None)` all call `data.require(KinematicsLevel.PLACEMENTS)`
  on entry and raise `StaleCacheError` (
  [17_CONTRACTS.md §2](../conventions/17_CONTRACTS.md)) if the cache is below.
- **Batch axis is load-bearing.** Every optional tensor carries `B...` and
  the implementation is *vectorised*, not Python-looped.
- **Readable identifiers.** Every field is named by what it is. The pinocchio
  shorthands (`oMi`, `oMf`, `liMi`, `nle`, `Ag`, `hg`) exist as
  `@property` shims during the migration window (§11); see
  [13_NAMING.md](../conventions/13_NAMING.md) for the full rename table and rationale.

### 3.1 Cache invariants — what is enforced (and what is not)

`Data.__setattr__` runs on **assignments**:

```python
data = forward_kinematics(model, q)         # _kinematics_level = PLACEMENTS
data.q = new_q                              # __setattr__ fires; downstream caches invalidated
J = compute_joint_jacobians(model, data)   # raises StaleCacheError — must call FK first
```

The `__setattr__` hook handles `q`, `v`, `a` — and any other declared input
field — by dropping caches above the appropriate level and resetting
`_kinematics_level`. Functions advance the level explicitly; nothing else
writes to it.

**Limitation: in-place tensor mutation is not detected.**

```python
data.q[..., 0] += 1.0      # silently stale; __setattr__ NOT called
data.q.add_(other)         # silently stale
data.q.copy_(new_q)        # silently stale
```

This is a fundamental Python limitation — tensor views and in-place ops
bypass dataclass attribute machinery. The contract this layer pins is:

> BetterRobot detects **reassignment** of `q`, `v`, `a`, and any public input
> field on `Data`, and invalidates downstream caches on reassignment.
> In-place tensor mutation of those fields is **not** detected; it is a
> documented misuse pattern.

User-facing guidance, mirrored in `Data`'s docstring:

- **Reassign**, do not mutate. `data.q = new_q`. Never `data.q[...] = new_q[...]`.
- If a hot path absolutely needs in-place updates, follow them with
  `data.invalidate(KinematicsLevel.NONE)` — the public method that resets
  the level explicitly.
- IK and trajopt solvers reassign; user code patterned after them stays in
  the safe zone.

A future hardening could store `q` behind a property returning a read-only
view — but that adds overhead and breaks autograd in subtle ways. We do not
pay that cost pre-1.0. The acceptance test in
`tests/contract/test_cache_invariants.py` includes a "documented limitation"
case asserting that `data.q[..., 0] += 1.0` is *not* detected, codifying
the scope of this contract so a future contributor cannot quietly claim more.

### 3.2 Future split — `RobotState` vs `AlgorithmCache`

Once trajectory optimisation matures (D7 in
[06_DYNAMICS.md §5](06_DYNAMICS.md), and the optim wiring of
[07_RESIDUALS_COSTS_SOLVERS.md](07_RESIDUALS_COSTS_SOLVERS.md) §16.A–F lands)
the `Data` class is likely to be split into a value-semantics `RobotState`
(`q`, `v`, `a`, `tau`) and a per-query `AlgorithmCache` (the FK / Jacobian
buffers and the `_kinematics_level` field). That refactor is **not in
scope for v1**. Doing it before optimiser internals stabilise would force
the optimisers to depend on a state surface we have not yet pinned. Revisit
post the optim-wiring landing; the change is small at that point.

## 4. `JointModel` protocol

```python
# src/better_robot/data_model/joint_models/base.py

from __future__ import annotations
from typing import Protocol, Literal
import torch

JointKind = Literal[
    "universe",
    "fixed",
    "revolute_rx", "revolute_ry", "revolute_rz", "revolute_unaligned", "revolute_unbounded",
    "prismatic_px", "prismatic_py", "prismatic_pz", "prismatic_unaligned",
    "spherical",
    "free_flyer",
    "planar",
    "translation",
    "helical",
    "composite",
    "mimic",
]

class JointModel(Protocol):
    """Protocol every concrete joint implements.

    A JointModel is a *stateless* dispatch object with a compile-time nq/nv.
    It does not store per-joint runtime data — that lives on `Model` via
    `joint_placements`, `idx_q`, `idx_v`, etc. Implementations are pure
    functions that operate on slices of the full (q, v, a) tensors.
    """

    kind: JointKind
    nq: int
    nv: int
    axis: torch.Tensor | None           # (3,) for axis-based joints; None otherwise

    def joint_transform(
        self,
        q_slice: torch.Tensor,          # (B..., nq)  the joint's own q slice
    ) -> torch.Tensor:                  # (B..., 7)   SE3 pose of child in parent frame
        ...

    def joint_motion_subspace(
        self,
        q_slice: torch.Tensor,          # (B..., nq)
    ) -> torch.Tensor:                  # (B..., 6, nv)  S(q)
        ...

    def joint_velocity(
        self,
        q_slice: torch.Tensor,          # (B..., nq)
        v_slice: torch.Tensor,          # (B..., nv)
    ) -> torch.Tensor:                  # (B..., 6)   spatial velocity in joint frame
        ...

    def integrate(
        self,
        q_slice: torch.Tensor,          # (B..., nq)
        v_slice: torch.Tensor,          # (B..., nv)
    ) -> torch.Tensor:                  # (B..., nq)  retraction q ⊕ v
        ...

    def difference(
        self,
        q0_slice: torch.Tensor,         # (B..., nq)
        q1_slice: torch.Tensor,         # (B..., nq)
    ) -> torch.Tensor:                  # (B..., nv)  q1 ⊖ q0
        ...

    def random_configuration(
        self,
        generator: torch.Generator | None,
        lower: torch.Tensor,            # (nq,)
        upper: torch.Tensor,            # (nq,)
    ) -> torch.Tensor:                  # (nq,)
        ...

    def neutral(self) -> torch.Tensor:  # (nq,)
        ...
```

Concrete joint modules (`revolute.py`, `prismatic.py`, …) implement this
protocol as **module-level stateless functions** grouped into a `@dataclass`.
No per-joint Python objects, no `nn.Module` overhead per joint — the
`joint_models` tuple on `Model` holds *kind instances*, and Model-level tensors
(`joint_placements`, `axes`) supply the per-joint numerical data.

Adding a new joint type is a single-file change. See
[15_EXTENSION.md §2](../conventions/15_EXTENSION.md).

## 5. Joint taxonomy (v1)

| Class | Kind | nq | nv | Notes |
|-------|------|----|----|-------|
| `JointUniverse` | `universe` | 0 | 0 | Root placeholder. Joint 0. |
| `JointFixed` | `fixed` | 0 | 0 | Rigid link to parent. |
| `JointRX` / `JointRY` / `JointRZ` | `revolute_r{x,y,z}` | 1 | 1 | Axis-aligned revolute. |
| `JointRevoluteUnaligned` | `revolute_unaligned` | 1 | 1 | Arbitrary 3-vector axis. |
| `JointRevoluteUnbounded` | `revolute_unbounded` | 2 | 1 | Continuous (cos,sin) param; no joint limit. |
| `JointPX` / `JointPY` / `JointPZ` | `prismatic_p{x,y,z}` | 1 | 1 | Axis-aligned prismatic. |
| `JointPrismaticUnaligned` | `prismatic_unaligned` | 1 | 1 | Arbitrary 3-vector axis. |
| `JointSpherical` | `spherical` | 4 | 3 | SO3 ball joint, (qx,qy,qz,qw). |
| `JointFreeFlyer` | `free_flyer` | 7 | 6 | SE3 floating base. |
| `JointTranslation` | `translation` | 3 | 3 | 3-DOF translation. |
| `JointPlanar` | `planar` | 4 | 3 | SE2 + planar pos (x,y,cosθ,sinθ). |
| `JointHelical` | `helical` | 1 | 1 | Pitch-coupled rotation + translation. |
| `JointComposite` | `composite` | Σ | Σ | Stack of sub-joints. |
| `JointMimic` | `mimic` | 0 | 0 | Zero-DOF, value derived from another joint via `mult*q + off`. |

### Mimic (PyRoki trick)

Instead of a runtime if-statement, `Model` carries:

```python
mimic_multiplier: torch.Tensor   # (njoints,)  1.0 for non-mimic joints
mimic_offset:     torch.Tensor   # (njoints,)  0.0 for non-mimic joints
mimic_source:     tuple[int, ...]  # index of source joint for mimic joints; self-index otherwise
```

The full-q expansion `q_full[i] = mult[i] * q_active[src[i]] + off[i]` is a
vectorised gather + multiply with **no Python branching** on joint type.

## 6. `Frame` — the indirection layer

```python
# src/better_robot/data_model/frame.py

import torch
from dataclasses import dataclass
from typing import Literal

FrameType = Literal["body", "joint", "fixed", "op", "sensor"]

@dataclass(frozen=True)
class Frame:
    """Pinocchio-style Frame: a named point rigidly attached to a parent joint.

    Every frame lives *on a joint*, not on a link; it stores a fixed placement
    `joint_placement` expressed in the parent joint's local frame. For a body
    frame this is the inertial origin of the link; for operational frames
    (e.g. "tool0", "camera_optical") it's wherever the user asked it to be.
    """

    name: str
    parent_joint: int
    joint_placement: torch.Tensor  # (7,) SE3 in parent joint's frame
    frame_type: FrameType
```

Frames let users attach arbitrary named reference points (tool tips, IMU
mount, virtual marker positions) to any joint without inflating the joint
count or the body count. IK targets, visualisations, and residuals all
address frames by name:

```python
frame_id = model.frame_id("panda_hand_tcp")
target_pose = data.frame_pose_world[..., frame_id, :]
```

## 7. Configuration layout: `q` and `v`

Everything lives in a **flat leading-batch tensor**:

```
q  shape  (B..., nq)
v  shape  (B..., nv)
```

with per-joint slices determined by `model.idx_qs[j]` and `model.nqs[j]`
(ditto `idx_vs`, `nvs`). The `forward_kinematics` implementation iterates
the topo order and slices:

```python
qj = q[..., idx_q : idx_q + nq_j]
vj = v[..., idx_v : idx_v + nv_j]
Tj = joint_models[j].joint_transform(qj)
```

No string dispatch, no type-specific branches outside the joint modules
themselves.

## 8. Universal `integrate` / `difference`

```python
def integrate(model: Model, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Manifold retraction: q ⊕ v. Per-joint dispatch via joint_models."""
    q_new = torch.empty_like(q)
    for j in range(model.njoints):
        jm = model.joint_models[j]
        iq, iv = model.idx_qs[j], model.idx_vs[j]
        qj = q[..., iq : iq + jm.nq]
        vj = v[..., iv : iv + jm.nv]
        q_new[..., iq : iq + jm.nq] = jm.integrate(qj, vj)
    return q_new
```

The for-loop over joints is at *model compile time*: after `build_model()`,
the topology is fixed and the loop unrolls cleanly under
`torch.compile`/`torch.jit.script`. This is how a floating base works without
any "floating base mode" — `JointFreeFlyer.integrate` handles the SE3
retraction; `JointRX.integrate` is simple addition. The `tasks/ik.py` layer
*never* sees the distinction.

## 9. `Body` and `Inertia`

```python
# src/better_robot/data_model/body.py

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Body:
    """A rigid body attached to a joint via `model.joint_placements[parent_joint]`.

    Bodies carry inertial properties. Inertia is stored as a packed 10-vector
    on `model.body_inertias`; this struct is the metadata.
    """
    name: str
    parent_joint: int
    visual_geom: Optional[int] = None     # index into collision/visual geometry
```

`Inertia` lives in `spatial/inertia.py` (see 03 LIE_AND_SPATIAL).

## 10. Already retired (reference)

These structures were removed when the skeleton landed; listed here so
downstream readers tracing old code can find their replacements:

- `src/better_robot/models/robot_model.py` → `data_model/model.py`
- `src/better_robot/models/data.py` → `data_model/data.py`
- `src/better_robot/models/joint_info.py` → absorbed into `data_model/model.py`
- `src/better_robot/models/link_info.py` → `data_model/body.py` + `frame.py`
- `RobotModel._frozen` custom guard → `@dataclass(frozen=True)`
- `RobotModel._fk_joint_parent_link / _fk_joint_child_link / _fk_joint_types /
  _fk_cfg_indices / _fk_joint_axes / _fk_joint_order / _root_link_idx` →
  proper `parents`, `topo_order`, `joint_models`, `idx_qs`, etc.
- "base_pose" as a separate argument — folded into `q[0:7]` via
  `JointFreeFlyer`. The `_solve_floating_*` family of solvers is gone.

## 11. Deprecated-name shims (one-release window)

`Data` field names follow the readable convention in
[13_NAMING.md §4](../conventions/13_NAMING.md) (`joint_pose_world`,
`frame_pose_world`, `bias_forces`, etc.). For one release, `Data`
exposes the Pinocchio cryptic aliases (`oMi`, `oMf`, `liMi`, `nle`, `Ag`)
as `@property` shims that emit `DeprecationWarning` and forward to the
new field. The shim table lives in `data_model/data.py::_DEPRECATED_ALIASES`.

```python
@property
def oMi(self):
    warnings.warn("Data.oMi is deprecated; use Data.joint_pose_world. "
                  "See docs/conventions/13_NAMING.md",
                  DeprecationWarning, stacklevel=2)
    return self.joint_pose_world
```

The shim set is exactly the left column of [13_NAMING.md §4](../conventions/13_NAMING.md).
All shims are removed in v1.1. Internal library code does **not** use the
shims — the contract test in `tests/contract/test_naming.py` enforces this.
