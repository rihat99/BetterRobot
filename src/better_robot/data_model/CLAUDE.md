# data_model/ — Model/Data and the Compute Seam

Pinocchio-style architecture: immutable Model (tree description) + mutable Data (per-query workspace).

## Core objects

| Object | Mutability | Purpose |
|--------|-----------|---------|
| `Model` | Frozen dataclass | Shared kinematic tree: topology, joint models, limits, frames |
| `Data` | Mutable dataclass | Per-query workspace: `joint_pose_world`, `frame_pose_world`, `joint_jacobians`, `mass_matrix`, etc. |
| `ModelStructure` | Frozen dataclass | Static Python topology mirrors plus equivalent flat device tables for whole-pass lanes |
| `ModelValues` | Frozen tensor pytree | Differentiable placements, inertias, limits, gravity, and mimic values |
| `ExecutionBatch` | Frozen dataclass | Flat-`E` broadcast ABI with per-input index maps and gradient reduction |
| `Frame` | Immutable | Metadata: name, parent joint, placement SE3, type |

Every `Model` constructs `.structure` and `.values`. Raw Torch passes consume
that explicit pair; an eligible Warp kernel consumes the same seam. Keep both
representations of topology: Python tuples preserve static unrolling, while
device tables provide the kernel ABI.

## Joint 0 Convention

Joint 0 is always `universe` (root placeholder). First real joint is joint 1. For floating-base robots, joint 1 is `JointFreeFlyer` — no special "floating base mode" flag.

## Joint Model Protocol

Every joint type implements `JointModel` with:
- `.nq`, `.nv` — configuration and tangent dimensions
- `.joint_transform(q)` — returns SE3 7-vector for the joint's own motion
- `.integrate(q, v)` — manifold retraction (addition for revolute, SE3 for free-flyer)
- `.difference(q0, q1)` — tangent vector between configurations

Per-kind implementations live in `joint_models/`; `joint_dispatch.py`
centralises the stable kind-code mapping used by whole-pass algorithms. Do
not duplicate parser-string dispatch inside FK or dynamics.

## nq != nv

Free-flyer: nq=7 (quaternion), nv=6 (twist). Spherical: nq=4, nv=3. `model.idx_qs` and `model.idx_vs` map each joint to its slice of q and v.

## Mimic Joints

Non-identity mimic joints are rejected at build (`NotImplementedError`); reduced-coordinate enforcement is scheduled for M3. Exact identity tags are temporarily accepted for Panda compatibility, but they are **not coupled**: both joints retain independent coordinates until M3. The zero-DOF `JointMimic` placeholder cannot be selected directly because that would bypass this policy.

## Adding a New Joint Type

1. Create class in `joint_models/` implementing `JointModel`
2. Add kind string to `Joint` enum in `joint.py`
3. Add (nq, nv) to `JOINT_DIMENSIONS`
4. Wire dispatch in `io/build_model.py`
