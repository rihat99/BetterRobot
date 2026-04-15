# data_model/ — Model, Data, Frame, Body, Joint

Pinocchio-style architecture: immutable Model (tree description) + mutable Data (per-query workspace).

## The Triple

| Object | Mutability | Purpose |
|--------|-----------|---------|
| `Model` | Frozen dataclass | Shared kinematic tree: topology, joint models, limits, frames |
| `Data` | Mutable dataclass | Per-query workspace: `oMi`, `oMf`, `J`, `M`, etc. |
| `Frame` | Immutable | Metadata: name, parent joint, placement SE3, type |

## Joint 0 Convention

Joint 0 is always `universe` (root placeholder). First real joint is joint 1. For floating-base robots, joint 1 is `JointFreeFlyer` — no special "floating base mode" flag.

## Joint Model Protocol

Every joint type implements `JointModel` with:
- `.nq`, `.nv` — configuration and tangent dimensions
- `.joint_transform(q)` — returns SE3 7-vector for the joint's own motion
- `.integrate(q, v)` — manifold retraction (addition for revolute, SE3 for free-flyer)
- `.difference(q0, q1)` — tangent vector between configurations

All per-kind logic lives in `joint_models/`. The FK loop does no type dispatch — it calls `jm.joint_transform(qj)` uniformly.

## nq != nv

Free-flyer: nq=7 (quaternion), nv=6 (twist). Spherical: nq=4, nv=3. `model.idx_qs` and `model.idx_vs` map each joint to its slice of q and v.

## Mimic Joints

Handled via tensors (`mimic_multiplier`, `mimic_offset`, `mimic_source`) with no Python branching. Mimic joints have nq=0, nv=0.

## Adding a New Joint Type

1. Create class in `joint_models/` implementing `JointModel`
2. Add kind string to `Joint` enum in `joint.py`
3. Add (nq, nv) to `JOINT_DIMENSIONS`
4. Wire dispatch in `io/build_model.py`
