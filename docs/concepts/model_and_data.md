# Frozen `Model`, mutable `Data`

BetterRobot follows the Pinocchio split:

| Object | Mutability | Purpose |
|--------|-----------|---------|
| `Model` | Frozen `@dataclass(frozen=True)` | Topology, joint models, limits, frames, body inertias |
| `Data` | Mutable `@dataclass` | Per-query workspace: `joint_pose_world`, `joint_jacobians`, `mass_matrix`, … |

`Model` is read-only and shareable across threads / GPUs. Every
kinematics / dynamics call takes a `Model` and a `Data`, writes into
`Data`, and never mutates `Model`. `model.create_data()` allocates a
fresh workspace shaped to a given batch.

## Cache invariants

`Data` carries a `_kinematics_level` (NONE → PLACEMENTS → VELOCITIES →
ACCELERATIONS). Reassigning `data.q` invalidates strictly-higher
caches; routines that depend on a level call `data.require(level)` to
abort early with a `StaleCacheError` instead of silently returning a
stale field.

The cache is the seam between the rigid-body algorithms — it lets
`compute_centroidal_map` and `crba` share work with `forward_kinematics`
without leaking that coupling into the public API.
