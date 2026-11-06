# 0024 Remove the public names nobody uses

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Public names with no caller outside BetterRobot's own tests (searched in BetterHuman, BetterVideoReconstruction and
BetterRetarget): `Trajectory`, `io.builders.*`, `StateMultibody`, the five asset resolvers and `register_parser`, the
`Joint` enum, `JOINT_DIMENSIONS`, `Body`, `Frame`, `FrameType`, `ExecutionBatch`, `flatten_execution_batch`,
`JOINT_KIND_CODES`, `reference_configurations`, `friction`, `rotor_inertia` (always zero), `JointComposite`,
`JointUniverse`, `_typing.py`, `spatial/ops.py`, `so3.adjoint` (an alias of `to_matrix`), `se3.apply_base`,
`viewer.skeleton_spec`, `spread_offsets` and the `kinematics.jacobian_columns` alias. `Model.frames` has one caller,
BetterRetarget `viewer/app.py`.

## Decision
We will remove every name listed in Context, so that the 187 listed public names become about 150.

## Example
```python { .no-run }
# BetterRetarget viewer/app.py: the one caller of Model.frames
parent_joint = model.frame_parent_joint_ids[frame_index]     # was model.frames[frame_index].parent_joint
placement = model.frame_placements[frame_index]
```

## Consequences
Fewer promises to keep and test. Features that look like a future need (a clip type, a composite joint) are not kept
on speculation. Related removals are argued in 0013 (joints), 0014 (spatial classes) and 0038 (resolvers, registry).

## Alternatives
Keep `Trajectory` as a future clip type or `JointComposite` as a future joint kind: each costs tests and a promise
for no caller. Adding a name back later is a compatible change, so removal is the cheaper error. Drake's stability
guarantee lists deprecated names with a removal date rather than carrying unused ones indefinitely.

## Compatibility
Breaks only code that imported an unused name; the one app caller of `Model.frames` is shown above. A name added back
later is compatible.

## Enforcement
`tests/contract/test_public_surface.py` and `tests/contract/test_api_snapshot.py` (fast tier) fail on any name
outside the decided surface; the changelog lists each removal under Removed with its replacement.

## Open questions
None.
