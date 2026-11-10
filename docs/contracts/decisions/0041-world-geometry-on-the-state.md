# 0041 The world geometry is a memoised value of `KinematicState`, owned by kinematics

**Status:** Accepted (integrator ruling R-2, 2026-11-10); amended by 0042. **Date:** 2026-11-10

## Context
Record [0011](0011-caches-and-plans.md) derived the dynamics' world geometry per call; its dated note of 2026-11-06
memoised it on the state after all, written by `src/better_robot/dynamics/_world.py::world_kinematics` through
`state._memoised("world_geometry", ...)`. That is dynamics computing a value and storing it in another package's
object through that package's private method: the side door principle 1.5 names as its anti-example. The memo
itself is needed: deriving the geometry per call put crba 11-16 % and aba 8-11 % behind R4 at batch 1, and
`forward_dynamics` derived it twice. The geometry is purely kinematic (the state's Jacobian columns moved to the
first joint, the joint rotations, their origins relative to it and that origin).

## Decision
We will make the world geometry a memoised value of `KinematicState`, built and stored by kinematics: the property
`KinematicState.world_geometry` returns `(columns (N, 6, nv_full), rotations (N, njoints, 3, 3), translations
(N, njoints, 3), origin (N, 3))` about each sample's first joint, derived by
`src/better_robot/kinematics/_jacobian.py::geometry_about_first_joint` from the state's own transforms and
`jacobian_columns`, and memoised by the state's one writer, `KinematicState._memoised`, under `"world_geometry"`
(keyed by grad mode, bypassed under inference mode and compile, like every state memo). Dynamics reads the
property; no package calls another's private method. This supersedes the "Note 2026-11-06" of record 0011 (the
writer it names); the rest of 0011 stands.

## Example
```python { .no-run }
# KinematicState, in kinematics
@property
def world_geometry(self):
    return self._memoised(
        "world_geometry", lambda: geometry_about_first_joint(self.joint_transforms, self.jacobian_columns)
    )

# world_kinematics, in dynamics
columns, rotations, translations, origin = state.world_geometry
```

## Consequences
Same numbers and the same cost as the note's memo: the derivation is the same operators in the same order (the
Pinocchio parity margins are byte-identical; crba/aba/rnea at batch 1 measured in the RA notes). The geometry
reads only the state and the structure of its model, so it is right for any model `check_state` accepts (same
structure). `KinematicState` gains one property; contract 03 C-STATE-4 lists `"world_geometry"` as filled by it.

## Alternatives
- Derive the geometry per call (record 0011 as written): crba +11-16 %, aba +8-11 % at batch 1.
- A record allowing memo keys written by other packages: needs principle 1.5 to change.

## Compatibility
No public name changes. `dynamics._world.world_geometry` (private) is gone; nothing outside dynamics used it.

## Enforcement
C-STATE-4 (the memo table); `tests/contract/test_caches.py` (one writer per cache, the memo keyed by grad mode and
bypassed under inference mode, read through the dynamics); principle 1.5.

## Open questions
