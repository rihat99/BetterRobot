# dynamics/ traps

The design notes and their numbers live in the `_world.py` docstring.

- `_world.py` is the only implementation: no per-joint loops, no second RNEA. Sums over ancestors or subtrees
  are products with the constant support matrices of `DynamicsPlan`.
- Never write the state's memo: `world_kinematics` reads `state.world_geometry`, which kinematics memoises
  (record 0041; keyed by grad mode, bypassed under inference mode and compile); the 6x6 inertias come
  from the model's value memo (`Model._value`, record 0040), never from the state (`tests/contract/test_caches.py`).
- World quantities are about the first joint's position (`WorldKinematics.origin`), not the world origin;
  shift twists and points by `origin`, and keep it, or float32 accuracy degrades far from the world origin
  (`tests/dynamics/test_state_dynamics.py`).
- `external_forces (B..., njoints, 6)` is one wrench `[force, torque]` per joint, in that joint's own frame;
  gravity enters as the base acceleration. Inside RNEA a joint force is the wrench the joint transmits to its
  subtree.
- `model.armature (nv,)` is added in the public wrappers (`+ armature * a`, `+ diag`), never in `_world`;
  centroidal quantities exclude it, as in Pinocchio.
- The kernel calls `spatial`'s unchecked `*_raw` twins; only the public wrappers check inputs, once.
- Frame changes use the joint rotation matrices, not the quaternion `spatial.transform_*` (why: `_world.py`).
- Pinocchio parity (`tests/parity/pinocchio/`) is the oracle; never weaken it.
