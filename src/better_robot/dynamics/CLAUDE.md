# dynamics/ — inverse and forward dynamics, mass matrix, centroidal quantities

Entry points, all on `(model, state, ...)` with `state = forward_kinematics(model, q)`:
`inverse_dynamics(*, v, a, external_forces=None)` (`inverse_dynamics.py`), `mass_matrix`,
`forward_dynamics(*, v, tau, external_forces=None)` (`mass_matrix.py`), `center_of_mass`, `centroidal_map`,
`centroidal_momentum(*, v)` (`centroidal.py`). Design notes and numbers live in the `_world.py` docstring;
contracts in `docs/contracts/00-index.md`.

## Traps

- `_world.py` is the only implementation: no per-joint loops, no second RNEA. Sums over ancestors or subtrees
  are products with the constant support matrices of `DynamicsPlan`.
- Never memoise on the state from here: `world_kinematics` reads `state.jacobian_columns` (kinematics' memo)
  and derives the rest per call (`tests/contract/test_caches.py`).
- World quantities are about the first joint's position (`WorldKinematics.origin`), not the world origin;
  shift twists and points by `origin`, and keep it, or float32 accuracy degrades far from the world origin
  (`tests/dynamics/test_state_dynamics.py`).
- `external_forces (B..., njoints, 6)` is one wrench `[force, torque]` per joint, in that joint's own frame;
  gravity enters as the base acceleration. Inside RNEA a joint force is the wrench the joint transmits to its
  subtree.
- `model.armature (nv,)` is added in the public wrappers (`+ armature * a`, `+ diag`), never in `_world`;
  centroidal quantities exclude it, as in Pinocchio.
- The kernel calls `spatial`'s unchecked `*_raw` twins; only the public wrappers check inputs, once.
- Frame changes use the joint rotation matrices; the quaternion `spatial.transform_*` cost 2-3x here.
- Pinocchio parity (`tests/parity/pinocchio/`) is the oracle; never weaken it.
- The functions `inverse_dynamics` and `mass_matrix` shadow their modules in the package namespace: reach a
  module with `importlib.import_module("better_robot.dynamics.mass_matrix")`.
