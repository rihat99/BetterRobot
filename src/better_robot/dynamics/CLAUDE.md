# dynamics/ — Inverse and Forward Dynamics, Mass Matrix, Centroidal

## Entry points

- `state.py`, all on a `KinematicState`: `inverse_dynamics(model, state, *, v, a, fext=)`,
  `mass_matrix(model, state)`, `forward_dynamics(model, state, *, v, tau, fext=)`,
  `centroidal_map(model, state)`, `centroidal_momentum(model, state, v=v)`,
  `center_of_mass(model, state)`. World-frame quantities are built once per state and
  memoised in `state._memo["world_kinematics"]`. Bias forces are `inverse_dynamics(model,
  state, v=v, a=0)`; gravity torque is `inverse_dynamics(model, state, v=0, a=0)`.

## One core, no joint loops

`_world.py` holds the only implementation. Everything lives in the world frame: the shared
Jacobian column table `C (N, 6, nv_full)` from `kinematics/_jacobian_columns.py`, each joint's
rotation and origin, the packed body inertias. Sums over ancestors or subtrees are products with
constant support matrices (`DynamicsPlan`, memoised on the structure), so operator counts are
independent of the joint count (Panda RNEA 387 operators, tree200 390).

- Twists and wrenches move between the world and a joint frame with `3x3` products and cross
  products (`WorldKinematics.to_joint_frames`, `to_world_force`), never with `6x6` adjoints.
- RNEA forms wrenches in the joint frames where the inertia is constant; CRBA needs the world
  inertia (`WorldKinematics.world_inertia`, assembled from `R I_com Rᵀ`).
- Constant matrices multiply from the right (`x.mT @ support`) so torch folds the batch into one
  `mm`; a left multiply becomes a slow batched `bmm`.
- ABA is a Cholesky solve on `M`; with mimic joints the projected equations are also the only
  correct form.

## Cost model

At batch 1 and in backward the operator count dominates. At large CPU batches every operator
streams an `(N, njoints, 6)` tensor through memory, so a per-joint loop on cache-resident
slices is not much slower in forward mode; the mass matrix `(N, nv, nv)` is then the cost
(tree200 at 4096 is 655 MB). Numbers: `benchmarks/results/p2_world_dynamics_cpu.json` vs
`0953f28_cpu.json`.

## Contracts

- `fext` is per joint in the joint's own frame; gravity enters as the base acceleration.
- Inside RNEA a joint force is the wrench the joint transmits to its subtree, in the joint frame.
- Parity with Pinocchio (`tests/test_pinocchio/`) is the oracle; do not weaken it.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
