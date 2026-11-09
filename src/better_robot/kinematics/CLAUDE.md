# kinematics/ — forward kinematics, Jacobians, configuration steps

Entry points: `forward_kinematics(model, q, *, check_quaternion_norm=False) -> KinematicState`;
`frame_jacobian`, `joint_jacobian`, `point_jacobian`, `frame_jacobian_time_variation` and
`joint_jacobian_time_variation` on `(model, state, *, ...)`; `integrate`, `difference`, `full_configuration`,
`euclidean_columns` (`configuration.py`); `subtree`, `support` (`tree.py`). Design notes and numbers live in the
module docstrings of `_fk.py`, `_jacobian.py` and `_plans.py`; contracts in `docs/contracts/00-index.md`.

## Traps

- Jacobians are `local_world_aligned` by default: linear rows are the velocity of the frame origin in world
  axes, angular rows the world angular velocity. For the body-frame twist pass `reference="local"`. Never
  apply a full adjoint to an LWA Jacobian (`se3.adjoint_inverse(T) @ J` adds a spurious cross term); by hand,
  rotate the two row blocks by `R.mT`.
- `J̇` pairs with a different acceleration per reference: `world` and `local` close `a = J v̇ + J̇ v` with the
  spatial acceleration, `local_world_aligned` with the classical (point) acceleration of the frame origin.
  Mixing them is an order-one error, not roundoff.
- A state's batch is `broadcast(B..., I...)` of `q` with the model's value batch; shapes that do not
  broadcast raise `ShapeError`. Add singleton axes yourself (`(B, 1, njoints, 7)` next to `(B, T, nq)`);
  nothing unsqueezes.
- `KinematicState` stores `[R | p]` matrices; `joint_poses` and `frame_poses` convert on first use with
  canonical-sign quaternions (`so3.from_matrix`), so a quaternion may be `-q` against a composition lane
  (same rotation). Compare rotations, not quaternion signs.
- `KinematicState._memo` has one writer, `_memoised`, keyed by `(name, grad mode)` and bypassed under
  `torch.inference_mode` and `torch.compile` like `Model._value`: `enable_grad` inside inference mode reports
  grad mode on, so a stored inference tensor would break a later autograd call of the same state.
  `world_geometry` (the dynamics' view of the state, record 0041) is a state property like the others; no other
  package writes the memo. A new name goes into the table of
  `docs/contracts/03-state-mutability.md` in the same change. `dataclasses.replace` starts with an empty memo
  and recomputes.
- Plans (`_plans.py`) are pure functions of the static fields, built by `Model._plan` once per dtype and
  device. Never read a tensor's `_version` or write through `object.__setattr__`
  (`tests/contract/test_caches.py`).
- Placement matrices are memoised on the model by grad mode and the tensor fields' `requires_grad` flags, never
  under `torch.inference_mode` (record 0040): a model whose tensors are edited in place after a query serves the
  old matrices. Build a new model with `with_values`.
- `check_quaternion_norm=True` reads values and syncs the device; it is the only value check and covers
  every spherical and free-flyer joint. Below the public wrappers everything trusts validated inputs: no
  `.item()`, `bool(tensor)` or Python branch on a value in the hot path
  (`tests/contract/test_hot_path_lint.py`).
- Public wrappers trace as one graph under `torch.compile(fullgraph=True)`
  (`tests/kinematics/test_fk_compile.py`): loop over joints and tree levels (Python ints from the plan),
  never over batch elements.
- `nq != nv` for quaternion and planar joints: step with `integrate(model, q, v)`, subtract with
  `difference(model, q0, q1)`, never add a tangent to `q`. Mimic joints have no public coordinates:
  `full_configuration` expands `q`, `reduced_columns` contracts Jacobian columns and generalized forces.
- Rare joint kinds (planar, translation, helical, custom) go through the joint's own `placement` inside the
  kernel; a new kind is a `JointModel` subclass, never a branch in `_fk.py`.
