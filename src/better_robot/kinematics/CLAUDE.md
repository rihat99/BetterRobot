# kinematics/ traps

The design notes and their numbers live in the module docstrings of `_fk.py`, `_jacobian.py` and `_plans.py`.

- Jacobians default to `reference="local_world_aligned"`: linear rows are the velocity of the frame origin in world
  axes, angular rows the world angular velocity. Ask for the body-frame twist with `reference="local"`; by hand,
  rotate both row blocks, never apply the full adjoint:
  ```python
  # Right: the same as frame_jacobian(model, state, frame=frame, reference="local").
  R = state.frame_rotations[..., frame, :, :]
  J_local = torch.cat([R.mT @ J[..., :3, :], R.mT @ J[..., 3:, :]], dim=-2)
  # Wrong: the full adjoint adds a spurious cross term.
  J_local = se3.adjoint_inverse(state.frame_poses[..., frame, :]) @ J
  ```
- `J̇` pairs with a different acceleration per reference: `world` and `local` close `a = J v̇ + J̇ v` with the
  spatial acceleration, `local_world_aligned` with the classical (point) acceleration of the frame origin.
  Mixing them is an order-one error, not roundoff.
- A state's batch is `broadcast(B..., I...)` of `q` with the model's value batch; shapes that do not
  broadcast raise `ShapeError`. Add singleton axes yourself (`(B, 1, njoints, 7)` next to `(B, T, nq)`);
  nothing unsqueezes.
- `KinematicState` stores the joints' `[R | p]` matrices; `frame_transforms` is derived from them on first use
  (dynamics never reads it). `joint_poses` and `frame_poses` convert on first use with
  canonical-sign quaternions (`so3.from_matrix`), so a quaternion may be `-q` against a composition lane
  (same rotation). Compare rotations, not quaternion signs.
- `KinematicState._memo` has one writer, `_memoised`, keyed by `(name, mode)` like `Model._value`: `mode` is the
  grad mode, or `"inference"` under `torch.inference_mode`, so an inference tensor is served only
  where autograd records nothing (`enable_grad` reports grad mode on there); `torch.compile` bypasses it, and reads
  `Model._value` under the grad-mode key without writing it.
  `world_geometry` (the dynamics' view of the state) is a state property like the others; no other
  package writes the memo. A new name goes into the table of
  `docs/contracts/03-state-mutability.md` in the same change. `dataclasses.replace` starts with an empty memo
  and recomputes.
- Plans (`_plans.py`) are pure functions of the static fields, built by `Model._plan` once per dtype and
  device. Never read a tensor's `_version` or write through `object.__setattr__`
  (`tests/contract/test_caches.py`).
- Placement matrices (the joints' folded into the FK map) are memoised on the model by mode and the tensor fields'
  `requires_grad` flags: a model whose tensors are edited in place after a query serves the
  old matrices. Build a new model with `with_values`.
- `check_quaternion_norm=True` reads values and syncs the device; it is the only value check and covers
  every spherical and free-flyer joint. Below the public wrappers everything trusts validated inputs: no
  `.item()`, `bool(tensor)` or Python branch on a value in the hot path
  (`tests/contract/test_hot_path_lint.py`).
- Public wrappers trace as one graph under `torch.compile(fullgraph=True)` (`tests/kinematics/test_compilation.py`
  and the zero graph breaks of `tests/contract/test_compile_counters.py`): loop over joints and tree levels (Python
  ints from the plan), never over batch elements.
- `compiled` (`compilation.py`) is the only `torch.compile` call. Each key (grad and inference
  mode; per argument a tensor's dtype, device, `requires_grad` and axes, a model's name, a string or number outside
  a dataclass, another object's type) has its own `torch.compile` and 8 graphs; its first call, and the first call of
  a model argument new to it, runs `fn` eagerly, so plans and memos are graph inputs, not traced code. A batched product of small matrices goes through `small_matmul`, and the FK joint
  map's einsum takes the same multiply-and-sum form, only while `compiled` traces (`fused_forms`): eagerly and under
  a caller's own `torch.compile` they stay `@` and einsum. Read the generated code before trusting a speedup:
  inductor can inline the level chain into one kernel and recompute it.
- `nq != nv` for quaternion and planar joints: step with `integrate(model, q, v)`, subtract with
  `difference(model, q0, q1)`, never add a tangent to `q`. Mimic joints have no public coordinates:
  `full_configuration` expands `q`, `reduced_columns` contracts Jacobian columns and generalized forces.
- Rare joint kinds (planar, translation, helical, custom) go through the joint's own `placement` inside the
  kernel; a new kind is a `JointModel` subclass, never a branch in `_fk.py`.
