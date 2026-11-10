# model/ traps

The field table is the `Model` class docstring (`model.py`); the plan sharing between copies is explained beside
`_plan_owner`.

- A model is a value: never edit its tensors in place. Queries memoise placement matrices and inertias on the model
  (record 0040), so an edited model keeps serving the old values.
  ```python
  # Right: a validated copy that shares the plans and starts with an empty value memo.
  model = model.with_values(joint_placements=placements)
  # Wrong: forward kinematics keeps the matrices it memoised before the edit.
  model.joint_placements[3, 2] += 0.1
  ```
  `Model.to(device, dtype)` moves every tensor field the same way. `Model(...)` itself validates nothing (no
  `__post_init__`): `src/better_robot/io/model_from_ir.py` builds the record, and `with_values` is the public rebind.
- A new field goes into the class docstring table as `static` or `tensor`; no property forwards to a part, and
  there is no `__getattr__` (`tests/contract/test_model_fields.py`). `nq`, `nv`, `njoints`, `dtype` and `device`
  are derived.
- Constant tensors derived from the static fields are plans, built once per dtype and device by
  `Model._plan(kind, build, *, dtype=, device=)`; no other cache exists (`tests/contract/test_caches.py`).
- Every public boundary checks each tensor input once with `src/better_robot/lie/_checks.py::check_tensor` (a dtype
  or device other than the model's raises) and aligns its batch with `batch_shape` and `expand_batch`; raw passes
  trust the result (`tests/contract/test_boundary_validation_count.py`).
- Joint 0 is the zero-DOF universe; a floating base is joint 1, a `JointFreeFlyer`. Index order is topological
  (`parent_joint_ids[i] < i`).
- `nq != nv` for manifold joints: step and subtract with `kinematics.integrate` and `kinematics.difference`, and
  slice a joint's coordinates with `idx_qs`/`idx_vs`, never with its joint id.
- Mimic targets have zero-width public slices. `mimic_source_ids`, `mimic_multipliers` and `mimic_offsets` hold each
  target's resolved chain; `kinematics.full_configuration` expands `q` to every joint's coordinates.
- `Model.q_permutation` gives the gather tables for another tool's joint order; do not reorder `q` by hand.
- A new joint kind: `.claude/skills/add-joint-kind/SKILL.md`.
