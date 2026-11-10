# `model/` — Robot Identity

`Model` (`model.py`) is one flat frozen dataclass: static fields (names, joints,
parent joint ids, the `q`/`v` layout, the resolved mimic chain) and eleven tensor
fields (placements, inertias, limits, armature, damping, gravity, `q_neutral`).
The class docstring holds the field table. Query results live in a frozen
`KinematicState` (`src/better_robot/kinematics/forward.py`), never on the model; do not mutate
model tensors in place.

- No `__post_init__` validation: `src/better_robot/io/model_from_ir.py` constructs the record;
  `Model.with_values(**fields)` is the validated public rebind (every tensor
  field; placements normalised; the three `I...`-batched fields must broadcast)
  and `Model.to(device, dtype)` moves every tensor field. Both return a new
  record that shares its source's plans (through `_plan_owner`) and starts
  with an empty value memo. The link is strong on purpose: a copy keeps the
  first model of its line (tensors, value memo, plans) alive and pickles it
  along; a weak link would break pickling and drop the sharing.
- `nq`, `nv`, `njoints`, `dtype`, `device`, ... are derived properties. Never
  add forwarding properties or `__getattr__`.
- `structure_key` hashes the static fields: equal keys mean the same topology
  and index tables.
- Plans (constant tensors derived from the static fields: FK, Jacobian,
  dynamics, mimic, manifold) are built on first use by
  `Model._plan(kind, build, *, dtype, device)` and kept in `_plans`
  (`init=False`, never copied; a copy with the same `structure_key` takes its
  `_plan_owner`'s plan instead of building one). The model is a registered
  pytree: leaves are the tensor fields, context the static fields (not the
  plan link).
- Every public boundary checks each tensor input once with the one validator,
  `src/better_robot/lie/_checks.py::check_tensor` (a dtype or device other than the model's
  raises), then aligns its batch with `_checks.py` (`batch_shape`,
  `expand_batch`); raw passes trust the result.
- `manifolds.py` holds the six `JointManifold`s (fixed, Euclidean, circle, rotation,
  plane, rigid): `integrate`/`difference` written once each as the group exponential
  and logarithm; `src/better_robot/kinematics/configuration.py` groups the public coordinates by
  manifold and runs one batched call per group behind `kinematics.integrate`
  and `kinematics.difference`.
- `joints/` holds the abstract `JointModel` and the nine final kinds, one file
  per kind; `topology.py` the pure tree helpers on `parent_joint_ids`.

## Coordinate rules

- Joint 0 is the zero-DOF universe; a floating base is joint 1 as
  `JointFreeFlyer`. Index order is a topological order (`parent_joint_ids[i] < i`).
- `nq != nv` for manifold joints. Use `kinematics.integrate` and
  `kinematics.difference`; use `idx_qs`/`idx_vs` for slices.
- `Model.q_permutation` supplies scalar gather tables for external joint
  orders and supports arbitrary leading batches.
- Mimic targets have zero-width public slices. `mimic_source_ids`,
  `mimic_multipliers` and `mimic_offsets` hold each target's resolved chain
  (public root joint, total scale, total offset); `src/better_robot/kinematics/configuration.py`
  builds the expansion maps from them, and `src/better_robot/io/model_from_ir.py` reduces the
  limits with the same chain.

## Extending joints

Subclass `JointModel` in `joints/` as a `@final` frozen dataclass declaring
`nq`, `nv`, `manifold` and `axis`, run `better_robot.testing.check_joint_model`
on it and export it from `src/better_robot/model/__init__.py`; `src/better_robot/io/model_from_ir.py` needs no
entry, it reads the limits' shape from the kind's `manifold`. `src/better_robot/kinematics/_plans.py` reads a kind's class only for
the affine fast path; every other kind goes through `placement`. Joints carry
no kind string. Preserve scalar-last quaternions and linear-first tangents
throughout.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
