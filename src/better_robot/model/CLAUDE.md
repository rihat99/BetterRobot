# `model/` — Robot Identity

`Model` (`model.py`) is one flat frozen dataclass: static fields (names, joints,
parents, the `q`/`v` layout, the resolved mimic chain, `meta`) and eleven tensor
fields (placements, inertias, limits, armature, damping, gravity, `q_neutral`).
The class docstring holds the field table. Query results live in a frozen
`KinematicState` (`kinematics/state.py`), never on the model; do not mutate
model tensors in place.

- No `__post_init__` validation: `io/build_model.py` constructs the record;
  `Model.with_values(**fields)` is the validated public rebind (every tensor
  field; placements normalised; the three `I...`-batched fields must broadcast)
  and `Model.to(device, dtype)` moves every tensor field. Both return a new
  record with no plans.
- `nq`, `nv`, `njoints`, `dtype`, `device`, ... are derived properties. Never
  add forwarding properties or `__getattr__`.
- `structure_key` hashes the static fields: equal keys mean the same topology
  and index tables.
- Plans (constant tensors derived from the static fields: FK, Jacobian,
  dynamics, mimic, manifold) are built on first use by
  `Model._plan(kind, build, *, dtype, device)` and kept in `_plans`
  (`init=False`, never copied). The model is a registered pytree: leaves are
  the tensor fields, context the static fields.
- `_checks.py` is the one structural validator (`check_tensor`, `batch_shape`,
  `expand_batch`): every public boundary calls it once per tensor input, raw
  passes trust the result, and a dtype or device other than the model's raises.
- `manifolds.py` holds the five `Manifold`s (Euclidean, circle, rotation, plane,
  rigid): `integrate`/`difference` written once each as the group exponential
  and logarithm. `_manifold.py` groups the public coordinates by manifold and
  runs one batched call per group behind `Model.integrate`/`Model.difference`.
- `joints/` holds the abstract `JointModel` and the nine final kinds, one file
  per kind; `topology.py` the pure tree helpers on `parents`.

## Coordinate rules

- Joint 0 is the zero-DOF universe; a floating base is joint 1 as
  `JointFreeFlyer`. Index order is a topological order (`parents[i] < i`).
- `nq != nv` for manifold joints. Use `Model.integrate` and
  `Model.difference`; use `idx_qs`/`idx_vs` for slices.
- `Model.q_permutation` supplies scalar gather tables for external joint
  orders and supports arbitrary leading batches.
- Mimic targets have zero-width public slices. `mimic_source_ids`,
  `mimic_multipliers` and `mimic_offsets` hold each target's resolved chain
  (public root joint, total scale, total offset); `kinematics/configuration.py`
  builds the expansion maps from them, and `io/build_model.py` reduces the
  limits with the same chain.

## Extending joints

Subclass `JointModel` in `joints/` as a `@final` frozen dataclass declaring
`nq`, `nv`, `manifold` and `axis`, run `better_robot.testing.check_joint_model`
on it, export it from `src/better_robot/model/__init__.py`, then map its IR kind string in
`io/build_model.py`. `kinematics/_fk_lean.py` reads a kind's class only for
the affine fast path; every other kind goes through `placement`. Joints carry
no kind string. Preserve scalar-last quaternions and linear-first tangents
throughout.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
