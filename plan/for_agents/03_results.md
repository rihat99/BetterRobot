# Order 03 results — model ownership redesign

Status: complete on `dev` (2026-07-20).

## Delivered

- `Model` now stores exactly `structure`, `values`,
  `reference_configurations`, and `meta`; all former flat fields are explicit,
  typed properties over their single owner.
- `ModelStructure` owns names, lookup/traversal/permutation methods, static
  frame metadata, and validated topology/device tables. `ModelValues` owns all
  differentiable tensors plus body/spatial inertia access.
- `build_model` directly constructs both parts and has the only source
  `Model(...)` call. The flat constructor, both part `from_model` classmethods,
  `_shallow_rebind`, and both `field_updates` dictionaries are gone.
- `with_values` and `to` wrap moved/replaced parts. Frame objects are synthesized
  from static metadata and the current placement table, so rebound/batched
  frame values cannot desynchronize.
- Updated the data-model guide, architecture/model concepts, migration table,
  frozen-shape contract, and generated API pages.

## Verification

- Full CPU gate: **1,507 passed, 2 skipped, 16 deselected**.
- CUDA gate on GPU 2: **15 passed, 1,510 deselected**.
- Pinocchio parity: **135 passed**.
- Contract suite: **335 passed**.
- Public autograd suite: **17 passed**.
- Documentation tests: **27 passed**; Sphinx doctest: **30 passed**.
- Sphinx dummy build succeeded with the same four unreachable external
  intersphinx inventories seen in prior orders.
- Changed Python files pass Ruff check/format; source compilation and
  `git diff --check` pass.

Adversarial parity compared an all-joint model against Order 02 and found
identical structure/value tensors and identical `integrate`, `difference`, and
seeded random-configuration results. It also verified gradients, batching,
serialization, q permutations, mixed-device frame packing, and `to("meta")`.

## Acceptance greps

- `model.py`: **305 lines** (requested 250–310).
- Dataclass fields: exactly the four target fields; no `__getattr__`.
- One `Model(...)` occurrence under `src/`, passing `structure` and `values`.
- Zero `ModelStructure.from_model`, `ModelValues.from_model`,
  `_shallow_rebind`, or `field_updates` references under the model construction
  scope.

## Physical source-line accounting

| Scope | Before | After | Delta |
|---|---:|---:|---:|
| `data_model/model.py` | 555 | 305 | −250 |
| `data_model/model_structure.py` | 480 | 323 | −157 |
| `data_model/model_values.py` | 186 | 160 | −26 |
| `data_model/_model_manifold.py` | 0 | 206 | +206 |
| `io/build_model.py` | 961 | 1,183 | +222 |
| **Listed implementation files** | **2,182** | **2,177** | **−5** |
| **All `src/**/*.py`** | **20,684** | **20,679** | **−5** |

## Deviations and findings

- No numerical behavior deviated from the plan. To satisfy both the required
  48 explicit properties and the hard 250–310-line shell target, the unchanged
  manifold implementations live in private `_model_manifold.py`; `Model`
  retains the three public methods as delegators. Autodoc does not expose the
  private module.
- The literal repo-wide `from_model(` grep still finds the unrelated,
  pre-existing `RobotCollision.from_model` constructor. The two classmethods
  targeted by this order are fully removed; renaming the collision constructor
  would be an unrelated public behavior change.
- `model.frames` is now a synthesized tuple, so tuple object identity is not
  stable across accesses. Its contents remain numerically equivalent and now
  correctly reflect rebound or batched frame placements, as required by the
  zero-redundancy design.
- `docs/concepts/model_and_data.md` received a one-line truth fix in addition
  to the explicitly listed documentation files: the parts are storage owners,
  not duplicated “views.”
