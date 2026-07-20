# Order 03 — Model = ModelStructure + ModelValues, no triplication

Read `plan/README.md` and `DESIGN_RULES.md` first. This is the widest-churn
order; the full gate, Pinocchio parity, and the autograd suite are the oracle
at every step. No numerical behavior may change.

## The problem (audited facts)

`Model` (`data_model/model.py`, 555 lines) declares ~48 data fields, then
`__post_init__` builds `ModelStructure.from_model(self)` and
`ModelValues.from_model(self)` **from those same fields** — ~44 fields exist
in triplicate. Keeping the copies coherent costs: the `_shallow_rebind`
field-copy loop (`model.py:130-154`), a hand-maintained `field_updates` dict
in `with_values` (`model.py:197-204`), and a second one enumerating 14 value
tensors + 3 structure tensors in `to()` (`model.py:218-254`). A missed entry
silently desyncs the flat field from the part. Consumers read both
representations inconsistently (compute passes take `structure, values`;
`optim/variables.py:431-447`, `optim/lm.py:223-225`, `residuals/limits.py`,
viewer code read flat fields).

## Target design

The owner's vision: `Model = ModelStructure + ModelValues`, zero redundancy.
Users may hold a `Model` and never learn the parts exist, or work with the
parts directly; the transition is trivial because Model is nothing but the
pair.

1. **Stored state:** `Model` keeps exactly `structure: ModelStructure`,
   `values: ModelValues`, `reference_configurations`, and `meta`. Nothing
   else is stored.
2. **Flat access stays working** — `model.nq`, `model.joint_names`,
   `model.joint_placements`, … become one-line `@property` delegators to the
   owning part, grouped in two clearly-titled blocks. Explicit properties,
   not `__getattr__` magic (typed, greppable).
3. **Methods move to their owner:** `joint_id`/`frame_id`/`body_id`,
   `get_subtree`/`get_support`, `q_permutation` → `ModelStructure`;
   `body_inertia`, `spatial_inertias` → `ModelValues`. `Model` keeps thin
   delegators for public-API stability plus the genuine combiners:
   `integrate`, `difference`, `random_configuration`, `create_data`,
   `with_values`, `to`.
4. **Construction:** `ModelStructure` and `ModelValues` are built directly
   (in `io/build_model.py:911-959`, today the only construction site) and
   passed to `Model(structure=..., values=...)`. `ModelStructure.from_model`
   / `ModelValues.from_model` disappear with the flat fields they read.
5. **Rebind/move become trivial:** `with_values` = build a new `ModelValues`,
   wrap; `to(device)` = move each part, wrap. Both `field_updates` dicts and
   `_shallow_rebind`'s field loop are deleted. `dataclasses.replace`
   semantics on the parts must keep the T4-era regression
   (`test_dataclasses_replace_keeps_spatial_inertia_physics_and_gradients_live`)
   green.
6. `execution_batch.py`, `data.py`, `joint*`, `topology.py` are out of scope
   except where they read a moved symbol.

## Consumers

Flat-field readers keep working through the properties — the deliberate
outcome is that most of the codebase does not churn. Update the sites that
construct or rebind models (`io/build_model.py`, `Model.to/with_values`
callers if signatures shift) and any code touching removed internals
(`from_model`, `_shallow_rebind`). Pytree/functorch registration and
`torch.compile` FK tests must stay green — check the pytree flatten path in
`data_model` if registered.

## Docs and contracts

- Update `data_model/CLAUDE.md`, `docs/concepts/architecture.md`, and the
  generated API pages for `Model`/`ModelStructure`/`ModelValues`.
- MIGRATION.md rows: removed classmethods and any renamed accessor.
- Contract files you may touch: docstring/public-import contracts listing
  `data_model` symbols. Layer-dependency and parity tests: read-only.

## Acceptance

- Full gate + parity + contracts + CUDA suite green; autograd tests through
  `with_values`/`replace` prove gradients still flow to replaced tensors.
- `model.py` ≤ ~320 lines; zero `field_updates` dicts; grep shows no
  `from_model(` in src.
- One construction path: `git grep "Model("` in src shows only the
  structure+values form (tests may keep convenience fixtures).
- Line accounting in `03_results.md`.
