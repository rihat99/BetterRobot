# 0009 `Model` is one flat frozen record

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`data_model/model.py::Model` forwards 47 one-line properties to `ModelStructure` and `ModelValues`, so every field has
two names. 44 of the 78 `ModelStructure` fields are never read outside build, structure and model code, and
`model_structure.py::validate_consistency` makes 13 `.tolist()` host round trips per build. A Panda `Model` prints
31,471 characters. Callers reach past `Model` into the pair: `dataclasses.replace(model.values, gravity=...)` in
BetterVideoReconstruction, `model.structure` in BetterRetarget, `model.structure.frame_parent_joint_ids` in
BetterHuman `terms.py::contact_wrenches`.

## Decision
We will make `Model` one frozen dataclass with flat fields (static topology, tensor values, one declared plan cache),
registered as a pytree over its tensor fields, with `with_values(...)` as the checked replace and no forwarding
properties.

## Example
```python { .no-run }
@dataclass(frozen=True, eq=False)
class Model:
    joint_names: tuple[str, ...]            # static
    parents: tuple[int, ...]                # static
    joints: tuple[JointModel, ...]          # static
    joint_placements: Tensor                # (I..., njoints, 7)  tensor
    q_neutral: Tensor                       # (nq,)               tensor
    _plans: dict = field(default_factory=dict, init=False, repr=False, compare=False)

model = br.load("panda.urdf").with_values(gravity=gravity)   # was dataclasses.replace(model.values, ...)
repr(model)   # "Model('panda', nq=8, nv=8, njoints=14, float64, cpu)"
```

## Consequences
One name per field; the 47 forwarding properties, the 44 unread fields and the per-build host syncs are deleted.
`torch.func` over placements and inertias keeps working through the pytree hook. Costs: the hook calls the private
`torch.utils._pytree` once, as `model_values.py` does today, and every field now lives on one larger object, so a
field table in the class docstring marks each field static or tensor.

## Alternatives
Keep the public pair with forwarding properties (today): two names per field. Keep the pair private behind `Model`:
hides nothing and keeps the 47 one-liners. Pinocchio's `Model` is one object; its split is `Model` (constant) and
`Data` (mutable workspace), which decision ADR 0002 already rejects. pyroki's `Robot` is one pytree dataclass with
static fields; Drake's `MultibodyPlant` is one object and its `Context` is the state.

## Compatibility
Breaks `ModelStructure`, `ModelValues`, `Model.structure` and `Model.values` for every caller: BetterVideoReconstruction
1 site, BetterRetarget 3, BetterHuman 1 plus `terms.py`, BetterRobot `terms/robot.py` 2. Undoing it after the apps
migrate is another breaking change.

## Enforcement
`tests/contract/test_frozen.py` (frozen); new `tests/contract/test_model_fields.py` (every field is declared static or
tensor in the class field table, a pytree round trip keeps the static fields, `Model` has no forwarding property and
no `meta` field); new `tests/contract/test_repr.py` (one line, principle 4.9).

## Open questions
None.
