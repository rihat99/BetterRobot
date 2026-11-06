# 0010 Identity equality for models and states, one by-value static key

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`hash(model.structure)` raises `TypeError: unhashable type: 'dict'`. Principle 5.5 says the static structure
compares and hashes by value, checked by a test that equal structures hash equal. A generated dataclass `__eq__` over
tensor fields cannot work: it returns a tensor, so `identity_a == identity_b` raises `RuntimeError: Boolean value of
Tensor ... is ambiguous` (verified on BetterHuman `Identity`). Two arms that differ only in link lengths must not
compare equal, and link lengths are tensor values.

## Decision
We will give `Model`, `KinematicState`, `HumanModel`, `Identity` and `HumanState` identity equality and identity hash
(`eq=False`), and give `Model` one by-value hashable static key, `structure_key`, that keys the plans (0011) and is
covered by the 5.5 test.

## Example
```python { .no-run }
a, b = br.load("panda.urdf"), br.load("panda.urdf")
a == b                                # False: identity equality
a.structure_key == b.structure_key    # True: a frozen tuple of the static fields
hash(a.structure_key)                 # works; two arms with different link lengths share it, by design
```

## Consequences
The ruling "static structure hashes by value" holds for the one object that needs it, for the price of one public
property. A model is never compared by value, so a test that wants numeric equality uses `torch.equal` on the fields.
`structure_key` ignores tensor values: it identifies a topology, not a robot.

## Alternatives
Identity equality only, with a written exception to principle 5.5: plans are then keyed per instance and two loads of
one URDF build their plans twice. By-value `__eq__` and `__hash__` over the static fields of `Model`: two arms with
different link lengths compare equal, which is wrong. JAX compares the static part of a pytree by value to key its
compilation cache, which is the role `structure_key` plays; Pinocchio's `Model::operator==` compares every field by
value, which a tensor library cannot do.

## Compatibility
Adds one name. Removes by-value dataclass equality that never worked on tensor fields. Reversible: yes.

## Enforcement
New `tests/contract/test_hashing.py` (two loads of one URDF share `structure_key` and its hash, a different topology
does not, models with different link lengths are not equal); `tests/contract/test_frozen.py`;
`BetterHuman/tests/contract/test_frozen.py` for the three BetterHuman types.

## Open questions
None.
