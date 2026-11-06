# 0021 One structural validator per entry, opt-in value checks, one internal-checks switch

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Two validators do one job (`_validation.py::check_tensor` and `kinematics/state.py::_validate_q`). Public paths scan
tensor values: `Inertia.from_mesh` runs `bool(torch.isfinite(vertices).all())` and `Model.random_configuration` checks
each joint, each a host sync. `integrate` promotes mixed dtypes silently. `_validate_free_flyer_quaternion_norm`
checks joint 1 only. Principle 2.8 sets the order of checks; 6.8 puts invariants that read values behind one switch.

## Decision
We will put types on every public signature, run one structural validator once per public entry, trust validated
state in the passes below it, make every value check an opt-in argument that names its cost, raise
`DtypeMismatchError` on mixed dtypes, and add one library-wide switch (off by default, on in the test tiers) that
turns on the internal invariants which read tensor values.

## Example
```python { .no-run }
state = br.forward_kinematics(model, q)                                # shape, dtype, device checked once; no value read
state = br.forward_kinematics(model, q, check_quaternion_norm=True)   # covers every quaternion joint; docstring: "one host sync"
inertia = br.spatial.inertia_of_mesh(vertices, faces, density=1000.0, check_mesh=True)   # opt-in
model.random_configuration(generator=generator)   # raises ModelError when a joint has no finite limits (a static fact)
```

## Consequences
Hot paths read no values by default. A finite-limit joint check becomes a static property of the model. The switch is
the environment variable `BETTER_ROBOT_INTERNAL_CHECKS`, read once by `better_robot._internal_checks`. Cost: a caller
who passes a NaN gets a NaN, not an error, unless the switch or the opt-in argument is on.

## Alternatives
The same without the switch: every value-reading invariant would need its own flag or be dropped. Value scans as
general input validation: syncs on every call. Drake validates sizes at the API boundary and uses assertions for the
rest; Pinocchio checks argument sizes with macros and trusts values.

## Compatibility
Breaks mixed-dtype `integrate` calls (now an error), the default value scans, and `random_configuration` on models
with infinite limits. Reversible: yes, a check can be added later.

## Enforcement
`tests/contract/test_boundary_validation_count.py` generalised over the `tests/contract/_public_ops.py` registry
(exactly one structural validation per entry); `tests/contract/test_submodule_public_imports.py::test_raw_kernels_are_not_public`;
`tests/contract/test_hot_path_lint.py` (no `isfinite` scan in a public path); `tests/contract/test_dtype_device.py`
(mixed dtypes raise); new `tests/contract/test_internal_checks.py` (the switch is read once; no value-reading check
outside its guard in `src`).

## Open questions
None.
