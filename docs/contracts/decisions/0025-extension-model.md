# 0025 Extension without registries: five bricks, one way in each

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Extension runs today through a module-global parser registry (`io.register_parser` accepts `".sdf"`, which never
matches), joint hooks found by `getattr` that nothing calls (`joint_bias_acceleration`,
`joint_motion_subspace_derivative`) and kind strings. Room for future features is wanted, without a mechanism that a
caller can misuse silently.

## Decision
We will extend the library through five bricks with one way in each: joint kinds subclass the abstract `JointModel`,
terms are functions added to a `Problem` by name, parsers are functions returning `IRModel`, layers subclass `Layer`
and solvers subclass `Solver`, with no global registry and any fast path landing behind the plain signature with the
plain path as its oracle and a decision record.

## Example
```python { .no-run }
def parse_sdf(path: Path) -> IRModel: ...                  # a new format is a function
model = br.io.build_model(parse_sdf(Path("robot.sdf")))    # no registration step

class MyLayer(br.viewer.Layer): ...                        # abstract methods enforce the contract
problem = problem.residual("reach", my_term)               # a term is a plain function wired by parameter names
```

## Consequences
Nothing mutates global state to extend the library, and an unknown extension fails where it is used. A second backend
goes on the commonly rejected list (8.3) with ADR 0001. A typed registry stays possible later as an addition.

## Alternatives
A typed parser registry so `br.load("robot.sdf")` works for formats BetterRobot does not ship: a global table with
import-order effects. MuJoCo keeps a process-global plugin registry and Theseus a mutable name-keyed `Objective`;
Pinocchio closes its joint set in a variant. Subclassing an abstract base gives errors at definition time, not at a
distant dispatch.

## Compatibility
Removes `register_parser`. Adding a registry later is compatible; no app uses one today.

## Enforcement
Contract C-EXT-* in `docs/contracts/08-extension-points.md`, rewritten to this form;
`tests/contract/test_abstract_or_final.py` (0013); new `tests/contract/test_extension_points.py` (`src/better_robot`
defines no module-level mutable registry and no `register_*` function); the commonly rejected list.

## Open questions
None.
