# 0049 `collision` imports `lie` for the one input validator

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Every public function checks each tensor input once at its boundary (contract 06, C-ERR-1), through the one
validator of the library, `src/better_robot/lie/_checks.py::check_tensor` and its batch form `check_tensors`, which
`lie`, `spatial`, `model`, `kinematics`, `dynamics`, `io` and `terms` import. The public `collision` functions read
tensors too, and the `collision` row of the edge table (`exceptions`, `model`, `kinematics`) could not reach `lie`.
An added edge needs a record (principle 11.6); the edge entered `ALLOWED` in
`tests/contract/test_layer_dependencies.py` on 2026-11-10 without one.

## Decision
We will let `collision` import `lie`, so its public functions validate their tensor inputs through
`src/better_robot/lie/_checks.py` like every other package.

## Example
```python { .no-run }
# src/better_robot/collision/distance.py
from ..lie._checks import check_tensor, check_tensors
```
```text
collision         -> exceptions lie model kinematics
```

## Consequences
The edge points down: `lie` sits below `kinematics`, which `collision` already imports, so the graph keeps no cycle
and no package changes band in `docs/figures/architecture.svg`. A wrong dtype, device or trailing shape at a
`collision` boundary raises the same typed error as at any other boundary.

## Alternatives
Move the validator into a private unit of its own beside `_internal_checks`, imported by every package: one more
unit and an edge from every row, for a module `lie` itself needs. Check through `model`'s boundary helpers: a
forwarding layer (principle 1.2) for a check `lie` already holds. Leave the `collision` functions unchecked: they
break C-ERR-1.

## Compatibility
An import inside the package only; no public name, signature or layout changes.

## Enforcement
`tests/contract/test_layer_dependencies.py` (`ALLOWED["collision"]`); `tests/docs/test_codemap.py` (the copy in
`docs/concepts/architecture.md`); `tests/contract/test_boundary_validation_count.py` (C-ERR-1).

## Open questions
- Accept the `collision` -> `lie` edge, or move the validator into a unit of its own (the first alternative)?
