# 0050 `testing` imports `optim` for the robust-kernel checker

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0035](0035-shared-testing-module.md) puts the checks that BetterRobot's and BetterHuman's tests share in
`better_robot.testing`, whose row of the edge table lists what its tools import: `exceptions` and `model` (for
`check_joint_model`). A robust kernel of `optim` (`Huber`, `Cauchy`, `Tukey`, `GemanMcClure`, or one a user writes)
has a contract of its own: `rho(0)` is 0, `rho` never decreases, and `weight` is `2 d rho / ds`, the factor the
solvers apply. One shared checker, `check_robust_kernel`, placed beside `check_joint_model`, runs every kernel
against that contract and takes an `optim.RobustKernel`, so `testing` needs an edge to `optim`. An added edge needs
a record (principle 11.6).

## Decision
We will let `testing` import `optim`, so `better_robot.testing.check_robust_kernel` checks any
`optim.RobustKernel` the way `check_joint_model` checks a joint kind.

## Example
```python { .no-run }
from better_robot import optim
from better_robot.testing import check_robust_kernel

check_robust_kernel(optim.Huber(delta=1.0), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
```
```text
testing           -> exceptions model optim
```

## Consequences
`testing` stays at the top of the graph (no package imports it) and `optim` at the bottom (it imports only
`exceptions`), so the graph keeps no cycle and no package changes band. `import better_robot.testing` loads
`optim`, which imports nothing outside the package but torch. A kernel a user writes is checked by the same
function as the library's.

## Alternatives
Keep the checker a private helper of BetterRobot's tests: BetterHuman and a user who writes a kernel could not run
it, unlike `check_joint_model`. Type the checker against a protocol of its own instead of `optim.RobustKernel`: a
second definition of the kernel interface.

## Compatibility
A public name added under `testing` and an import inside the package; no existing name, signature or layout
changes.

## Enforcement
`tests/contract/test_layer_dependencies.py` (`ALLOWED["testing"]`); `tests/docs/test_codemap.py` (the copy in
`docs/concepts/architecture.md`).

## Open questions
- Accept the `testing` -> `optim` edge, or keep the kernel checker out of `better_robot.testing`?
