# 0017 `optim` takes strings for reads and an immutable `Problem`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Terms are plain functions wired by parameter names, but `Problem.residual` probes `getattr(fn, "jac")` and `"window"`,
and `run` probes `getattr(phase.solver, "warm_keys", ())`. Reads are `str | CacheRef`. `problem.data` is mutated by
users (`examples/01_ik.py`) and by `Phase.on_start`. `result["q"]` aliases `result.x["q"]`. `TorchOptim(optimizer_cls,
**kwargs)` hides its signature, and `optim` raises plain `ValueError` because it imported only torch.

## Decision
We will provide `residual(name, fn, *, reads=None, jacobian=None, window=None)` with `TermFn` as the one bundle type,
`Solver.warm_keys` declared on the abstract base, string reads (no `CacheRef`), a `Problem` that is immutable after
construction and changed by `problem.with_(data=, weights=, free=)`, `Phase.on_start(problem, x) -> (x, data)`,
`TorchOptim(factory, *, scheduler=None)` and the full-word names `iterations`, `gradient_tolerance`,
`loss_tolerance`, `step_tolerance` and `jacobian`, with `optim` allowed to import `exceptions` (0020).

## Example
```python { .no-run }
problem = terms.ik_problem(model, {"panda_hand": target}, q0=q0)
problem = problem.with_(data={"preferred_elbow": torch.tensor(-1.8)})        # was problem.data[...] = ...
adam = optim.TorchOptim(lambda params: torch.optim.Adam(params, lr=0.05))    # was TorchOptim(torch.optim.Adam, lr=0.05)
result = optim.solve(problem, adam, iterations=1500)                         # result.x["q"]; no result["q"]
```

## Consequences
`solve_layer`'s promise "differentiable in the data" is visible at the call site: new data is a new `Problem`.
No `getattr` probe remains in `optim`. Cost: a callback that used to write into `problem.data` now returns data.

## Alternatives
Keep `problem.data[...] = ...` as one documented mutable seam: users and callbacks keep writing shared state. Keep it
as is. Theseus keeps a mutable name-keyed `Objective` that cost functions and optimizers update in place, which ADR
0003 rejects; optax and jaxopt model a pure `(init, update)` over value-typed state, which is the direction taken
here.

## Compatibility
Breaks `CacheRef`, `Result.__getitem__`, `problem.data` writes, the old tolerance and `iters` names and
`TorchOptim(cls, **kwargs)`. Migration is mechanical: BetterVideoReconstruction 5 `TorchOptim` calls, 2 `on_start`
callbacks and 2 `State.it` reads; BetterRetarget 12 sites; BetterHuman's IK example.

## Enforcement
`tests/optim/test_problem.py` (C-STATE-5; `test_unknown_read_fails_at_registration` for C-ERR-5; new
`test_problem_is_immutable`: `Problem` is frozen and `with_` returns a new problem);
`tests/contract/test_keyword_only.py`; the edge table row `optim` allows only `exceptions`.

## Open questions
None.
