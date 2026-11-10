# 0047 `Problem` registration only adds names; derived problems are sealed

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0017](0017-optim-api.md) makes a `Problem` "immutable after construction and changed by `problem.with_`", yet
terms and shared intermediates are added by the registration calls `residual`, `scalar` and `cached` after the
constructor. Before this record a callback could register a term on the view `run` handed it and change the problem
its caller still held, and a term registered after `with_` reached one problem but not the other. Callers also had
no public way to see a problem's cached values or which solver made a `State`.

## Decision
We will let a `Problem` be built by its constructor and its registration calls, which only add new names; keep every
field of a registered term fixed; seal the problems that `with_` returns and the views that `run`,
`solve_constrained` and `solve_layer` hand to callbacks, so that registering on them raises `ArgumentError`; and add
the read-only `Problem.cached_values` and `optim.State.solver` (the solver class that made the state, which a warm
start checks).

## Example
```python { .no-run }
problem = optim.Problem({"x": x0})
problem.residual("fit", fit_term)                 # registration adds a name
problem.cached("prediction", predict)
heavier = problem.with_(weights={"fit": 10.0})    # a new, sealed problem; `problem` is unchanged
heavier.residual("extra", extra_term)             # ArgumentError: register every term before with_
result = optim.solve(problem, optim.LevenbergMarquardt(), iterations=50)
assert result.state.solver is optim.LevenbergMarquardt
```

## Consequences
A problem a caller holds changes only through the calls that caller makes, and a callback cannot reach back into it.
Cost: a term must be registered before the first `with_`; a builder that wants another term on a derived problem
registers it on the original and derives again. Whether a problem should be built entirely by its constructor, with
no registration methods, is left to a later record.

## Alternatives
Construction only, with every term passed to the constructor: the strict reading of 0017, a larger change to every
caller. Registration everywhere, with `with_` copying the registries: a term added later reaches one problem but not
its copies. JAX optimizers (optax, jaxopt) build the objective once as a pure function; Theseus mutates an `Objective`
in place.

## Compatibility
Breaks registration on a `with_` result or on a callback's view; adds two read-only names.

## Enforcement
C-STATE-5; `tests/optim/test_problem.py` (`test_problem_is_immutable`,
`test_callbacks_cannot_register_on_the_views_run_hands_them`); `tests/contract/test_api_snapshot.py`.

## Open questions
None.
