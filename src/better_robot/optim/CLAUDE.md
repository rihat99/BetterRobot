# optim — Agent Guide

Optimization layer over named tensors. It imports only torch. Robot terms live in `better_robot.terms`.

## Model

- A variable is a named tensor `(B, *event)`; the `B` problems of a batch are independent.
- A term is a plain function. Its parameter names are the variables, data tensors or cached
  values it reads (`reads=` overrides). `TermFn` bundles a function with `reads`, an analytic
  `jac` and a `window`.
- `Problem.loss` is an ordinary differentiable `(B,)` tensor. `Problem.linearize` builds
  Gauss-Newton normal equations from the same terms: one traced evaluation for all autodiff
  terms, analytic blocks where a term has `jac`.
- `Problem(time=(...))` marks variables `(B, T, *event)`. A residual with `window=k` reads `k`
  consecutive knots. If every residual is time-local or reads no time variable, the normal
  equations are block-tridiagonal (`BandedNormal`), bordered by variables without a time axis
  (`ArrowNormal`). Otherwise they are dense (`Normal`). All three share one interface.
- Solvers are `init(problem, x) -> State` and `step(problem, state) -> State`. `Phase` + `run`
  chain them with weight overrides and free masks. `solve_layer` makes a solve differentiable
  w.r.t. data. `solve_constrained` adds hard constraints (augmented Lagrangian).

## Invariants

- Nothing inside a solver `step` reads a tensor back to Python (two exceptions: the validity check
  of autodiff columns of shared variables in a time-local term, and the `LBFGS` line search, which
  stops once every problem has its step: one read per trial); shapes never change. Done or
  rejected batch elements keep their old values through `torch.where`.
- Cached values are computed at most once per evaluation and stay inside the autodiff trace.
- `window=` is trusted; `check_jacobian` is the test for it and for every analytic `jac`.
- After `step`, `State.extra` carries the diagnostic keys named in `solvers.DIAGNOSTICS` (key set in the
  `State` docstring); `run` stacks them into `Result.diagnostics`. A solver's `warm_keys` are the `extra`
  entries a warm start (`state=`) keeps; everything else is recomputed by `init`. Every solver also writes
  `extra["terms"]` (per-term losses of the evaluation `State.loss` comes from; `TorchOptim` adds `extra["info"]`,
  the dicts of terms returning `(value, info)`), kept as device tensors: no host sync.
- A `time=` problem that mixes windowed terms with an unwindowed term reading a time variable warns and
  goes dense. Normal equations are built through a sink (`add`/`finish`); a new structure is a new sink.
- `Problem.with_` returns a view with its own `terms`, `data`, `weights`; never mutate the parent.
- Linear solves run in float64 with unit-diagonal scaling whatever the problem dtype.
- No `NotImplementedError` here (the contract test inventories those); raise `ValueError`.

## Checks

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest tests/optim -q
uv run python examples/01_ik.py   # 02..05 likewise
```

A new Jacobian route or linear-solver path must be compared against `Problem(dense=True)`
and row-by-row autodiff in `tests/optim/test_problem.py`.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
