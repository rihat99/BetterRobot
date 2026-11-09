# optim — Agent Guide

Optimization layer over named tensors. It imports torch and `better_robot.exceptions` only
(`tests/contract/test_layer_dependencies.py`, `tests/contract/test_optional_imports.py`). Robot terms live in
`better_robot.terms`.

## Model

- A variable is a named tensor `(B, *event)`; the `B` problems of a batch are independent.
- A term is a plain function. Its parameter names are the variables, data tensors or cached
  values it reads (`reads=` overrides). `TermFn` bundles a function with `reads`, an analytic
  `jacobian` and a `window`.
- `Problem.loss` is an ordinary differentiable `(B,)` tensor. `Problem.linearize` builds
  Gauss-Newton normal equations from the same terms: one traced evaluation for all autodiff
  terms, analytic blocks where a term has `jacobian`.
- `Problem(time=(...))` marks variables `(B, T, *event)`. A residual with `window=k` reads `k`
  consecutive knots. If every residual is time-local or reads no time variable, the normal
  equations are block-tridiagonal (`BandedNormal`), bordered by variables without a time axis
  (`ArrowNormal`). Otherwise they are dense (`Normal`). All three share one interface.
- Solvers are `init(problem, x) -> State` and `step(problem, state) -> State`. `Phase` + `run`
  chain them with weight overrides and free masks. `solve_layer` makes a solve differentiable
  w.r.t. data. `solve_constrained` adds hard constraints (augmented Lagrangian).

## Invariants

Each invariant names its enforcer; `[review]` means a reviewer checks it.

- Nothing inside a solver `step` reads a tensor back to Python (two exceptions: the validity check
  of autodiff columns of shared variables in a time-local term, and the `LBFGS` line search, which
  stops once every problem has its step: one read per trial); shapes never change. Done or
  rejected batch elements keep their old values through `torch.where`.
  `tests/contract/test_hot_path_lint.py` (watches `src/better_robot/optim/solvers.py`),
  `tests/optim/test_solver_state.py::test_lbfgs_line_search_stops_once_every_problem_has_a_step`, `[review]`.
- Cached values are computed at most once per evaluation and stay inside the autodiff trace.
  `tests/optim/test_problem.py::test_linearize_gradient_matches_autograd_of_loss`.
- `window=` is trusted; `check_jacobian` is the test for it and for every analytic `jacobian`.
  `tests/optim/test_problem.py::test_time_local_terms_match_row_by_row_autodiff`,
  `tests/terms/test_robot_terms.py::test_analytic_jacobians_match_autodiff`.
- After `step`, `State.extra` carries the diagnostic keys named in `solvers.DIAGNOSTICS` (key set in the
  `State` docstring); `run` stacks them into `Result.diagnostics`. A solver's `warm_keys` are the `extra`
  entries a warm start (`state=`) keeps; everything else is recomputed by `init`. `State.solver` is the class
  that made the state; a warm start with another kind raises `ArgumentError`. Every solver also writes
  `extra["terms"]` (per-term losses of the evaluation `State.loss` comes from; `TorchOptim` adds `extra["info"]`,
  the dicts of terms returning `(value, info)`), kept as device tensors: no host sync.
  `tests/optim/test_solver_state.py` (`test_solvers_write_diagnostics_and_run_stacks_them`,
  `test_warm_started_lbfgs_keeps_its_memory`, `test_warm_start_from_another_solver_kind_names_both_kinds`,
  `test_term_info_and_per_term_losses_come_from_the_step_evaluation`).
- A `time=` problem that mixes windowed terms with an unwindowed term reading a time variable warns and
  goes dense. Normal equations are built through a sink (`add`/`finish`); a new structure is a new sink.
  `tests/optim/test_solver_state.py::test_time_problem_without_window_warns_on_dense_fallback`,
  `tests/optim/test_problem.py::test_block_tridiagonal_path_equals_dense`, `[review]`.
- Registration (`residual`, `scalar`, `cached`) only adds names to the problem it is called on; every field of
  a registered term is fixed; changes go through `with_`, which returns a new problem and leaves its parent as
  it was. The problems `with_` returns and the views `run`, `solve_constrained` and `solve_layer` make are
  sealed: registration on them raises `ArgumentError`, so a callback cannot change a solve.
  `tests/optim/test_problem.py::test_problem_is_immutable`,
  `tests/optim/test_problem.py::test_callbacks_cannot_register_on_the_views_run_hands_them`.
- Linear solves run in float64 with unit-diagonal scaling whatever the problem dtype. `[review]`.
- Every class here is abstract or `@final`. `tests/contract/test_abstract_or_final.py`.
- Every entry that takes `x` (`run`, hence `solve`; the `Problem` evaluation methods; `check_jacobian`) checks it
  once with `evaluation.checked_values`: one `(B, *event)` tensor per variable, so `x` never overrides data.
  `Phase.on_start` must return `(x, data)`.
  `tests/optim/test_problem.py::test_x_holds_one_batch_tensor_per_variable_and_never_overrides_data`,
  `tests/optim/test_problem.py::test_on_start_must_return_x_and_data`.
- A private name is used only by the module that defines it: `evaluation`, `linearization`, `phases` and
  `constrained` read a `Problem` through its public fields (`cached_values`, `manifolds`, `slices()`, ...).
  The band layout is memoised by `functools.lru_cache` on the problem's structure (C-STATE-4).
  `tests/optim/test_problem.py::test_each_private_name_is_used_only_by_the_module_that_defines_it`.
- Every error is a class of the error tree in `better_robot.exceptions` (record 0020), never a builtin
  `ValueError` and never `NotImplementedError`. `tests/contract/test_error_tree.py`,
  `tests/contract/test_roadmap_stub_inventory.py`.

## Checks

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest tests/optim tests/terms -q
uv run python examples/01_ik.py   # 02..05 likewise
```

A new Jacobian route or linear-solver path must be compared against `Problem(dense=True)`
and row-by-row autodiff in `tests/optim/test_problem.py`.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
