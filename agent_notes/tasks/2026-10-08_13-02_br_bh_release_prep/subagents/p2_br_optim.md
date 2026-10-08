# P2 sweep — slice `br_optim`

Scope: BR `src/better_robot/{optim,terms,collision}` + `examples/*.py`. This agent had a 50k-token budget, so the
sweep is **partial**: the solver/phase/layer/constraint core was read and probed; the rest is listed as not read.

## Files read
Read end to end:
- `src/better_robot/optim/solvers.py` (LM, GN, LBFGS, TorchOptim, `select`, tolerances)
- `src/better_robot/optim/phases.py` lines 52-118 (`run`, `solve`, `solve_layer`; `Phase`/`Result` dataclasses skimmed)
- `src/better_robot/optim/constrained.py` (`_shifted`, `solve_constrained`)
- `src/better_robot/optim/manifold.py`
- `tests/optim/test_problem.py` (only the `solve_layer` tests, l. 251-297)

Not read (left for a follow-up sweep): `optim/problem.py` (only its def map and the `.shape[0]` sites),
`optim/linear.py`, `optim/kernels.py`, `optim/__init__.py`, `terms/*.py`, `collision/*.py`, `examples/*.py`
(they are executed headless by `tests/examples/test_examples_run.py`; none of them has a `--check` flag).

## Fixed (what, where, test)
1. **Empty batch (B=0) crashed every solver.** `Euclidean.tangent_dim` used `x[0].numel()`, which raises
   `IndexError` when `B == 0`; `Problem.__init__` calls it, so `Problem({"x": torch.zeros(0, 2)})` failed.
   Fix: `x.shape[1:].numel()` in `src/better_robot/optim/manifold.py`. Same pattern in
   `TorchOptim._mask_gradients` (`gradient[0].numel()` -> `gradient.shape[1:].numel()`, two lines) in
   `src/better_robot/optim/solvers.py`. After the fix LM, GN, LBFGS and TorchOptim(Adam) all return
   `x: (0, 2)`, `loss: (0,)`. Pinning test: `tests/optim/test_problem.py::test_solvers_accept_an_empty_batch`
   (parametrized over the four solvers; it raised `IndexError` before the fix, verified by probe).

## Reported, not changed (ranked; severity; evidence; suggested fix)
1. **Low — LBFGS flags convergence one step late.** `LBFGS.step` sets `done` from `g` (gradient at the
   pre-step `state.x`), not from `g_new` at the accepted point (solvers.py l. 210-211). A converged element runs
   one extra step (up to `line_search`=8 loss evaluations, then a forgotten history). Suggested fix: test
   `g_new` (masked by the bounds at `best_x` the same way as `g`). Changes iteration counts, so left to the owner.
2. **Low — `TorchOptim` docstring vs behaviour.** It says `State.loss` is "the loss before the step". For
   `torch.optim.LBFGS` the closure runs several times per step, so `last["loss"]` is the last line-search
   evaluation, not the pre-step loss. Also `TorchOptim` never sets `done`, so `solve` always runs the full
   `iters` budget; the docstring does not say so. Suggested fix: one docstring sentence each.
3. **Low — `solve_constrained(outer=0)`** reaches `torch.stack([])` / `violations[-1]` and raises a bare
   `RuntimeError`/`IndexError`. Suggested fix: `if outer < 1: raise ValueError(...)` at the top.
4. **Info — `solve_layer` gradients (gradcheck, float64, B=2, n=3).**
   | problem | solver | `torch.autograd.gradcheck` |
   |---|---|---|
   | `[s*x - a, x - b]` (linear in x, non-zero residual: GN H exact), w.r.t. `s, a, b`, `damping=0` | LM | pass |
   | same | GaussNewton | pass |
   | same | LBFGS | **fail** |
   | `x + 0.2*roll(x)^2 - goal` (nonlinear, zero residual), w.r.t. `goal` | LM | pass |
   The LBFGS failure is on the finite-difference side, not the analytic one: `solve_layer` builds the derivative
   from `problem.linearize` regardless of the solver (identical to the LM path that passes), and LBFGS reaches
   1.2e-12 error in 8 iterations, so the numeric derivative of the iterate (eps 1e-6) is limited by the
   solver's stopping point. Not a bug; worth one line in the `solve_layer` docs: check gradients with a
   second-order solver.
5. **Info — `solve_constrained` math checked by reading:** inequality shift is PHR `max(0, c + λ/ρ)`, multiplier
   update `λ = max(0, λ + ρ c)` uses the unclamped `c` (correct), equality update `λ += ρ c` (correct). Raises
   `KeyError` for unknown constraint names (the optim guide only forbids `NotImplementedError`; fine).

Probe script: `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/p2_probe.py`.
Other probe results: float32 LM/LBFGS on a 3x2 fit set `done` for all elements (4 and 3 iterations), dtype stays
float32; warm start via `x=` at the optimum finishes in 1 step with `done=True`.

## Performance notes (measured)
None measured: the machine load average was 570+ during this run, so timings would be meaningless, and the
`perf` slice profiles the LM step. From reading: no Python loop over batch in the solvers; the LBFGS two-loop
recursion loops over `history` (10) only; `run` syncs once per `stop_every` steps (`bool(done.all())`), as
designed; `TorchOptim.step` clones every variable each step (`current = ...clone()`), one small allocation.

## Gate results
- `uv run ruff check src/better_robot/optim tests/optim`: all checks passed.
- Empty-batch probe for the four solvers: passes after the fix (failed with `IndexError` before).
- Slice tests `tests/optim tests/collision` (with the new test): 35 passed, 1 warning, 281.8 s.
- The full suite was **still running when this agent's budget
  ran out** (load average 570+). Full-suite log:
  `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/p2_br_optim_full.log`.
  The orchestrator must read its tail before committing. Not verified here: full suite, repo-wide ruff, mkdocs
  (no docs-referenced docstrings were touched).
- `examples/*.py`: none has `--check`; they run headless through `tests/examples/test_examples_run.py` (part of
  the full suite above). A manual serial run was started but did not finish under the load.
