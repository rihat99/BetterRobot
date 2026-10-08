# w1 slice B `br_optim` — done (items 1-5)

## Changes (file: change)
- `src/better_robot/optim/solvers.py`: `DIAGNOSTICS = ("grad_norm","step_norm","accepted","damping")`; the key set
  is documented once in the `State` docstring. LM, GaussNewton, LBFGS write `grad_norm`, `step_norm` (0 where no
  step was taken), `accepted` (bool) into `State.extra` each step; LM also keeps `damping`. `TorchOptim` writes none.
  Class attr `warm_keys`: LM `("damping","growth")`, LBFGS `("s","y","rho")`. New private `_projected(g, bounds)`.
  L-BFGS fix: `done`/`grad_norm` now from `g_new` at the new point (projected with the new point's bounds), not
  the old `g`.
- `src/better_robot/optim/phases.py`: `Result.diagnostics: {phase: {key: (steps, B)}}` and `Result.state`
  (final solver state of the last phase). `run(..., state=None)` / `solve(..., state=None)`: warm start keeps
  `warm_keys` of `state.extra`, init recomputes the rest; start point is `x`, else `state.x`, else variables.
  `solve_layer` returns `replace(result, x=attached)` (keeps diagnostics/state).
- `src/better_robot/optim/problem.py`: `warnings.warn(UserWarning)` in `_time_layout` when windowed terms are
  mixed with an unwindowed term reading a time variable (names the terms; `dense=True` silences it). "Once"
  relies on Python's default once-per-call-site filter (no per-Problem flag).
- `src/better_robot/optim/constrained.py`: `outer < 1` -> `ValueError`.

## Deviation to review
- Brief said "`history` exposes them". I put diagnostics in a separate `Result.diagnostics` instead of new keys
  in `history`, because `examples/03_clip_fit_phases.py:72`, `docs/guides/fit-a-clip.md:117-122` and
  `docs/concepts/time-and-structure.md:127` iterate `history.items()` as phase -> losses. Easy to move if wanted.

## Tests added
- `tests/optim/test_solver_state.py` (8 tests): diagnostics keys/shapes per solver (LM, GN, LBFGS); warm-started
  LM (damping=1e3) takes fewer iterations than cold on a shifted Rosenbrock and reaches the same x; L-BFGS warm
  start keeps nonzero `rho`; `pytest.warns` on dense fallback + no warning with `dense=True`; L-BFGS done from new
  gradient; `solve_constrained(outer=0)` raises.

## Docs touched
- `docs/contracts/08-extension-points.md`: new section "Optimization: normal equations are assembled through a
  sink (SHOULD)" (no C- rule ID, so no index/enforcer row; inserted before agent D's collision section).
- `docs/concepts/optimization.md`: diagnostics keys, `state=` warm start, `Result.diagnostics`; new "Reserved
  vocabulary" section (equality/inequality reserved, C-TENS-5 batch-axis link, dense-fallback warning).
- `src/better_robot/optim/CLAUDE.md`: invariants for diagnostics, warm_keys, dense warning, sink seam.

## Gates (full worktree, other slices in flight)
- pytest: `1628 passed, 12 skipped, 7 deselected, 33 warnings in 196.95s`
- ruff: `Found 87 errors` worktree-wide (82 baseline; the 5 extra are in other slices — `ruff check src/better_robot/optim tests/optim` is clean); `mkdocs build --strict`: built, no warnings.

## Left / call sites
- `TorchOptim` writes no diagnostics (documented). No BH call sites known to break (additive API only).
