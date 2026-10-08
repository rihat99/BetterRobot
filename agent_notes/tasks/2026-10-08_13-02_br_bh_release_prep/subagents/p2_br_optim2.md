# P2 sweep — slice `br_optim`, second half

Scope: BR `optim/{problem,linear,kernels}.py`, `terms/*`, `collision/*`, `examples/*.py` (the first half,
`optim/{manifold,solvers,phases,constrained}.py`, is in `p2_br_optim.md`). Budget 50k tokens and load average
600-800, so this sweep is **partial**: read where the risk is highest, probe numerically, list the rest as not read.

## Files read
- `optim/problem.py`: def map; l. 192-203 (`residual`), 282-371 (`linearize`, `gradient`, `retract`, `project`,
  `step_bounds`), 521-548 (`_analytic_pieces`). Grep of every `[0]`/`.shape[0]`/`.item`/`float(`/`detach`/loop site.
- `optim/kernels.py`: def map + numeric probe of every kernel (below). `optim/linear.py`: def map only.
- `terms/robot.py`: l. 20-146 (all of `RobotManifold`, `frame_pose`, `frame_position`, `joint_limits`, `rest`,
  `finite_difference`). `terms/collision.py` l. 19-37 (`sphere_clearance`). `terms/ik.py` l. 51 only.
- `collision/distance.py`: all function bodies (grep + l. 64-84 `sphere_sdf`). `collision/robot_spheres.py`: def map.
- Not read: `optim/__init__.py`, `optim/linear.py` bodies (Normal/Banded/Arrow solves, sinks), `problem.py`
  l. 1-191, 372-520, 549-650 (`_windowed_pieces`, `_band_layout`, `check_jacobian`), `terms/ik.py` rest,
  `terms/__init__.py`, `collision/robot_spheres.py` bodies, `examples/*.py` (run headless by the tests only).

## Fixed (what, where, test)
1. **Scalar term that does not depend on the variables crashed `linearize`.** A `problem.scalar(...)` that reads
   only data (or a cached value independent of the variables) made `torch.autograd.grad(scalar_total.sum(), delta)`
   raise `RuntimeError: element 0 of tensors does not require grad`. `gradient()` and `loss()` were fine, so LM/GN
   failed where LBFGS/Adam worked. Fix in `optim/problem.py` `linearize`: guard `scalar_total.requires_grad` and pass
   `allow_unused=True, materialize_grads=True` (the second covers `solve_layer`, where data requires grad but
   `delta` is unused). Test: `tests/optim/test_problem.py::test_scalar_term_that_reads_only_data_linearizes`
   (compares `linearize` loss and `g` with `gradient()`; the probe raised before the fix).
2. **Empty batch (B=0) with `RobotManifold` crashed** (same bug the first agent fixed in `optim/manifold.py`):
   `RobotManifold.tangent_dim` used `x[0].numel()` -> `IndexError`. Fix: `x.shape[1:].numel()` in
   `terms/robot.py`. With that fixed, the next failure was in `Problem._analytic_pieces`:
   `block.reshape(batch, 1, -1, width)` is ambiguous for 0 elements. Fix: explicit row count
   `weighted.shape[1:].numel()` in `optim/problem.py`. Test:
   `tests/optim/test_robot_terms.py::test_robot_manifold_accepts_an_empty_batch` (Panda, `rest` term with an
   analytic jac, LM 2 iters; verified it fails with `IndexError` when the `tangent_dim` fix is reverted).

## Reported, not changed (ranked; severity; evidence; suggested fix)
1. **Info — robust kernels are consistent.** For L2, Huber, Cauchy, Tukey, GemanMcClure, `weight(s)` equals
   `2 * d rho / d s` from autograd to <= 1.1e-16 on s in [0.01, 30], and `rho(0) = 0`. So IRLS in `linearize`
   gives the exact gradient of `Problem.loss`.
2. **Info — analytic Jacobians checked by reading:** `frame_pose` jac = `Jr^-1(err) @ J_local` (right
   perturbation, consistent with `integrate`); `frame_position` uses the default `LOCAL_WORLD_ALIGNED` linear rows;
   `rest`/`joint_limits` diagonal jac only when `nq == nv`, autodiff otherwise. `sphere_sdf` trilinear weights and
   upper-boundary clamp (`lower <= size-2`, weight 1) are correct; `_norm` has a finite gradient at zero.
3. **Low — `terms/ik.py` l. 51** draws restart samples with `torch.rand(..., generator=...)` on CPU and then
   `.to(device)`: one host-to-device copy per call, presumably deliberate (CPU generator reproducibility). Not a bug.

## Performance notes (measured)
None measured: load average 610-785 during the run (`uptime`), above the 100 threshold, so micro-benchmarks were
skipped. From reading: `_analytic_pieces` loops over blocks and window offsets only (no batch loop);
`sphere_sdf` loops over the 8 corners (fixed). No `.item()`/`.cpu()` on the hot path in the parts read; the one
documented sync is the `allclose` check in `_windowed_pieces` (optim/CLAUDE.md lists it as the exception).

## Gate results
- `uv run ruff check src/better_robot/optim src/better_robot/terms tests/optim`: all checks passed.
- New pinning tests: both pass with the fixes (and fail without them, verified by probe / revert).
- Slice tests: see the tails appended below (logs in the scratchpad: `p2_optim2_tests.log` = optim + collision +
  examples, started before the edits, so its in-process optim tests ran the old code while the example
  subprocesses picked up whatever was on disk; `p2_optim2_tests_final.log` = optim + collision on the final code).
- Not run: full suite, mkdocs (no docs-referenced docstrings changed).

### Test-log status at hand-back
Both slice runs were **still running** when this agent's budget ran out (load average ~785): no failures so far
(25 and 26 dots, no `F`/`E`). Each log ends with `EXIT <code>` when done; the orchestrator must read both tails:
`/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/p2_optim2_tests_final.log`
and `.../scratchpad/p2_optim2_tests.log` (the second one includes the 8 headless examples).

Update after hand-back: `p2_optim2_tests.log` finished with 43 passed, 1 warning, EXIT 0 (883 s); the warning comes from a test that calls float() on a tensor with grad. `p2_optim2_tests_final.log` was still running.

Final-code run (optim + collision) finished:
    37 passed, 1 warning in 756.52s (0:12:36)
    EXIT 0
