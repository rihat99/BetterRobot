# M0 Results — Truth and Correctness

**Date:** 2026-07-17

**Branch:** `dev`

**Status:** Complete and committed on `dev`.

## Delivered

| Task | Result |
|---|---|
| T0.1–T0.2 | Added safe-`where` dummy substitution throughout SO(3)/SE(3) exp/log and right-Jacobian singular paths. Cutoffs are dtype-aware (`1e-5` fp32, `1e-8` fp64), with first- and second-order identity tests plus the SMPL/rest and spherical-joint regressions. |
| T0.3 | Applied the owner-selected identity exemption. Non-identity mimic parameters fail fast with M3 guidance; exact identity tags remain loadable, preserve metadata, and are explicitly documented as independent coordinates rather than enforced mimics. Stock Panda remains loadable. |
| T0.4 | Batched `solve_ik` now raises an actionable M2b error. `SolverState` has a second guard for other batched residual paths. |
| T0.5 | Bounded LM is documented and regression-tested as projection-before-evaluation with no active-set, projected-gradient, or KKT handling; bounds-limited failure truthfully ends as `maxiter`. |
| T0.6 | LM acceptance now uses `sum(kernel.rho(r²))`; its public raw `residual_norm` remains `0.5 * ||r||²`. Kernel protocol and consistency tests cover `rho`. |
| T0.7 | `solve_ik` preserves the working dtype and casts limits to the input device/dtype without forced fp32 conversion. |
| T0.8 | Removed the nonfunctional `AUTODIFF`/`FUNCTIONAL`, `cg`, and `trust_region` selections. AUTO/FD now document the unbatched `2 * nv + 1` evaluation cost. Consumer greps found no use of the removed selections. |
| T0.9 | Corrected code/docs/CLAUDE claims, removed ignored collision IK arguments, added an exact executable front-page example, and added an AST-checked complete file inventory for explicit `NotImplementedError` raises. |
| T0.10 | Removed the quaternion tensor-to-Python check from raw FK. Public FK exposes an opt-in `check_quaternion_norm` debug check; free-flyer raw FK is covered by `torch.compile(fullgraph=True)` checks on CPU and CUDA. |

## Verification

- `UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q`:
  **940 passed, 1 skipped, 21 warnings** in 62.94 s. This includes the
  unchanged Pinocchio-parity suite.
- After routing torch through the official CUDA 12.6 index, the regenerated
  lock selected `torch 2.13.0+cu126`. A GPU-visible locked run on an RTX 6000
  Ada (`CUDA_VISIBLE_DEVICES=1 uv run --locked pytest tests/ -q`) completed
  with **941 passed, 0 skipped, 19 warnings** in 81.81 s.
- Targeted CUDA checks passed: tensor/backward smoke; 12 Lie identity
  gradchecks and 12 gradgradchecks; fullgraph free-flyer FK matching eager;
  and Panda IK convergence in three iterations in both fp32 and fp64.
- `.venv/bin/sphinx-build -q -b dummy docs docs/_build/dummy`: succeeded.
  Four warnings were only unreachable external intersphinx inventories in
  the network-restricted environment.
- `git diff --check`: clean.
- Ruff passes for every new standalone M0 test module. A sweep over every
  changed Python file still reports 57 largely pre-existing style findings
  in legacy files (mainly complexity, local imports, and unused imports);
  M0 did not broaden into unrelated cleanup.
- Consumer search in `BetterVideoReconstruction` and `BetterHumanForce`
  found no BetterRobot callers of the removed Jacobian/config choices or
  ignored collision arguments. The only `solve_ik` matches were third-party
  `pymomentum` calls.

## Deviations and important findings

1. The plan's requested zero gradient for `so3.exp(w).sum()` at `w=0` is
   mathematically wrong. Its derivative is `[0.5, 0.5, 0.5]`; the regression
   locks that finite, correct value while all requested gradchecks and
   gradgradchecks pass.
2. Per owner decision, T0.3 intentionally deviates from blanket mimic
   rejection. Identity tags are exempt for Panda compatibility. They are
   **not coupled**, so code assuming mimic enforcement can still produce
   wrong gripper kinematics until M3.
3. Built-in robust kernels use the normalized convention
   `weight = 2 * rho'`, not the plan's literal `weight = rho'`. The protocol
   also lacked `rho()` even though robust acceptance needs it; both code and
   docs now use the actual convention.
4. The plan's global claim that `stalled` was never emitted was incorrect:
   LBFGS emits it. LM and GN specifically do not; the bounded-LM result and
   docs retain truthful `maxiter` behavior.
5. T0.10 uses an explicit per-call public debug flag rather than a mutable
   module-global flag. This keeps normal FK compile-friendly and makes sync
   cost visible at the call site.
6. The front-page example had two additional failures beyond the stale frame
   name: an unresolved `panda.urdf` path and an undefined target. The tested
   example now resolves `robot_descriptions`' Panda URDF and constructs a
   reachable target before solving.
7. The initial CUDA failure was environmental, not a BetterRobot failure:
   driver 560.35 supports CUDA 12.6, while the PyPI wheel was
   `torch 2.11.0+cu130` and CUDA 13 requires driver 580 or newer. Switching
   to the locked `torch 2.13.0+cu126` build enabled the CUDA validation above.
   The project now routes torch through an explicit CUDA 12.6 uv source so
   lock regeneration and future syncs preserve the compatible build.

The pre-existing `.gitignore` change adding `references` was preserved and
not modified as part of M0.
