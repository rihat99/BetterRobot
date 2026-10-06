# Fact-check of external-library claims in study_optim_design.md (mars_aux, web, 2026-10-06)

Corrections to the study's [U] claims:
- jaxls: decorator is `@jaxls.Cost.factory` (not `create_factory`). Linear solvers: sparse CG
  (default), dense Cholesky, CHOLMOD; no sparse QR. Constraints confirmed:
  `kind="constraint_eq_zero" | "constraint_geq_zero" | "constraint_leq_zero"`, augmented
  Lagrangian. Analytic override `jac_custom_fn`; also `jac_custom_with_cache_fn` (residual
  returns `(r, cache)` reused by the Jacobian). Docs: https://brentyi.github.io/jaxls
- optimistix: LM default linear solver is `lineax.QR()` (no normal equations by default).
  `least_squares(..., adjoint=ImplicitAdjoint())` confirmed. v0.1.0 2026-02-16.
- PyPose 0.9.5 (2026-04-12): LM has a built-in `sparse=True` path. One damping value for the
  whole batch (inferred from `modjac`).
- Theseus: last release 0.2.3 on 2025-02-10, near-dormant. `AutogradMode.VMAP` = vmap(jacrev).
- torchmin 0.1.0 (2026-01-23): single-problem, no manifolds. torchopt: stale since 2023-11.
- torch.optim.LBFGS: single coupled problem, host syncs in `step()` (confirmed from source).
  Only maintained batched L-BFGS found: inside cuRobo (not standalone).
- torch.func limits confirmed from docs: no data-dependent control flow on vmapped tensors, no
  `.item()`, no in-place write of batched data into unbatched tensors.
- No serious new torch-native batched NLLS library with manifolds found.

Reviewer's objections to the design:
1. "Existing loss code drops in unchanged" is false for autodiff until the kinematics code is
   vmap/jacfwd-clean; autodiff through FK per term is slow (Theseus users' main complaint).
   Analytic frame Jacobians must be the main path for kinematic terms.
2. Normal equations + Cholesky square the condition number; risky in float32. Needs a policy
   (column scaling, damping floor, float64 linear solve, or QR option).
3. Arrowhead/Schur for global variables should be prototyped before it is promised.
