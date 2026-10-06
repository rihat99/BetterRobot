# Adversarial review of optim2 + terms (clean-context mars_aux, 2026-10-07 ~02:15)

The reviewer had no write tool; this note records its findings and what was done. Its evidence
script was in the session scratchpad (`review/exp1.py`, float64, CPU).

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | critical | `solve_layer` derivative wrong with any scalar term (H has no scalar curvature: 1e9 instead of 1; 1.0 instead of 0.25) | fixed: `solve_layer` raises when scalar terms are active; test added |
| 2 | major | tensor weight `(B,)` broadcasts along the last event axis silently | fixed: weight must have the output's shape with or without the batch axis, else `ValueError`; test added. A `(B,)` weight on a `(B, B)` output is still ambiguous |
| 3 | major | `with_` shared `terms`, `_cached` with the base problem; registering on a view changed the base | fixed: dicts copied; test added |
| 4 | minor | float32 LM stops only on gradient < 1e-7; can run all iterations | partly: LM now also stops when damping reaches its maximum. A relative gradient test is not added |
| 5 | minor | a `window` term that reads outside its window corrupts H silently | documented in `Problem.residual`; `check_jacobian` detects it |
| 6 | minor | `TorchOptim` state aliased the parameter dict | fixed: `State.x` is a detached copy |
| 6b | minor | scalar terms under LM/GN have no curvature, damping acts as a step size | by design; documented in `Problem.scalar` |

Found sound by the reviewer: IRLS consistency of loss, g and H; coloured autodiff (including fewer
output rows); `BandSink`/`BandLayout` with two time variables of widths 2 and 3, T=7, group 2
(H, g, solve, predicted_decrease equal dense to 1e-14); LM acceptance, damping, frozen done
elements, delta=0 after failed factorization; L-BFGS recursion, Armijo, history drop; rejection
of windowed autodiff terms that read non-time variables.

Not reviewed: `manifold.py`, `terms/robot.py`, `terms/ik.py`, `solve_layer` with nq != nv,
mixed devices or dtypes, `TorchOptim` free masks at run time.

Found by the author while running the examples (after the review):
- `solve_layer` moved an unconverged solution by a full Gauss-Newton step (16 rad on a Panda
  sitting on its joint limits). Fixed: the value is exactly the solver's result, only the
  derivative comes from the Newton step; `damping=1e-6` added for unconstrained directions.
