# Order 01 — Objective algebra: outer weight, reduction, activity, safety

Read `plan/README.md` and `DESIGN_RULES.md` first. This order changes the
meaning of `Residual.weight` across the library. It is the mathematical
heart of the round; everything else builds on it.

## Motivation (verified)

BR's objective today is `Σ_groups ρ(‖weight·rows‖²)` with the term weight
inside the kernel argument (`optim/problem.py:344-356`). Term importance
and kernel outlier scale are therefore coupled, and there is no reduction,
so a downstream that wants the obvious `w · mean_rows[GM(d²; s²)]` must
hand-derive `α = √(2w/N)`, set `weight = α/s`, and set the kernel scale to
`α` — re-deriving it per residual and getting silent factor-2
inconsistencies between robust and plain terms (verified in BVR's
`tools/human_optim/residuals.py:134-186`). Separately, robust kernels are
not gradient-safe on exact-zero rows (Huber's `sqrt` under `torch.where`
NaNs the backward), which downstream code works around by dropping kernels.

## Target semantics

Per residual, with `rows = row_weight.apply(error())` reshaped into
`group_size` groups `k`:

```text
cost = Σ_k  active_k · w_k · ρ(‖rows_k‖²) · norm
```

- `weight` (`w_k`) is a plain outer coefficient and must be
  **non-negative**: a Python scalar (negative raises at assignment), or a
  tensor whose shape is exactly `()`, the trainable batch shape, or
  `(*batch_shape, n_groups)` — nothing else, no broadcasting guesses.
  Tensor non-negativity is a documented domain contract, not a runtime
  value scan (the validation policy forbids scanning user tensor values in
  hot paths). Python `0.0` keeps today's `is_inactive()` skip semantics;
  tensor zero stays graph-visible. No `Weight` class is involved.
- `row_weight` is the old √-information inner multiplier (`ScaleWeight` /
  `DiagonalWeight` / raw tensor coerced as today), default identity. It is
  for information whitening (per-pixel σ, per-row units), not importance.
- `kernel` is unchanged and keeps its own scale. The hard-wired `0.5` in
  every `ρ` stays; document prominently that the L2 objective is
  `0.5 · w · Σ_k ‖rows_k‖² · norm`.
- `reduce="sum"` (default, `norm = 1`) | `"mean"` (`norm = 1/n_groups`,
  static) | `"mean_active"` (`norm = 1/clamp(Σ_k active_k, 1)` per batch
  element, **detached**).
- `active_groups()` is a new optional `Residual` hook returning a boolean
  `(..., n_groups)` mask (default: all active). The mask is
  **authoritative**: it multiplies the cost term tensorically (see the
  `active_k` factor above — required because `ρ(0) ≠ 0` is legal for a
  custom kernel, see `tests/optim/test_problem_robust_gradient.py:21-30`)
  and zeroes the IRLS row scale. Shipped gated residuals still zero their
  inactive rows in `error()` as a convention, enforced by focused contract
  tests only — never by a runtime value check (the LM inner loop stays
  fixed-shape and sync-free).
- `enabled: bool = True` on `Residual`; every skip site
  (`_error_current`, `_objective_current`, `jacobian_blocks`) treats
  `not enabled` exactly like a Python-zero weight. Toggling `enabled` does
  not bump `_update_serial` (optimizer state survives, layout is
  unchanged — row slots are reserved at freeze as today).

**The evaluation bundle.** Activity, per-group coefficients, and rows must
be captured *inside* the same evaluation scope that produced them: LM
evaluates candidates via `_error_at`, whose scope closes before
`_robustify` runs (`problem.py:340-342`, `lm.py:310-312,640-644`), so
re-querying `active_groups()` afterwards would read the wrong iterate, and
zero rows cannot encode activity (a genuinely active row may be exactly
zero). Introduce one private per-evaluation structure — rows, active
masks, effective per-group coefficients `w_k · norm`, per-term costs —
built while the scope is open and holding **no node memos**. `objective`,
`error`, `term_costs`, LM's `_robustify`/`_robust_decrease`, structured
assembly, and implicit reconstruction all consume it; the exact objective
formula then lives in exactly one place.

Linearization: LM/GN's IRLS row scale for group `k` becomes
`√(active_k · w_k · norm · kernel.weight(‖rows_k‖²))`. Be honest about
what this is: **unchanged uncorrected IRLS** (no Triggs second-order
correction, same as today), gradient-consistent with the objective only on
a fixed active set; at an activity threshold the detached mask/count makes
the objective non-differentiable, which is accepted and documented. Test
first-order consistency on fixed active sets; test threshold crossings
separately as behavioral (descent still achieved), not derivative, checks.

## Work items

1. `residuals/base.py`: `Residual.__init__` gains `row_weight`, `reduce`,
   keeps `weight` with the new outer meaning (non-negativity validation
   for Python scalars); `enabled` attribute; `active_groups()` hook;
   `weighted_error()` updated or removed (decide against its callers).
   `residuals/temporal.py` (`TimeIndexedResidual`) copies the old
   weight/kernel fields (`temporal.py:51-57`) — it must forward the full
   new set (`weight`, `row_weight`, `reduce`, `kernel`, `enabled`).
2. `optim/problem.py`: the evaluation bundle; `_error_current`,
   `_objective_current`, `jacobian_blocks`, `gradient` honor the new
   algebra and `enabled`; `term_costs()` public method returning
   `dict[str, Tensor(batch)]` from one evaluation using the bundle.
   Public `error()` keeps returning **whitened rows** (`row_weight` only,
   outer factors excluded) — see item 6 for the consequences.
3. `optim/lm.py`: extend the existing kernel row-scale path
   (`_robustify`, applied at `lm.py:320-321`) to consume the bundle with
   `√(active_k · w_k · norm)`; the inner loop stays input-pure,
   fixed-shape, sync-free (`Σ active` is a tensor; never a host branch).
   Known traps: `_robust_decrease` recomputes ρ per group and must gain
   the same outer/active factors, and `_robust_finite` asserts IRLS
   weights `<= 1.0` — that invariant holds only for kernel weights, so
   keep outer factors out of that tensor (or split the check) rather than
   loosening it.
4. **`optim/temporal.py` and `optim/implicit.py` are in scope.** The
   banded assembly calls `item.weight.is_inactive()` and
   `item.weight.apply_jacobian()` (`temporal.py:380-405`) and would crash
   or apply the wrong algebra; `implicit.py:187-213` independently
   reconstructs the kernel objective and would silently ignore outer
   weights, reductions, and activity. Route both through the shared
   formula. Define implicit-differentiation eligibility: a problem with
   `reduce="mean_active"` or a non-default activity mask is **implicit-
   ineligible** (actionable error naming the residual) — the detached
   count has no consistent derivative.
5. `optim/kernels.py`: make every kernel's `rho`/`weight` gradient-safe at
   exact zero and at masked (zero-filled) rows — double-`where` or clamped
   intermediate, whichever reads better. Add gradcheck-at-zero tests
   (float32, per the testing convention; never float64).
6. **Migration — the dominant breakage is every non-unit scalar weight.**
   Today `weight=a` is a row multiplier (L2 coefficient `a²`); afterwards
   it is the coefficient `a` itself. Sweep `src/`, `tests/`, `examples/`
   exhaustively and migrate **by intent, preserving solve behavior**:
   - importance-like scalars become `weight=a²` (or stay `row_weight=a`
     where the value genuinely whitens rows) — includes `tasks/ik.py`
     `IKCostConfig` weights and its weight-stashing refinement path
     (`ik.py:224-252,300-313`), `examples/05_panda_trajopt.py:65-89`;
   - `tasks/contact_forces.py` deletes `_residual_multiplier`
     (`contact_forces.py:186-189`) — its coefficients were already
     pre-square-rooted and now pass through as outer weights directly;
   - semantic tests updated deliberately: `tests/optim/test_problem.py`,
     `test_problem_robust_gradient.py`, `test_residual_base.py`,
     `test_ad_strategies.py`, plus every task-level weight;
   - `solve_ik`/`solve_trajopt` expose `problem.error()` in their results
     (`ik.py:318`, `trajopt.py:180,194`) — document those fields as
     whitened rows, not objective-scaled rows, and add the MIGRATION row.
   The results file carries the full old→new table per call site.
7. Docs in the same order: `residuals/CLAUDE.md`, `optim/CLAUDE.md`,
   `docs/concepts/residuals_costs_and_solvers.md`, and
   `docs/conventions/contracts.md` get the new formula verbatim; the L2
   `0.5`, the detached `mean_active` normalization, the uncorrected-IRLS
   statement, and the implicit-eligibility rule are called out.

## Acceptance

- New focused tests: outer-weight vs kernel-scale independence (a GM term
  with `weight=w, kernel=GemanMcClure(c=s), reduce="mean"` equals
  `w · mean_k[0.5·s²·d²/(s²+d²)]` analytically); `mean_active`
  normalization against a hand computation with a changing mask; a
  `ρ(0) ≠ 0` custom kernel with inactive groups contributes nothing;
  negative Python weight raises; `enabled` round-trip preserving optimizer
  state; `term_costs()` summing to `objective()` exactly.
- LM/GN consistency matrix (weight form × reduce × kernel × active) on
  fixed active sets; a separate threshold-crossing descent test; existing
  LM numeric and Pinocchio parity tests untouched and green.
- Banded/temporal route and implicit differentiation produce the same
  objective as the dense path under the new algebra; the
  implicit-ineligibility error fires for `mean_active`.
- Kernel gradcheck at zero rows passes for all five kernels.
- Full CPU gate green; net src growth ≤ +300.
