# Plan — downstream-driven optimization plumbing (BVR round)

State at planning time (2026-07-22, branch `dev`, 2230063): the simplify
round and the TorchOptimizer single-forward fix are delivered and verified;
full CPU gate 1,534 passed; docs strict-build clean. The previous plan
(simplify / model redesign / test prune / Warp FK+RNEA) is fully executed;
its files were deleted with this commit and its durable outcomes live in
`docs/CHANGELOG.md`, `docs/concepts/the_compute_seam.md`, and
`docs/concepts/design_decisions.md`. Recover the old orders via
`git log -- plan/`.

## Why this round

BetterVideoReconstruction (BVR) fitted SMPL-X bodies to video through BR's
optimizer and had to fork every mesh-reading residual plus ~900 lines of
plumbing (`BetterVideoReconstruction/tools/human_optim/`). Their wishlist is
`BetterVideoReconstruction/BR_OPTIMIZER_WISHLIST.md`. Two independent
audits verified every claim against both codebases, and the drafted plan
survived an adversarial review (20 findings, all resolved or consciously
rejected below) before being committed:

- The wishlist's **math is exact**: BR's objective is
  `Σ_groups ρ(‖weight·rows‖²)` with the term weight inside the kernel
  argument (`problem.py:344-356`), which forces the `α = √(2w/N)` algebra
  BVR re-derives in every robust residual, plus a `√(2·loss)` fake row for
  scalar penalties.
- The wishlist's **savings are overstated** (its ~250-line item-1 figure is
  a class-size count; most of that body is domain math that stays in BVR)
  and two of its nine asks already exist: `residual.weight = 0` is a clean
  one-call disable, and static `Variable` inputs are already updatable
  mid-solve via `problem.update()`.
- The wishlist **missed real frictions** the audits found: robust kernels
  NaN on zero-filled rows, `√weight` tricks to keep confidence linear
  through the squaring, a re-derived scene-SDF with point-to-*plane*
  semantics BR lacks, `ProjectionResidual`'s inability to project anything
  but frame rows, and three optimizer-lifecycle semantics (static updates
  discarding Adam moments, sticky terminal statuses, no resume) that make
  any multi-phase loop hand-rolled today.

The theme of this round is the same as the last one: BR carries the generic
plumbing (weight algebra, activity bookkeeping, composition, scalar terms,
tangent groups, phase lifecycle), downstreams keep their domain math.
Nothing here adds a solver, a protocol, or a configuration system.

## Decisions taken in this plan (owner may veto before launch)

1. **The term weight moves outside the kernel.** Per residual the
   objective becomes
   `Σ_k active_k · w_k · ρ(‖row_weight · rows_k‖²) · norm` — `weight` is a
   plain non-negative outer coefficient (scalar, batch-shaped, or
   per-group), the kernel keeps its own scale,
   `reduce ∈ {"sum","mean","mean_active"}` sets `norm`, the activity mask
   is authoritative and detached, and the old √-information inner
   multiplier survives as the separate `row_weight`. Breaking change;
   every in-tree caller migrates in the same order (the dominant breakage
   is every scalar weight — its L2 meaning flips from `a²` to `a`). The
   hard-wired `0.5` in every `ρ` stays. All objective consumers —
   `problem.py`, `lm.py`, `temporal.py`, `implicit.py` — route through one
   shared per-evaluation bundle so the formula exists in exactly one
   place. LM stays uncorrected IRLS (no Triggs), gradient-consistent on a
   fixed active set; `mean_active`/masked problems are
   implicit-differentiation-ineligible with an actionable error.
2. **Activity masks are fixed-shape and captured in-scope.** A gated
   residual zeroes its inactive rows and reports a boolean per-group mask;
   masks, coefficients, and rows are bundled inside the evaluation scope
   that produced them (LM's candidate evaluations close their scope before
   linearization — re-querying is a wrong-iterate bug). Zero-row
   discipline is enforced by contract tests, never by runtime value
   checks; the LM inner loop stays fixed-shape and sync-free.
3. **Nodes compose.** A `Node` input may be another `Node`; `Problem`
   harvests nodes and leaf variables transitively and scopes every
   discovered node's memo. Updatable inputs are explicitly constructed
   static `Variable`s — `Variable` learns to hold bool/int data when
   `trainable=False` (masks, labels), and `Problem`'s dtype validation
   relaxes accordingly. No auto-wrapping (the residuals layer cannot
   construct `optim.Variable` without reversing the DAG); raw-tensor
   inputs stay frozen-at-construction and say so in their docstrings.
4. **No phase/curriculum object — but the lifecycle gets honest.** A Phase
   abstraction would be config as code; we ship a how-to guide instead.
   What makes the plain loop actually work is order 05's three fixes:
   same-layout `problem.update()` preserves optimizer moments, `resume()`
   returns terminal elements to RUNNING, and `TorchOptimizer` takes an LR
   scheduler factory. (Explicitly rejects wishlist item 5.1.)
5. **Scalar penalties are one library class**, `ScalarCost`, an exact
   safe-sqrt `√(2f)` row (double-`where`, no epsilon, no dead zone) whose
   objective contribution is exactly `w·f` for the documented domain
   `f ≥ 0` — zero changes to `Problem` or LM. Honest caveats are part of
   the contract: zero gradient at exactly `f = 0`, ill-conditioned GN
   columns as `f → 0`, and no implicit differentiation.
6. **Tangent-group freeze returns, smaller.** The old `mask`/`scale`
   subsystem (deleted at `1d124b7` for having zero callers) now has a real
   caller. It returns as a construction-time `frozen_groups` mask on
   `RobotVariable` only — named groups derived from the model (per joint,
   `"root"`, `"root_lin"`, `"root_ang"`, `"joints"`), no `scale`, no
   mutable mask, no raw-mask parameter. Public geometry (`difference()`,
   the `Difference` residual) is untouched; the free/full seam lives in
   optimizer-facing plumbing, analytic Jacobian columns are reduced
   centrally in `Problem`, and the LM bounds layout, temporal width, and
   implicit solution coordinates follow. Changing the frozen set means
   constructing a new variable/problem — exactly the phase pattern.
7. **Higher-order smoothness without gratuitous breakage.**
   `SmoothnessResidual(order ∈ {2,3,4})` (forward k-th differences of the
   tangent first-difference sequence, per-coordinate `coordinate_weight`)
   replaces `AccelerationResidual` (whose rows `order=2` reproduces
   bit-for-bit); **`VelocityResidual` stays unchanged** — no caller wants
   a forward order-1 and the wishlist asks only for orders 2–4. Analytic /
   banded blocks are declared only for models whose tangent difference is
   affine (all scalar joints); spherical/free-flyer models warn and take
   the AD/dense route — and order 04 first *verifies* the suspected
   pre-existing defect that today's constant-identity blocks are wrong for
   those models.
8. **Not doing, on purpose:** a live (variable-tracking) kernel scale —
   rejected outright, BVR's pre-fit keeps computing its scale inside its
   own `ScalarCost` function, which is the honest place for it; auto-skip
   of short-horizon smoothness (constructors keep raising; callers
   guard); a per-term metrics framework (`problem.term_costs()` is a
   readout, not a logging system); a Problem-level scalar-cost pathway;
   state-dependent manifold smoothness Jacobians (deferred to the roadmap
   with its own contract).

## The five orders

| Order | What | Acceptance in one line |
|---|---|---|
| `for_agents/01_objective_algebra.md` | Outer weight + `reduce` + activity bundle + `enabled` + `row_weight`; kernel zero-row gradient safety; `term_costs()`; temporal/implicit routed through one formula; exhaustive weight migration | Algebra tests + gradcheck-at-zero pass; banded/implicit match dense; every in-tree caller migrated; full gate green |
| `for_agents/02_node_composition.md` | Node-in-node inputs, transitive harvest and scoping, bool/int static Variables, `ScalarCost` | Nested-node counting + stale-memo tests fail closed; `w·f` exact; gate green |
| `for_agents/03_mesh_residuals.md` | Scene/chamfer accept node clouds and masks; per-head gates + external masks + linear detached confidence; point-to-plane option; `PointProjectionResidual` with explicit event axes | The BVR scene/chamfer/reproj shapes are expressible without subclassing; defaults regression-pinned; gate green |
| `for_agents/04_tangent_groups_smoothness.md` | Named tangent groups (incl. root_lin/root_ang), construction-time freeze wired through problem/LM-bounds/temporal/implicit, `SmoothnessResidual` orders 2–4 | Frozen-root warm-up end to end; banded trajopt routes frozen; spherical-block defect verified and gated; gate green |
| `for_agents/05_optimizer_docs_slice.md` | Layout-aware refresh, `resume()`, LR scheduler; staged-fit how-to; BVR-shaped vertical slice v2; docs/roadmap/changelog sync | Moments survive static swaps; slice exercises every order in < ~150 lines and converges; strict Sphinx green |

Run them in order: 02 depends on 01's weight semantics only trivially, but
03 needs both; 04 is independent of 02–03 in content yet lands after them
to keep merge noise down; 05 assembles everything. Each order ends with the
full gate green.

## Soft budgets

Feature rounds grow code; budgets keep it honest. Net `src/**/*.py` growth
per order: 01 ≤ +300, 02 ≤ +150, 03 ≤ +250, 04 ≤ +350, 05 ≤ +150 (docs and
tests excluded). The two largest budgets reflect deliberately inventoried
wiring (01: temporal + implicit routing; 04: five-file freeze seam), not
license to sprawl. An order that cannot meet its budget stops and reports
rather than redefining success.

## Ground rules

1. **Read `DESIGN_RULES.md` (repo root, untracked) before writing code.**
2. **No backward compatibility.** Removed or renamed public symbols get a
   row in root `MIGRATION.md`; never an alias, shim, or deprecation path.
3. **The safety net stays green.** Pinocchio parity, contract tests, and
   the full non-bench/non-CUDA gate pass at the end of every order; never
   weaken them to land a change. CUDA is agent-runnable on this host (8×
   RTX 6000 Ada; pin with `CUDA_VISIBLE_DEVICES`, 2/3/7 usually free) —
   run the CUDA suite when an order touches evaluation paths.
4. **No silent behavior.** Fallbacks warn (`AutodiffFallbackWarning`
   policy), infeasible constructions raise, and any numeric-semantics
   change (order 01's algebra, order 03's confidence linearity) is
   documented in the same order that lands it.
5. **Docs stay true per order** — every page, docstring, and `CLAUDE.md`
   an order falsifies is updated in the same order, including
   `optim/CLAUDE.md` and `residuals/CLAUDE.md` contract lines.
6. **Honest results files.** Each order writes `for_agents/NN_results.md`:
   delivered, deviations (findings, not failures), exact test counts, line
   accounting by the `wc -l` method. Order 01's results file additionally
   carries the full old→new weight table per call site; order 05's closes
   the round with a per-wishlist-item disposition table.
7. **Subagents run on Opus.** Codex is available for adversarial review;
   treat its taste for abstraction with suspicion, its factual findings
   with respect.
8. **Everything on `dev`.** No feature branches, no worktrees unless asked.
9. **BVR is evidence, not a spec.** When this plan and BVR's fork disagree,
   design the generic thing and note the divergence in the results file;
   do not import BVR's factor-2 conventions, its `1e-12` clamps, or its
   sign conventions without deciding them consciously.
