# M3.5 results — simplify and prune

**Status:** complete on branch `dev` (2026-07-17). The B-spline/trajopt
stack and Model/ModelValues state dedup were not touched; they remain assigned
to M5 and the M4 migration respectively. BHF/BVR and all other repositories
were neither read nor modified, per the owner's stricter repository boundary.

## Outcome

- Removed the dead damping factory, `optim.solve`, `ResidualSpec`/`.spec`,
  inactive Warp backward scaffolding, the CRBA orphan buffer, and three
  zero-caller convenience members.
- Public FK/frame/dynamics boundaries now validate `ModelValues` once; raw
  passes trust prevalidated inputs. A counter contract covers every public
  boundary.
- Shared solver batching/blending and mimic covector reduction are single
  implementations. Warp's active VJP recomputes through the raw FK/frame
  passes instead of a second joint-dispatch implementation.
- One frozen joint-coordinate descriptor now owns box, unit-norm, and q↔v
  mapping semantics. Mimic reduction returns a frozen named result instead of
  a 17-position tuple.
- Removed duplicate/non-asserting tests and added an end-to-end public
  `solve_ik` test on a reduced-coordinate mimic gripper.
- Every removed consumer-facing symbol is recorded in
  `docs/reference/m3_removed_symbols.md`; stale capability claims were fixed.

## Size by task

Counts are exact Git line deltas under `src/` and `tests/` from the M3 close
commit through the M3.5 implementation (documentation outside those trees is
not included).

| Task | `src/` net | `tests/` net | Note |
|---|---:|---:|---|
| T1 damping factory | -10 | -10 | Removed its only factory test. |
| T2 `optim.solve` | -40 | -14 | Removed two wrapper-only tests. |
| T3 `ResidualSpec`/`.spec` | -82 | 0 | Includes adjacent source truth text. |
| T4 Warp backward scaffolding | -71 | 0 | Active recompute VJP preserved. |
| T5 CRBA orphan buffer | -4 | 0 | Removes the wasted allocation. |
| T6 zero-caller members | -22 | -1 | Dense normal matrix is formed inline. |
| T7 validate once | +22 | +99 | Includes an 88-line counter test and test-file formatting. |
| T8 shared solver helpers | -4 | 0 | One internal helper module serves LM and Adam. |
| T9 mimic reductions | -2 | 0 | Public readable names remain thin wrappers. |
| T10 raw-FK VJP reuse | -88 | 0 | Deletes both duplicate FK helpers. |
| T11 joint layout descriptor | -24 | 0 | Less than the plan's estimated 30–40 lines. |
| T12 named mimic result | +5 | 0 | Readability win, intentionally not a size win. |
| T14/T15 + mimic coverage | 0 | -134 | Five duplicate/advisory tests removed; one stronger test added. |
| Tier-6 source truth | -2 | 0 | Corrects the retained legacy matrix-free description. |
| **Total** | **-322** | **-60** | **382 net lines removed.** |

## Verification

- Full CPU gate: **1,362 passed, 1 skipped, 3 deselected** with
  `-m "not bench and not cuda"`.
- Focused gates included 447 FK/dynamics/Pinocchio/contract tests, 122 API and
  hygiene tests, 84 solver/mimic tests, 50 mimic-build tests, 43 bound-layout
  tests, and 12 Warp CPU parity/gradcheck/gradgradcheck tests.
- Sphinx HTML build succeeds; only the four expected offline intersphinx
  inventory warnings remain.
- Scoped Ruff, formatting, `git diff --check`, and offline lock validation
  pass.
- CI remains manual-only (`workflow_dispatch`); this milestone did not start
  or enable CI.

## Deviations and findings

1. T2's “zero test callers” statement was false: two tests exercised only the
   removed wrapper. They were deleted. T6 likewise named one affected test,
   but two Adam tests also monkeypatched `Problem.normal_matrix`; their stale
   entries were removed.
2. T7 intentionally moved two invalid-input assertions from the raw FK helper
   to the public boundary. Formatting that test file added mechanical churn;
   raw helpers now deliberately trust prevalidated inputs.
3. T10 could not pass `ModelStructure`/`ModelValues` through a
   `torch.library` tensor schema. The active path therefore uses a small
   `torch.autograd.Function` wrapper around the forward/fake/compile custom op
   and recomputes via `forward_kinematics_raw`. This keeps first- and
   second-order gradients green without a global registry. M6 must certify
   the active wrapper under CUDA capture; the direct custom op's existing
   full-graph compile test remains green.
4. The plan requested one commit per task. Overlapping API/docs files made
   T1–T3 and T6 one reversible dead-surface commit, while T8/T9 share another.
   Additionally, the T2/T3 ledger rows entered the earlier T4 commit because
   agents shared one worktree. The final history is coherent, but not strictly
   task-atomic.
5. The expected test reduction was larger than achieved because T7's required
   counter contract and the new mimic-IK regression add real coverage. The
   exact milestone delta is reported above rather than the plan estimate.
6. The standing external-consumer grep was not run because the owner forbade
   access outside this repository. Removed surfaces were verified against
   BetterRobot's source, tests, examples, and docs only; downstream migration
   verification remains an M4 external-work item and is deferred.

## Commits

- `c0d6e51` — remove dead Warp backward scaffolding
- `aa6cb62` — remove orphan CRBA buffer
- `e1e35e8` — prune dead optimization surfaces
- `03dbd55` — remove duplicate coverage and add mimic IK
- `6f24bb1` — validate public pass inputs once
- `1f02b12` — share solver and mimic helpers
- `bb186f4` — reuse raw FK in the Warp VJP
- `56f2b0a` — name the mimic reduction result
- `23b2a8c` — unify joint coordinate layouts
