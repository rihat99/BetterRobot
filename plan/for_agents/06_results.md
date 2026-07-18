# 06 Results — Loose ends

**Status:** complete on `dev` (2026-07-18).

## Numbered disposition

| Item | Resolution |
|---|---|
| 1. Manifold error | Verified the owner-written de-milestoned message and its exact-string test; both are included in this task. |
| 2. Inactive objective | Verified the Task-02 skip in residual and objective paths. Added a kernel whose `rho(0) != 0`; zero-weight objective and gradient stay zero and never call `rho`. |
| 3. Contact point | `solve_contact_forces` and the task reference now say forces act at joint origins as `[force, torque=0]`; arbitrary offsets and their `r × force` moment remain future work. |
| 4. Placeholder baseline | Ran the full pinned eight-case CPU sweep. Its acceptance was inconclusive, so deleted both placeholder baseline files and removed the trajectory harness's implicit committed-output promise; explicit-output measurements remain supported. |
| 5. Benchmark naming | Renamed the active 144-selector harness/test/artifact to neutral `baseline` names, changed active schema identifiers, and resynced definitions and references. |
| 6. Agent briefings | Rewrote the four requested `CLAUDE.md` files from 450 to 204 lines, with current APIs, layering, numerical contracts, and extension rules. |
| 7. Examples | All four examples use live APIs and pass with `--no-viewer`. The Panda trajopt viewer now consumes the solved trajectory directly; the stale `Trajectory(u=...)` and unnecessary fp32 copy are gone. |
| 8. Changelog | Rewrote Unreleased as user-facing changes/removals and linked each removal group to the migration ledger. Audited Tasks 1–4 and filled missing public-member/import-path migrations. |
| 9. Stray directory | Verified `src/better_robot/utils/` is absent. |

## Verification

| Check | Result |
|---|---|
| Full non-benchmark/non-CUDA gate | `1469 passed, 2 skipped, 16 deselected, 38 warnings` |
| Manifold + robust objective focus | `33 passed` |
| Example/contact focus | `21 passed, 18 warnings` |
| Benchmark normal-suite gate | `23 passed, 1 deselected`; renamed-harness focus `18 passed` |
| Four `examples/*.py --no-viewer` smokes | all passed; Panda trajopt converged in 9 iterations with residual `3.681e-07` |
| Sphinx HTML | succeeded; only four expected offline-intersphinx warnings |
| Placeholder acceptance grep | zero hits in `src/`, `tests/`, `docs/`, and `benchmarks/` |
| Deleted-API example sweep | zero hits |
| Task-scoped Ruff, format, and diff checks | clean |

CUDA-marked tests were not run, per the standing owner-run rule.

## Line accounting

- The four agent briefings: 450 → 204 lines, net **−246**.
- Entire Task-06 commit, including benchmark definitions, examples, tests,
  plans, and this report: **446 additions, 653 deletions, net −207** by Git
  numstat with rename detection.

## Deviations and findings

1. The canonical CPU attempt completed its matrix but could not produce an
   accepted baseline: dense `T=50` succeeded, `T=125` timed out at 600
   seconds, and `T=250/500` exceeded the 16 GiB cap. Structured
   `T=50/125/250/500` all succeeded at median 0.190, 0.259, 0.394, and
   0.672 seconds/update; its time and incremental-RSS slopes were 0.548 and
   0.914. With only one dense point, acceptance was inconclusive.
   The generated result was therefore not committed; the plan's deletion path
   was used instead.
2. Historical `m6-*` labels and cache paths inside already-measured JSON are
   retained as immutable measurement provenance. Active filenames, Python
   modules, test names, schema identifiers, and prose use neutral names.
3. The migration audit found more gaps than the known `ReferenceFrame` row:
   removed `ImplicitDiffConfig` knobs, `VarSpec.check_feasible`, scalar
   objective members, a deep Jacobian-strategy path, and matrix-free members
   also needed exact ledger coverage.
4. A headless live-viewer probe did not terminate cleanly and was interrupted.
   The required `--no-viewer` smokes passed, and source inspection confirmed
   the viewer performs its own render-boundary dtype conversion.
5. The exact placeholder grep initially found ignored stale Sphinx output and
   three old bytecode files. Those reproducible artifacts were removed; no
   tracked source or user data was deleted.
6. Task 01 removals were already fully represented in the migration ledger;
   Task 03 was complete after the `ReferenceFrame` row, and Task 04 removed no
   public symbol or import path.

No requested Task 06 work remains incomplete.
