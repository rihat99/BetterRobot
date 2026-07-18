# 03 Results — Open the core

**Status:** complete on `dev` (2026-07-18).

## Delivered

- Exported all seven tensor-only raw passes. FK, frame placements, and joint
  Jacobians now return frozen named result dataclasses, matching the existing
  dynamics convention; the Jacobian raw pass no longer requires `Data` cache
  sequencing.
- Exported `ModelStructure` and `ModelValues` at package root, made
  `br.spatial` attribute-reachable, and removed the unused `ReferenceFrame`
  enum in favor of the literal-string API.
- Made `Data` optional and keyword-only across RNEA, ABA, CRBA, bias/gravity,
  COM, and centroidal wrappers. Supplying an exact `Data` still populates it;
  omitting one allocates it after execution-batch resolution.
- Added differentiable implicit IK wiring and retained target-pose gradients;
  contact-force result tensors no longer detach the graph available from the
  forward dynamics pass.
- Removed the enumerated exported stubs, the `nle` alias, the empty collision
  and manipulability residual modules, and `JointAccelLimit`. Migration and
  future-work records now state the supported alternatives honestly.
- Added `tests/autograd/test_public_differentiability.py`: 16 fast float64 VJP
  tests cover public/raw kinematics and dynamics, centroidal quantities,
  integrate/difference, model-value swaps, and a pose residual.
- Regenerated the API reference and migrated task, test, benchmark, Warp, and
  Pinocchio consumers to the named raw results and optional-`Data` signatures.

## Verification

| Check | Result |
|---|---|
| Full non-benchmark/non-CUDA gate | `1439 passed, 2 skipped, 16 deselected, 38 warnings` |
| Integrated Task 03 focused suite | `302 passed, 33 warnings` |
| Public differentiability module | `16 passed` |
| Pinocchio dynamics focus | `37 passed` |
| Sphinx HTML | succeeded |
| Sphinx doctest | `1 passed`; four expected offline-intersphinx warnings |
| Task-owned Ruff check / format and diff check | clean |
| Retired-symbol/API-doc stale grep | no actionable hits |

CUDA-marked tests were not run, per the standing owner-run rule.

## Line accounting

- Task-owned Python under `src/`: 210 additions, 387 deletions, net **−177**.
- Budget: net ≤ +150 lines; actual headroom is 327 lines.
- The public differentiability regression module is 201 lines and runs in
  roughly 2.6 seconds in isolation.

## Deleted-test disposition

- `tests/dynamics/test_integrators.py` removed the parametrized
  `test_dynamic_integrators_still_stubs` (three cases) with the deleted
  semi-implicit Euler, symplectic Euler, and RK4 placeholders. All numerical
  `integrate_q` tests remain.
- `tests/test_pinocchio/test_rnea_matches_pinocchio.py` removed
  `test_compute_coriolis_matrix_still_raises` with that deleted placeholder.
  All RNEA numerical, batch, external-force, data-population, and autograd
  tests remain.
- No tests existed solely for the other removed stub exports.

## Deviations and findings

1. The approved contract-file list omitted
   `tests/contract/test_boundary_validation_count.py`, but the required
   dynamics signature migration necessarily changed its call syntax. Its
   assertion and expected validation count are unchanged.
2. Pinocchio parity tests also required the same mechanical call migration.
   No parity assertion, tolerance, seed, or expected value changed. Their
   pre-existing local-import/unused-import Ruff findings were intentionally
   not broadened into unrelated cleanup.
3. The plan described `ReferenceFrame` as unused, but a few residual type
   annotations/defaults still referenced it. They were migrated to the
   already-supported literal strings before removing the enum.
4. `JointAccelLimit` survived Task 02 as an exported `__call__` placeholder,
   while `ModelValues` has no acceleration-limit field. It was removed with a
   ledger entry rather than continuing to advertise an unusable residual.
5. The obsolete `src/better_robot/utils/` contained only ignored bytecode
   cache files, so removing it produces no tracked deletion.
6. The first full gate found two benchmark harnesses still unpacking the old
   FK tuple. Both were migrated to named result fields; the focused benchmark
   subprocess and the repeated full gate then passed.
7. Remaining explicit `NotImplementedError` sites are honest runtime guards
   for deferred COM acceleration, angular contact consistency, jerk, and
   nullspace behavior; none is an enumerated Task 03 export stub.
8. Existing user edits in `docs/concepts/index.md`, untracked
   `docs/concepts/design_decisions.md`, and the matched manifolds source/test
   wording pair were preserved and excluded from this task's commit.

No requested Task 03 behavior remains incomplete.
