# 02 results — Core truth and LEGO fixes

**Status:** complete.

## Delivered

| Task | Delivered disposition |
|---|---|
| T1 — live residual surface | Deleted the raise-only `NullspaceResidual` and `JerkResidual` implementations and exports. Removed the unsupported `angular` argument from `ContactConsistencyResidual`, made the remaining smoothness hook abstract instead of raise-only, updated the supported-residual narrative and roadmap inventory, regenerated the three affected API pages, and added migration-ledger rows. The unimplemented ideas remain future work rather than placeholder API. |
| T2 — caller-graph cleanup | Deleted the unused public `ModelValues.execution_batch_shape`; production continues through the validated private `_execution_batch_shape`. Replaced the optimizer's duplicate batch-slice expression with `Variable.batch_shape_of`; order 01 had already consolidated the other intended locations. The generated `ModelValues` page and migration ledger reflect the removal. |
| T3 — revolute consolidation | Replaced the duplicated RX/RY/RZ and unaligned method bodies with one `_Revolute` mixin and axis hooks. Public joint classes, dataclass fields, joint kinds, and dispatch behavior remain unchanged. `revolute.py` fell from 243 to 194 physical lines. |
| T4 — stale inertia | Deleted `ModelValues.body_inertias_6x6`; `spatial_inertias()` now always derives from the current packed inertias. Removed cache plumbing from model construction, rebinding, transfer, and pytree reconstruction. Added the exact `dataclasses.replace` physics-and-gradient regression, documented the functional seam, and regenerated the API page. |
| T5 — raw frame Jacobian | Added and exported `frame_jacobian_raw`, including optional joint-Jacobian reuse and `world`, `local_world_aligned`, and `local` references. `get_frame_jacobian` now routes through the same implementation. Added public-import, autograd, all-reference parity, and fullgraph compile coverage and updated the raw-pass docs/API. |
| T6 — result convention | Changed `CCRBAResult` from `NamedTuple` to a frozen dataclass, migrated every in-repo unpacking caller to named fields, added a frozen-instance assertion, updated conceptual/generated docs, corrected the changelog claim, and recorded tuple removal in the migration ledger. |
| T7 — briefing truth pass | Removed the exact `named-block` jargon from source briefings and remaining source messages, refreshed kinematics/residuals guidance for the object-referenced API and raw frame pass, and verified zero source hits. |

## Verification

| Gate | Final result |
|---|---|
| Full non-bench/non-CUDA suite | **1,579 passed, 2 skipped, 15 deselected**; 38 warnings. |
| Contracts | **353 passed**. |
| Pinocchio parity | **135 passed**. |
| Documentation tests | **26 passed**. |
| Consolidated Part 2 acceptance focus | **252 passed**; 33 warnings. This covers roadmap/submodule/protocol contracts, raw-frame export/autograd/parity/compile, revolute joint and FK compile coverage, stale-cache dynamics/gradients, residual removals, and frozen CCRBA behavior/parity. |
| T1/T2 development focus | **351 residual/optimizer tests passed** plus a separate **157 focused tests passed**. |
| T3 development focus | **110 focused tests passed** plus the **135-test parity** gate. |
| T4 development focus | **104 data-model, dynamics, autograd, and dynamics-parity tests passed**. The post-fix probe had zero physics error and 25 finite, nonzero inertia-gradient entries. |
| Sphinx HTML | Fresh build of **158 documents succeeded** with four offline intersphinx DNS warnings. Seven affected generated API pages were refreshed; deleted symbols are absent and the raw frame/frozen CCRBA surfaces are present. |
| Sphinx doctest | **19 passed, 0 failed**, with the same four offline intersphinx warnings. |
| Changed-file style | **22 changed Python files**: Ruff check passed, Ruff format reported all 22 formatted, and `git diff --check 9191f9f` passed. |
| Full-repository Ruff | **138 findings**, improved from the 140-finding baseline but not green; see findings below. |

Acceptance evidence:

- `NullspaceResidual` and `JerkResidual` have zero source/test hits.
- `named-block` has zero source hits.
- The removed public `ModelValues.execution_batch_shape` definition and call
  form have zero hits.
- `from better_robot.kinematics import frame_jacobian_raw` works and the name
  is present in `better_robot.kinematics.__all__`.
- The replace-then-gradient stale-inertia regression exists and passes.

### T4 measurement and decision

Pre-fix probe: replacing packed inertias with `dataclasses.replace` retained
the old cache (`max_abs_physics_error=0.25`) and produced an output with no
gradient route to the replacement tensor.

Measurement used the canonical 25-joint SMPL-like CPU float32 eager RNEA
workload, one thread, with seven blocked-autorange repeats. Expanding packed
inertias took 128.972 us. Recomputing on each pass changed median RNEA time
from 21,128.643 to 21,230.585 us at `B=1` (+0.482%) and from 30,703.289 to
30,833.703 us at `B=256` (+0.425%).

Decision: delete `body_inertias_6x6` and always derive spatial inertias from
the current packed tensor. The measured cost is negligible, and deletion
makes stale replacement state impossible without an identity-tracking cache.
This follows the preferred plan branch; there is no deviation.

## Line accounting

Method: raw physical Python lines for the baseline/current totals, and Git
added/deleted line accounting for the per-task attribution.

| Scope | Added | Deleted | Net |
|---|---:|---:|---:|
| T1 — residual surface | 7 | 54 | -47 |
| T2 — dead wrapper/deduplication | 1 | 11 | -10 |
| T3 — revolute consolidation | 24 | 73 | -49 |
| T4 — stale-inertia cache | 1 | 9 | -8 |
| T5 — raw frame Jacobian | 62 | 79 | -17 |
| T6 — CCRBA result | 3 | 4 | -1 |
| T7 — Python wording | 2 | 2 | 0 |
| **Python source total** | **100** | **232** | **-132** |

`src/**/*.py` moved from **21,596** to **21,464** physical lines. Including
the edited `src/**/CLAUDE.md` briefings, the complete `src/` diff is 104 lines
added and 235 deleted, net **-131**. The required net source delta is therefore
met.

## Deleted-test disposition

| Previous test/assertion | Disposition |
|---|---|
| `test_deletion_owned_placeholders_are_v2_residuals` | Deleted with the two raise-only classes. It tested only that unsupported public placeholders raised. Zero-symbol/API inventory checks now protect their removal; supported temporal and regularization behavior remains covered. |
| `ContactConsistencyResidual(..., angular=True)` raises | Rewritten as a signature assertion that `angular` is absent, matching the removed surface. Numerical contact finite-difference coverage remains in the same test. |
| `test_static_spatial_inertia_cache_matches_live_derivation` | Replaced by `test_dataclasses_replace_keeps_spatial_inertia_physics_and_gradients_live`, which exercises the previously broken operation without manually clearing a cache. |
| Static-cache reuse assertions in `test_with_values.py` | Rewritten to require fresh live derivation and preserved gradients after `Model.with_values`. |
| `test_ccrba_returns_named_tuple_compatible_fields` and tuple-unpacking callers | Rewritten as `test_ccrba_returns_frozen_dataclass_fields`; all dynamics, autograd, mimic, and Pinocchio callers now read `.centroidal_map` and `.momentum`. |

No T3 or T5 numerical tests were deleted; their existing coverage was retained
and extended.

## Deviations and findings

- The acceptance command's broad substring
  `grep -rn "...\|execution_batch_shape" src/ tests/` cannot produce zero:
  the intended private `_execution_batch_shape`, public
  `broadcast_execution_batch_shape`, and test-oracle parameter names are live
  and required. The removed public method itself has zero definition/call
  hits. This is a contradiction in the written grep, not an implementation
  deviation.
- Full-repository Ruff is not green: 138 baseline findings remain, although
  that is two fewer than before the work and every changed Python file passes
  lint and formatting. No Ruff suppression or contract weakening was added.
- The four Sphinx warnings are offline failures to resolve external
  intersphinx inventories; HTML and doctest generation both completed.
- Apart from the contradictory grep, T1–T7 followed the accepted plan. T4
  selected the plan's preferred deletion branch based on the recorded
  measurement, and no behavior/parity deviation was observed.
