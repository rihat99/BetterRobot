# 02 — Core truth and LEGO fixes

**Goal:** close the gaps the 2026-07-19 verification found outside `optim/`:
exported classes that only raise, a dead wrapper, copy-paste the polish round
half-finished, one genuinely dangerous cache footgun, and the one missing raw
pass. Small, evidence-backed, every item cited.

**Size budget:** net `src/` delta ≤ 0 (the additions — `frame_jacobian_raw`
and the cache fix — are smaller than the deletions).

**Contract-test authorization:** `tests/contract/test_public_api.py`,
`test_submodule_public_imports.py`, `test_roadmap_stub_inventory.py`. No
other contract file changes; parity untouched.

---

## T1 — The surface stops advertising raise-only classes

- Delete `NullspaceResidual` (`residuals/regularization.py:272-288`) and
  `JerkResidual` (`residuals/smoothness.py:255-273`) and their exports
  (`residuals/__init__.py:18,20`, `__all__:47,53`). Both are placeholders
  whose `__call__` only raises; the polish order's letter excluded them, its
  goal did not. MIGRATION.md rows; `for_future.md` records the features
  (nullspace regularization needs a posture-projection contract; jerk needs
  a third-difference temporal declaration).
- Remove the `angular` parameter from `ContactConsistencyResidual`
  (`residuals/contact.py:48`, raise at `:51-52`) — it exists only to raise.
  The future feature note already lives in `for_future.md`.
- Update the roadmap page's explicit-raise inventory and its contract test,
  the generated API pages for the deleted classes, and the placeholder
  mention at `docs/concepts/residuals_costs_and_solvers.md:134`.

## T2 — Dead code found by the caller graph

- Delete `ModelValues.execution_batch_shape`
  (`data_model/model_values.py:87-95`): zero callers in `src/` and `tests/`;
  every production path uses `_execution_batch_shape`. Its only job was the
  hot-path re-validation the polish round removed.
- Deduplicate the batch-shape slice idiom: `variables.py` (v2 location),
  `_solver_common.py:15`, `problem.py` — one helper, reused. (If order 01
  already consolidated these inside `optim/`, verify and close.)

## T3 — Finish the joint-family consolidation

`JointRX`/`JointRY`/`JointRZ` (`data_model/joint_models/revolute.py:72-156`)
are three copies of the same seven methods differing by an axis constant, while
the prismatic family already shares a `_Prismatic` mixin with a `ClassVar`
axis (`prismatic.py:50-116`). Fold revolute the same way (`_Revolute` mixin).
No behavior change: pinocchio parity and the joint/protocol/compile focus
suites must pass untouched, and the dispatch table stays compile-friendly.

## T4 — Kill the stale-inertia footgun

`dataclasses.replace(values, body_inertias=...)` currently keeps the stale
`body_inertias_6x6` cache, and `spatial_inertias()`
(`model_values.py:150-155`) silently returns wrong physics — verified: a
gradient probe needed a manual `body_inertias_6x6=None` to be correct. This
is a correctness bug on the documented raw seam, not polish. Fix so a stale
cache is **impossible**: either drop the cache field (measure the dynamics
hot-path cost first — if the recompute is negligible, deletion wins) or make
`spatial_inertias()` detect that its source tensor is no longer the one the
cache was built from and recompute. Add a regression test that does the
`replace`-then-gradient probe. Document the outcome on the seam contract
(`docs/concepts/the_compute_seam.md`) if user-visible behavior changes.

## T5 — The missing raw block: frame Jacobians

The raw seam covers joint Jacobians but not frame Jacobians — the block a
robotics user actually wants (end-effector). Add `frame_jacobian_raw` beside
`joint_jacobians_raw`: tensor-in/tensor-out, reusing the LOCAL_WORLD_ALIGNED
frame-offset math at `kinematics/jacobian.py:225-256` (one implementation —
the public `get_frame_jacobian` routes through the same helper). Export it,
give it a case in `tests/autograd/test_public_differentiability.py` and a
parity check against the public path. Update the raw-function inventory at
`docs/concepts/kinematics_and_jacobians.md:193` and the generated API page.

## T6 — Result-type convention straggler

`ccrba` returns a `CCRBAResult` NamedTuple kept tuple-compatible while its
raw sibling returns the frozen `CentroidalResult` dataclass
(`dynamics/centroidal.py:68,80,217`). No-backcompat rules apply: make the
public result a frozen dataclass consistent with its family, update the
(in-repo) unpacking callers, the tuple-unpacking claim at
`docs/concepts/dynamics.md:78`, the generated API page, ledger row.

## T7 — Agent-briefing jargon sweep

`src/better_robot/tasks/CLAUDE.md:5,12,19,25`,
`residuals/CLAUDE.md:1`, `kinematics/CLAUDE.md:39` still say "named-block".
After order 01's API rewrite these briefings need a truth pass anyway —
re-sync all `src/**/CLAUDE.md` to the v2 vocabulary and delete the
"named-block" term everywhere (it referred to a distinction that no longer
has a second side).

## Acceptance

- `grep -rn "NullspaceResidual\|JerkResidual\|execution_batch_shape" src/ tests/` → 0.
- `grep -rn "named-block" src/` → 0.
- `from better_robot.kinematics import frame_jacobian_raw` works and is in
  `__all__`; its autograd + parity tests pass.
- The stale-cache regression test exists and passes; the chosen mechanism is
  recorded in the results file with the measurement that justified it.
- Parity, contracts (authorized only), full gate, ruff, Sphinx green; net
  delta and per-item disposition in `02_results.md`.
