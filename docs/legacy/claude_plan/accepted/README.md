# Accepted strategic proposals (audit trail)

These are the 17 strategic proposals from `claude_plan/` accepted on
2026-04-25 and folded into the canonical specs under
[`../../design/`](../../design/) and
[`../../conventions/`](../../conventions/).

This folder is **historical**. The proposal bodies are preserved
verbatim so future reviewers can trace any decision back to its
authoring rationale; the canonical specs link in the other direction
when they contain a "gpt-plan-driven correction" or "an earlier draft
proposed X" line.

For the index of where each proposal landed, see
[`../README.md`](../README.md). For the implementation sequence, see
[`../../UPDATE_PHASES.md`](../../UPDATE_PHASES.md).

| # | Proposal | One-line summary |
|---|----------|------------------|
| 01 | [01_lie_typed_value_classes.md](01_lie_typed_value_classes.md) | Add typed `SE3`/`SO3`/`Pose` wrappers in `lie/types.py`; keep `lie/` functional |
| 02 | [02_backend_abstraction.md](02_backend_abstraction.md) | Make `backends/` load-bearing via explicit `Backend` objects; global is sugar |
| 03 | [03_replace_pypose.md](03_replace_pypose.md) | Plan a pure-PyTorch SE3/SO3 backend to retire PyPose's autograd bug |
| 04 | [04_typing_shapes_and_enums.md](04_typing_shapes_and_enums.md) | Adopt jaxtyping shape annotations; replace string literals with enums |
| 05 | [05_value_types_audit.md](05_value_types_audit.md) | Reconcile `Inertia` typed wrapper vs packed-10 ambiguity; `Symmetric3` polish |
| 06 | [06_public_api_audit.md](06_public_api_audit.md) | Audit `__all__`; promote `SE3` and `ModelBuilder`; submodule access for the rest |
| 07 | [07_data_cache_invariants.md](07_data_cache_invariants.md) | Make `_kinematics_level` enforce; add stale-cache detection |
| 08 | [08_trajectory_lock_in.md](08_trajectory_lock_in.md) | Pin the `Trajectory` dataclass and temporal-residual API before trajopt body lands |
| 09 | [09_human_body_extension_lane.md](09_human_body_extension_lane.md) | Reserve clean seams for anatomical joints, muscles, SMPL — without polluting core |
| 10 | [10_user_docs_diataxis.md](10_user_docs_diataxis.md) | Add Sphinx + MyST + Diátaxis; `docs/` becomes the user manual, not just specs |
| 11 | [11_quality_gates_ci.md](11_quality_gates_ci.md) | Make every contract test in 16 actually run on every PR |
| 12 | [12_regression_and_benchmarks.md](12_regression_and_benchmarks.md) | Land `fk_reference.npz`; commit perf baselines; advisory-then-blocking ladder |
| 13 | [13_style_reconciliation.md](13_style_reconciliation.md) | Reconcile the two `style/*.md` drafts with the actual codebase |
| 14 | [14_dynamics_milestone_plan.md](14_dynamics_milestone_plan.md) | Sequence D1–D7 against the new value-type and backend story |
| 15 | [15_packaging_extras_releases.md](15_packaging_extras_releases.md) | Pyproject extras (`[urdf]`, `[warp]`, `[docs]`, `[bench]`); SemVer; deprecation discipline |
| 16 | [16_optim_wiring_and_matrix_free.md](16_optim_wiring_and_matrix_free.md) | Wire `OptimizerConfig` knobs; `ResidualSpec`; matrix-free `gradient(x)`; `BSplineTrajectory`; `MultiStageOptimizer` |
| 17 | [17_io_ergonomics_and_assets.md](17_io_ergonomics_and_assets.md) | Builder kind helpers; fix `revolute_z` mismatch; IR `schema_version`; `AssetResolver` Protocol |

## Why preserve the bodies?

Two reasons:

1. **Reasoning context.** When a future contributor asks "why is `SE3`
   in `lie/types.py` and not `spatial/`?" the answer is in
   [01 §"Why `lie/`, not `spatial/`?"](01_lie_typed_value_classes.md) —
   the gpt_plan review caught a layering inversion in an earlier draft.
   The canonical spec captures the decision; the proposal captures
   *why* it was made.
2. **Pattern of correction.** Several proposals carry "gpt-plan-driven
   corrections" — places where the colleague's review caught a real
   bug or surfaced a missing concept. Preserving the proposal bodies
   keeps that pattern visible for the next round of review.
