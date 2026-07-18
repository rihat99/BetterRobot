# 02 Results — Rebuild the optimizer core

**Status:** complete on `dev` (2026-07-18).

## Delivered

- Flattened `better_robot.optim` into thirteen Python modules. The public
  surface is re-exported from `better_robot.optim`; the old `blocks/`, deep
  kernel/solver packages, and `structure.py` are gone.
- Added the mutable `Problem` builder, callable residual wrapping and default
  names/reads, `RobotConfig.joint_bounds()`, and automatic robot-state provider
  wiring. Direct IK now needs only the robot manifold, one variable, one pose
  residual, and LM.
- Removed scalar objectives, `ResidualState`, residual-owned finite-difference
  Jacobians, the Jacobian-strategy enum, provider access policing, and all
  `_prevalidated` evaluation twins. Structural validation remains at public
  boundaries; value-content scans do not run per evaluation.
- Replaced the custom Adam and phase engine with a 96-line adapter around any
  `torch.optim.Optimizer`. IK's `adam` and `lm_then_adam` modes retain their
  solution-quality regressions; staged refinement is two explicit calls with
  a rebuilt weighted problem.
- Reduced LM/GN to dense and block-banded routes. Removed `LSTSQ`, the
  Cholesky rank-deficient fallback, `NormalCG`, `NormalOperator`, matrix-free
  routing, warm-start linear steps, and pure linear-solve diagnostics.
- Slimmed implicit differentiation from 816 to 477 formatted lines while
  retaining convergence/active-set checks, quaternion branch-cut checks,
  robust-kernel kink rejection, residual-checked adjoints, and dense/banded
  materialization gates. `ImplicitDiffConfig` now has the two externally
  meaningful controls.
- Rebuilt the optimizer documentation and generated API pages. The exact
  architecture line-fit example is a passing Sphinx doctest.

## Verification

| Check | Result |
|---|---|
| Full non-benchmark/non-CUDA gate | `1415 passed, 2 skipped, 16 deselected, 38 warnings` |
| Pinocchio parity | `136 passed` |
| Contract suite | `312 passed` |
| Implicit differentiation | `23 passed` |
| Final LM/implicit focused regression | `131 passed` |
| Line-fit architecture example | solution `[2, 1]`, converged, cost `1.7764e-15` |
| Direct builder IK | covered by `test_direct_builder_ik_auto_wires_robot_state` |
| Sphinx HTML | succeeded |
| Sphinx doctest | `1 passed`; four expected offline-intersphinx warnings |
| Ruff check / format / diff check | clean |
| Removed-runtime stale grep | 0 actionable hits |

CUDA-marked tests were not run, per the standing owner-run rule.

## Line accounting

- Python under `src/better_robot/optim`: 6,288 → **3,779** lines, net
  **−2,509** (hard cap: 3,800).
- All Python under `src/`: 25,060 → 22,148 lines, net **−2,912**.
- Largest survivors: `lm.py` 943, `problem.py` 662, `implicit.py` 477,
  `temporal.py` 460, and `solvers.py` 295 lines.

## Deleted-test disposition

Every removed test function is accounted for below. Renamed tests are listed
because Git records the old function as deleted even when its surviving
assertions moved into a narrower replacement.

- `tests/kinematics/test_jacobians.py`:
  `test_finite_diff_fallback_costs_two_nv_plus_one_evaluations` tested the
  deleted `ResidualState` finite-difference engine.
- `tests/optim/test_config_wiring.py`:
  `test_solve_ik_honours_linear_solver_string` exercised only the removed
  `linear_solver="lstsq"` route; unknown-option and Cholesky wiring remain.
- `tests/optim/test_implicit_diff.py`:
  `test_matrix_free_and_large_dense_materialization_are_rejected_before_backward`
  became `test_large_dense_materialization_is_rejected_before_backward`; only
  the deleted matrix-free assertion was removed.
- `tests/optim/test_linear_operators.py`:
  `test_normal_cg_matches_dense_with_ridge_and_batch_axes`,
  `test_exact_warm_start_converges_without_an_iteration`,
  `test_cg_failure_statuses_are_isolated_and_solutions_are_safe`,
  `test_iteration_cap_reports_max_iter_and_returns_safe_zero`, and
  `test_cg_validates_operator_and_warm_start_contracts` died with
  `NormalCG`/`NormalOperator`. The old fallback test
  `test_cholesky_informative_path_is_strict_but_public_fallback_survives`
  became `test_cholesky_informative_path_reports_per_element_failure` and now
  pins strict public Cholesky behavior.
- `tests/optim/test_linear_solvers.py`:
  `test_dense_solvers_accept_arbitrary_batch_axes`,
  `test_dense_solvers_apply_per_element_ridge`,
  `test_dense_solvers_accept_scalar_ridge_and_legacy_two_arg_call`, and
  `test_tensor_ridge_must_share_working_dtype_and_batch_shape` became
  Cholesky-only equivalents after `LSTSQ` was removed.
  `test_cholesky_fallback_is_independent_per_batch_element` died with the
  rank-deficient fallback.
- `tests/optim/test_manifolds.py`:
  `test_robot_config_prevalidated_projection_skips_bounds_revalidation` died
  with the shadow API;
  `test_infeasible_panda_neutral_is_rejected_without_clamping` and
  `test_public_retract_and_difference_reject_infeasible_base_state` pinned
  removed value policing; and
  `test_initial_group_values_must_belong_to_their_manifold` plus
  `test_initial_robot_configuration_rejects_invalid_unit_coordinates` were
  consolidated into the structural-validation test.
- `tests/optim/test_phase_engine.py`: all eight tests were deleted with the
  one-caller phase engine:
  `test_phase_masks_are_functional_and_cannot_unfreeze_base_mask`,
  `test_zero_weight_skips_residual_and_its_lazy_provider`,
  `test_original_problem_is_unchanged_when_phase_solver_raises`,
  `test_on_start_runs_once_even_for_zero_iterations`,
  `test_phase_override_names_and_mask_shapes_fail_clearly`,
  `test_adam_mask_transition_uses_fresh_reduced_moments`,
  `test_mixed_lm_then_adam_phases_share_problem_and_converge`, and
  `test_batched_phase_run_matches_sequential_elements`. IK regression and
  inactive-weight/provider tests retain the required outcomes.
- `tests/optim/test_problem_blocks.py`:
  `test_scalar_term_weighted_gradient_diagnostics_and_second_order_fence` and
  `test_tensor_objective_weight_requires_scalar_or_exact_batch_shape` died
  with scalar objectives;
  `test_declared_reads_and_working_dtype_fail_fast` split into structural
  reads and dtype tests with undeclared access now allowed; and
  `test_prevalidated_solver_path_matches_public_path_without_host_tensor_reads`
  died with the duplicate private path.
- `tests/optim/test_problem_robust_gradient.py`:
  `test_scalar_objective_keeps_linear_weight_alongside_robust_residual` died
  with scalar objectives.
- `tests/optim/test_providers.py`:
  `test_problem_rejects_unknown_provider_inputs_and_item_reads` became the
  `reads`-named equivalent, while
  `test_provider_cannot_read_an_undeclared_dependency` became
  `test_provider_may_read_an_undeclared_dependency` under the new declaration
  policy.
- `tests/optim/test_solver_adam_matrix_free.py`: ten custom-state tests were
  removed with `AdamState`/`AdamStatus` and their authorized semantics:
  `test_solver_and_state_are_frozen_tensor_pytrees_with_reduced_moments`,
  `test_update_uses_only_prevalidated_matrix_free_gradient`,
  `test_run_never_materializes_a_jacobian`,
  `test_so3_identity_step_is_finite_and_stays_on_manifold`,
  `test_arbitrary_batch_axes_match_sequential_runs`,
  `test_external_update_loop_matches_run_values_and_moments`,
  `test_warm_start_retains_moments_but_refreshes_target_artifacts`,
  `test_compatible_warm_start_needs_fewer_new_steps_than_cold`,
  `test_scalar_objectives_are_supported_and_default_run_is_detached`, and
  `test_invalid_batch_element_fails_without_moving_valid_neighbor`. Three new
  tests cover generic Adam/SGD factories, persistent leaf buffers with
  per-element stopping, and manifold/mask/bounds projection.
- `tests/optim/test_solver_lm_pattern.py`:
  `test_second_order_entry_points_reject_scalar_objectives` died with the
  objective subsystem.
- `tests/optim/test_solver_lm_routing.py`:
  `test_auto_banded_matches_forced_dense_and_matrix_free` became
  `test_auto_banded_matches_forced_dense`, and
  `test_missing_numeric_blocks_are_operator_only` became the dense-fallback
  test.
- `tests/optim/test_temporal_structure.py`:
  `test_cached_analysis_keeps_zero_weight_and_distinguishes_operator_direct`
  became the numeric-block eligibility test; and
  `test_block_banded_scaled_restricted_and_normal_operator_metadata` became a
  direct dense oracle without operator metadata.
- `tests/residuals/test_temporal_structure.py`:
  `test_contact_named_blocks_and_transpose_match_dense` became the direct
  block-vs-dense test after transpose hooks were removed; and
  `test_static_horizon_validation_is_eager_and_legacy_mode_remains` became
  `test_static_horizon_validation_is_eager_and_required` after legacy mode
  was deleted.
- `tests/tasks/test_ik_block_rebase.py`:
  `test_built_in_kinematic_residuals_match_legacy_state_protocol` became the
  shared-context protocol test.
- `tests/tasks/test_ik_config_truth.py` and
  `tests/test_skeleton_signatures.py`:
  `test_jacobian_strategy_requires_the_public_enum` and
  `test_jacobian_strategy_enum_values` died with the enum; string validation
  remains in optimizer/task tests.

## Deviations and findings

1. The work order explicitly required removing `JacobianStrategy` from the
   package-root API, but its contract allowlist omitted
   `tests/contract/test_public_api.py`. The one-line required-symbol removal is
   the only contract-file authorization deviation.
2. Candidate `LMState` fields were verified before removal. `increase_factor`
   is required for rejection escalation and warm starts; `gain_ratio` and
   `relative_decrease` remain pinned algorithm diagnostics. The removable
   fields were `previous_linear_step`, `projected_gradient`, and all
   `linear_solve_*` state diagnostics.
3. A first `make -C docs doctest` invocation could not locate `sphinx-build`
   because the virtualenv was not on that shell's `PATH`. Running the same
   builder explicitly as `.venv/bin/sphinx-build -b doctest ...` passed; this
   was an invocation issue, not a documentation failure.
4. The pre-existing user edits in `docs/concepts/index.md`, the untracked
   `docs/concepts/design_decisions.md`, and the wording-only hunk in
   `tests/optim/test_manifolds.py` were preserved but intentionally excluded
   from this task's commit.

No solution-quality tolerance changed, and no required Task 02 behavior was
left incomplete.
