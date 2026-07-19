# 01 results — Optimization API v2

**Status:** complete. Review findings and accepted follow-up work are recorded
below.

## Pre-implementation line budget

Method: raw physical lines from `wc -l src/better_robot/optim/*.py`, including
blank lines, comments, and docstrings. The current 13 modules total 3,742 lines.

| Module | Before | Target | Delta | Disposition |
|---|---:|---:|---:|---|
| `__init__.py` | 68 | 72 | +4 | Replace old exports with the v2 surface. |
| `_solver_common.py` | 41 | 35 | -6 | Retain only shared tensor helpers. |
| `autograd.py` | 83 | 0 | -83 | Delete; absorb the private tangent AD core into `problem.py`. |
| `first_order.py` | 96 | 0 | -96 | Replace with `optimizers.py`. |
| `implicit.py` | 477 | 465 | -12 | Route gradients through static variables. |
| `kernels.py` | 139 | 105 | -34 | Move weight mechanics to the single `Weight` implementation. |
| `lm.py` | 935 | 960 | +25 | Preserve the numerical step; add the object-owned driver/private state. |
| `manifolds.py` | 223 | 55 | -168 | Keep `Bounds` and private helpers; move geometry into variables. |
| `optimizers.py` | 0 | 200 | +200 | Add `Optimizer`, `OptimizerInfo`, and `TorchOptimizer`. |
| `problem.py` | 621 | 600 | -21 | Add freezing/epochs and private AD; delete dict/provider churn. |
| `providers.py` | 105 | 0 | -105 | Delete; nodes live outside `optim`. |
| `solvers.py` | 295 | 340 | +45 | Add dense `LU` and consolidate the solver family. |
| `temporal.py` | 440 | 445 | +5 | Retarget unchanged banded math to object dependencies. |
| `variables.py` | 219 | 450 | +231 | Add the variable hierarchy and relocated geometry/layout logic. |
| **Total** | **3,742** | **3,727** | **-15** | **73-line reserve below 3,800.** |

The budget is feasible only if `autograd.py`, `first_order.py`, and
`providers.py` are actually deleted and the duplicate context, provider,
weight, and lifecycle machinery is not retained under new names.

## Baseline verification

| Gate | Result |
|---|---|
| Full non-bench/non-CUDA suite | 1,469 passed, 2 skipped, 16 deselected |
| Pinocchio parity | 135 passed |
| Contracts | 319 passed |
| Full-repo Ruff | 140 pre-existing findings |

## Delivered

The current worktree implements the object-referenced optimizer rewrite rather
than retaining the old dictionary surface behind aliases.

| Area | Delivered disposition |
|---|---|
| Variables | `Variable`, `SO3Variable`, `SE3Variable`, and `RobotVariable` own values, names, trainable/static role, bounds, masks, scales, batch declarations, temporal layout, and retract/difference geometry. Public manifold classes and `VarSpec` are gone; `Bounds` remains. |
| Residuals and weights | `Residual` is an ABC in the lower `residuals` layer. `ScaleWeight` and `DiagonalWeight` apply the same square-root-information multiplier to errors and Jacobian rows. `@residual` replaces both callable adapters, and `Difference` supplies the generic geometry-aware prior. |
| Problems | `Problem` harvests direct and node-reachable variables, freezes names/layout on first use, swaps trial tensors with exception-safe restoration, supports atomic named updates, and owns AD/FD, objective, gradient, dense, and temporal assembly. |
| Shared work | `Node`, `RobotState`, and `SceneSDFState` replace providers. `RobotState` merges by variable/model identity; other nodes share only by object identity. Memos are scoped to an active Problem evaluation. A late review found and fixed standalone `Node.value()` retaining stale data: standalone reads now recompute, while nested Problem scopes retain once-per-assignment sharing. |
| Optimizers | `Optimizer`, minimal `OptimizerInfo`, `OptimizerStatus`, and `TorchOptimizer` live in `optimizers.py`. Optimizers own a `Problem`; public control is `step()` / `optimize()` / `reset()`, and solved tensors remain on variables. Persistent tangent buffers, rebasing, per-element stopping, Adam/SGD factories, and the summed-objective L-BFGS closure path are present. |
| LM/GN | LM's numerical iteration and tensor state are private; the public object writes accepted candidates back to variables and reports only `OptimizerInfo`. Damping, active bounds/KKT, robust grouped IRLS, step limits, dense/banded routing, fixed-shape private update, final refresh, warm update behavior, and GN remain covered by migrated tests. |
| Implicit differentiation | `optimize(differentiate="implicit")` attaches the guarded backward. Graph-carrying static variables are the declared inputs; convergence, bounds, nonsmooth kernels, quaternion branches, rank, size, and routing guards remain. |
| Linear solvers and temporal path | Dense `LU` joins `Cholesky` and `BandedCholesky` under `LinearSolver`. Temporal analysis and numeric blocks use variable references and preserve route/reason reporting without inferring sparsity from numerical zeros. |
| Residual library | Pose, limits, human, regularization, temporal, smoothness, contact, projection, chamfer, and scene-SDF residuals now hold variables/nodes directly. No `ctx[...]`, `reads`, provider output, or `ResidualItem` path remains in production residual code. |
| Task facades | IK builds a bounded `RobotVariable`, static target/rest variables, residual objects, and public optimizers; `lm_then_adam` snapshots/restores residual weights. Trajopt accepts a caller-owned trajectory variable or residual factories and requires a `Problem -> Optimizer` factory that owns the supplied Problem. Contact-force fitting uses a force variable and shared dynamics node, retains coefficient-to-row-weight conversion, and explicitly re-evaluates graph-bearing diagnostics. |
| In-repo consumers | The Panda trajopt example, sparse trajopt benchmark/smoke, vertical-slice support, front-page/docs snippets, generated API pages, contracts, and affected direct tests were migrated to the v2 surface. |
| Deleted implementation | `optim/autograd.py`, `optim/first_order.py`, and `optim/providers.py` are deleted rather than shimmed. Their generated reference pages are deleted; new generated pages cover `optim.optimizers` and `residuals.nodes`. |
| Migration ledger | The root `MIGRATION.md` has a superseding object-referenced rewrite table for removed variables/manifolds, providers/context, residual items/adapters, public LM state/lifecycle, first-order function API, tangent-autograd wrappers, external parameters, and scattered weights. |

## Verification

| Gate | Final result |
|---|---|
| Full non-bench/non-CUDA suite | 1,574 passed, 2 skipped, 15 deselected, 38 warnings (final post-review run) |
| Pinocchio parity | 135 passed |
| Contracts | 352 passed |
| Documentation tests | 26 passed |
| Sparse trajopt benchmark smoke | 4 passed |
| Sphinx HTML | Succeeded; 4 offline intersphinx DNS warnings |
| Sphinx doctest | 19 tests, 0 failures; 4 offline intersphinx DNS warnings |
| Full-repo Ruff | 140 findings, exactly the recorded baseline debt; not a green full-repo gate |
| Changed-file Ruff | All 93 changed/new Python files passed |
| Full-repo Ruff format | 55 untouched baseline files remain unformatted; not a green full-repo gate |
| Changed-file Ruff format | All 93 changed/new Python files already formatted |
| Legacy production-source surface greps (excluding caches) | Zero matches |
| CUDA-marked suite | Owner-run; not claimed here |

## Line accounting

Method: raw physical lines from `wc -l`, matching the pre-implementation
budget method.

| Module | Before | After | Actual delta | Target |
|---|---:|---:|---:|---:|
| `__init__.py` | 68 | 69 | +1 | 72 |
| `_solver_common.py` | 41 | 44 | +3 | 35 |
| `autograd.py` | 83 | 0 | -83 | 0 |
| `first_order.py` | 96 | 0 | -96 | 0 |
| `implicit.py` | 477 | 445 | -32 | 465 |
| `kernels.py` | 139 | 107 | -32 | 105 |
| `lm.py` | 935 | 961 | +26 | 960 |
| `manifolds.py` | 223 | 34 | -189 | 55 |
| `optimizers.py` | 0 | 251 | +251 | 200 |
| `problem.py` | 621 | 585 | -36 | 600 |
| `providers.py` | 105 | 0 | -105 | 0 |
| `solvers.py` | 295 | 362 | +67 | 340 |
| `temporal.py` | 440 | 440 | 0 | 445 |
| `variables.py` | 219 | 479 | +260 | 450 |
| **Total** | **3,742** | **3,777** | **+35** | **3,727 planned; at most 3,800 hard** |

The implementation is 50 lines above the planned 3,727 total but 23 lines
under the hard 3,800-line ceiling. Per-module targets were estimates rather
than individual gates; the table makes their misses visible.

| Scope | Before | After | Delta | Required result |
|---|---:|---:|---:|---|
| `src/better_robot/residuals/*.py` | 2,676 | 2,675 | -1 | net at most +0 — met |
| `src/better_robot/tasks/*.py` | 1,710 | 1,571 | -139 | net at most +0 — met |

## Deleted-test disposition

### Deleted provider suite

`tests/optim/test_providers.py` is deleted because the provider/context graph
it tested no longer exists. Its behavioral coverage was disposed as follows:

| Deleted test | Disposition |
|---|---|
| `test_provider_runs_once_per_evaluation_across_three_residuals` | Ported to `test_explicit_node_memo_is_once_per_epoch_and_released_afterward`, the scene-state sharing test, and the IK FK-count test. |
| `test_python_zero_weight_keeps_lazy_provider_inactive` | Ported to `test_inactive_residual_does_not_evaluate_its_node`. |
| `test_provider_cycle_error_is_deterministic_and_exact` | Deleted with recursive name-resolved provider DAGs. Nodes are explicit object calls; no cycle-resolution API remains. |
| `test_problem_rejects_unknown_provider_reads_and_item_reads` | Deleted with string `reads`; construction/freeze now validates object references and names. |
| `test_provider_may_read_an_undeclared_dependency` | Deleted with declaration policing; node variables are explicit and harvested transitively. |
| `test_problem_rejects_context_and_item_name_collisions` | Context namespaces no longer exist. Duplicate residual/variable names remain covered by `test_duplicate_names_are_rejected_when_problem_freezes`. |
| `test_robot_state_provider_runs_fk_once_per_evaluation` | Ported to `test_robot_state_nodes_merge_by_variable_and_model_identity` and `test_kinematic_residuals_share_one_fk_per_evaluation`. |
| `test_evaluation_context_does_not_escape_problem_or_residual` | Ported to node memo release, standalone freshness, and nested-scope tests in `test_nodes_v2.py`. |
| `test_detach_values_returns_graph_free_artifacts` | The helper is removed. Detached default optimizer output is covered by `test_public_optimize_is_detached_by_default_and_implicit_only_by_opt_in`; value ownership is covered by variable/LM v2 tests. |
| `test_transitive_provider_dependencies_drive_jacobian_structure` | Ported to `test_node_variables_drive_transitive_jacobian_structure`. |

### Renamed first-order suite

`tests/optim/test_solver_adam_matrix_free.py` is deleted and replaced by
`tests/optim/test_torch_optimizer.py` plus broader v2 optimizer suites.

| Deleted test | Disposition |
|---|---|
| `test_generic_torch_optimizer_factories_solve_without_jacobians` | Rewritten as `test_generic_torch_optimizer_classes_solve_without_jacobians`; the class/factory contract is also covered in `test_optimizers_v2.py` and `test_torch_optimizer_v2.py`. |
| `test_optimizer_and_leaf_buffers_persist_while_batch_elements_stop_independently` | Ported as `test_optimizer_and_buffers_persist_while_batch_elements_stop_independently`. |
| `test_retraction_enforces_masks_bounds_and_manifold_projection` | Ported as `test_retraction_enforces_masks_bounds_and_group_projection` using variable-owned geometry. |

### Other removed, merged, or renamed test identifiers

| Previous test | Current disposition |
|---|---|
| `test_exceptions_module_symbols_have_docstrings` | Generalized into parameterized `test_public_submodule_symbols_have_docstrings`; `test_public_type_alias_metadata_is_valid` was added for aliases. |
| `test_residual_instances_satisfy_protocol` | Replaced by `test_residual_is_an_abstract_base_class` and parameterized `test_residual_instances_satisfy_abc`. |
| `test_named_block_update_modules_are_watched` | Renamed to `test_optimizer_update_modules_are_watched` and retargeted to `optimizers.py` plus the private LM iteration. |
| `test_root_se3_remains_the_lie_type_after_named_block_freeze` | Renamed to `test_root_se3_remains_the_lie_type_after_optim_v2`. |
| `test_named_block_api_is_qualified_and_has_no_lie_name_collision` | Renamed to `test_optim_v2_api_is_qualified_and_has_no_lie_name_collision`; retired-symbol coverage was split into `test_retired_optim_api_stays_removed`. |
| `test_multi_axis_batched_jacobian_matches_sequential` | Folded into the `batch_shape` parameterization of `test_batched_jacobian_matches_sequential`. |
| `test_shared_external_vector_is_not_misread_as_a_batch_axis` | Renamed/reworked as `test_shared_static_vector_is_not_misread_as_a_batch_axis`. |
| `test_batched_tensor_residual_weights_scale_each_jacobian` | Renamed/reworked as `test_batched_tensor_weights_scale_each_jacobian`. |
| `test_public_solve_is_detached_by_default_and_implicit_only_by_opt_in` and `test_public_solve_runtime_annotations_resolve` | Renamed to the corresponding `test_public_optimize_*` tests. |
| `test_implicit_solve_rejects_optimized_external_parameter_identity_collision` | Reworked as `test_object_owned_solve_rebases_shared_initial_and_static_tensor_roles`. |
| `test_declared_differentiable_parameter_must_reach_terminal_optimality` | Reworked as `test_graph_carrying_static_variable_must_reach_terminal_optimality`. |
| `test_robot_config_checks_every_quaternion_event_for_log_branch_cut` and `test_robot_config_nq_ne_nv_maps_ambient_output_cotangent` | Renamed to the corresponding `test_robot_variable_*` geometry tests. |
| `test_identity_only_item_weight_binding_is_rejected_as_disconnected` | Renamed to `test_identity_only_weight_binding_is_rejected_as_disconnected`. |
| `test_panda_robot_config_wrapper_parity`, `test_floating_spherical_robot_config_wrapper_parity`, and `test_feasible_retraction_clamps_only_box_coordinates_and_preserves_units` | Rewritten as typed-variable parity/retraction tests. |
| `test_group_bounds_raise_the_exact_global_box_error` and `test_robot_config_rejects_bounds_on_quaternion_coordinates_exactly` | Rewritten as `test_group_bounds_raise_actionably` and `test_robot_variable_rejects_bounds_on_quaternion_coordinates`. |
| `test_value_validation_checks_structure_without_scanning_manifold_content` | Rewritten as `test_value_validation_is_structural`. |
| `test_external_tensor_parameters_are_enumerated_and_graph_visible` | Rewritten as `test_static_variables_are_harvested_and_graph_visible`. |
| `test_unused_gradient_block_stays_connected_to_declared_external_parameter` | Rewritten as `test_unused_gradient_block_stays_connected_to_static_variable`. |
| `test_reads_declare_structure_without_policing_access` | Deleted with string `reads`; direct variable/node references are now the structure. |
| `test_retraction_rejects_a_mixed_dtype_step` and `test_difference_validates_both_inputs_and_rejects_mixed_dtype` | Merged into `test_retraction_and_difference_reject_mixed_dtype`. |
| `test_warm_start_rejects_damping_dtype_mismatch_at_the_boundary` | Deleted because an optimizer owns one fixed Problem; changing the whole working dtype requires a new graph/optimizer. `Problem.update` still rejects incompatible mixed working types. |
| `test_solver_hyperparameters_and_state_are_frozen_tensor_values` | Public `LMState` field assertions were explicitly authorized for deletion. Optimizer control validation and private-state structure are covered elsewhere. |
| `test_update_is_pure_and_preserves_state_structure` | Renamed to `test_private_update_is_pure_and_preserves_state_structure`; the numerical pure step remains covered. |
| `test_external_update_loop_is_identical_to_run` | Deleted with the public `init_state/update/finalize/run` lifecycle. `step()`/`optimize()` behavior and the private step are covered by LM v2 and pattern tests. |
| `test_detached_run_never_appears_to_support_unrolled_backpropagation` | Rewritten as `test_detached_optimize_never_appears_to_support_unrolled_backpropagation`. |
| `test_varspec_scale_sets_scaled_mu_and_balances_mixed_unit_step` | Renamed to `test_variable_scale_sets_scaled_mu_and_balances_mixed_unit_step`. |
| `test_item_kernel_overrides_solver_default_per_semantic_group` | Reworked as `test_item_kernels_apply_per_semantic_group`; kernels now live on residuals and there is no optimizer-default kernel. |
| `test_tangent_grad_returns_only_free_masked_coordinates` | Rewritten as `test_retracted_gradient_returns_only_free_masked_coordinates`. |
| `test_velocity_zero_on_constant_trajectory` | Its zero-value assertion survives inside `test_velocity_preserves_arbitrary_batch_axes`. |
| `test_acceleration_analytic_jacobian_matches_fd` and `test_velocity_analytic_jacobian_matches_fd` | Merged into parameterized `test_analytic_jacobian_matches_tangent_finite_difference`. |
| `test_autograd_through_difference` | Narrowed/renamed to `test_autograd_through_acceleration_difference`; constructor structure coverage was added separately. |
| `test_static_horizon_validation_is_eager_and_required` | Split into `test_contact_horizon_validation_is_eager` and `test_temporal_horizon_validation_is_eager_and_variable_owned`. |
| `test_time_indexed_without_analytic_inner_falls_back_to_ad` | Reworked as `test_custom_time_local_residual_falls_back_to_ad`. |
| `test_projection_named_target_is_declared_and_graph_visible` and `test_projection_named_observations_validate_runtime_dtype` | Renamed to `test_projection_static_target_is_declared_and_graph_visible` and `test_projection_observations_validate_working_dtype`. |
| `test_one_scene_provider_pass_feeds_all_three_heads_per_problem_evaluation` | Renamed to `test_one_scene_state_pass_feeds_all_three_heads_per_problem_evaluation`. |
| `test_contact_force_provider_is_differentiable_through_fext` | Renamed to `test_contact_force_node_is_differentiable_through_fext`. |
| `test_built_in_kinematic_residuals_share_context_protocol` | Renamed to `test_built_in_kinematic_residuals_share_object_protocol`. |
| `test_facade_declares_and_reads_differentiable_target_parameters` | Renamed to `test_facade_declares_and_reads_static_target_variables`; graph visibility remains asserted. |
| `test_block_kinematic_residuals_share_one_fk_per_context` | Renamed to `test_kinematic_residuals_share_one_fk_per_evaluation`. |

### Rewritten existing test inventory

Every Part-1-modified test or support file is accounted for below. Files
listed as “whole file” retained every current test identifier unless a
deletion/merge is explicitly listed above.

| File | Rewritten coverage |
|---|---|
| `tests/autograd/test_public_differentiability.py` | `test_pose_residual_reaches_q_and_model_values` now uses `RobotVariable` and `PoseResidual.error()`. |
| `tests/bench/bench_trajopt_sparse.py` | Benchmark-only support was rewritten from VarSpec/items/providers/public LM lifecycle to a trajectory variable, residual/node graph, Problem, and object-owned LM steps. |
| `tests/bench/test_trajopt_sparse_smoke.py` | `test_trajopt_sparse_structured_t50_one_update` now accepts the v2 benchmark result schema; definition/error classification tests are retained. |
| `tests/contract/test_docstrings.py` | `test_every_public_symbol_has_a_docstring`, `test_public_submodule_symbols_have_docstrings`, and `test_public_type_alias_metadata_is_valid` cover the changed surface. |
| `tests/contract/test_hot_path_lint.py` | `test_new_forbidden_patterns_are_detected` and `test_optimizer_update_modules_are_watched` track the new module/lifecycle names. |
| `tests/contract/test_pluggable_protocols.py` | `test_linear_solver_protocol`, `test_robust_kernel_protocol`, and `test_kernel_rho_method_exists` were retargeted to LU/current protocols. |
| `tests/contract/test_protocols.py` | `test_protocol_is_runtime_checkable`, residual ABC tests, linear-solver instance tests, and robust-kernel instance tests were rewritten for the new class/protocol split. |
| `tests/contract/test_public_api.py` | `test_root_se3_remains_the_lie_type_after_optim_v2` preserves the root-name collision guard. |
| `tests/contract/test_submodule_public_imports.py` | `test_optim_v2_api_is_qualified_and_has_no_lie_name_collision` and `test_retired_optim_api_stays_removed` pin additions/removals and deleted module paths. |
| `tests/data_model/test_mimic_reduced_map.py` | `test_sign_aware_limits_and_generalized_capacities_are_reduced` now constructs a `RobotVariable` and evaluates `JointPositionLimit.error()`; its reduced-limit assertions are unchanged. |
| `tests/docs/test_front_page.py` | `test_front_page_example_executes_exactly_as_published` executes the v2 graph/optimizer example. |
| `tests/io/test_smpl_like.py` | `test_smpl_rest_residual_backward_is_nan_free` now evaluates an object-owned `RestResidual` through `RobotVariable`; its backward/finite-gradient contract is unchanged. |
| `tests/kinematics/test_jacobians.py` | The four `_analytic_vs_finite_diff` callers and `test_pose_residual_resolves_frame_name` use `RobotVariable`/`Problem` while retaining the exact central-FD epsilons, tolerances, and frame coverage. |
| `tests/optim/slice_support.py` | Shared vertical-slice support was rewritten from providers/context to explicit nodes and object residuals; the exact custom-guide extraction remains. |
| `tests/optim/solver_quality_support.py` | Panda quality fixtures now return variables/problems and construct object-owned pose/limit/rest residuals. |
| `tests/optim/test_accept_icp_config.py` | `test_icp_step_caps_relative_damping_and_external_stop_are_configuration` now drives the private LM step without changing its step-cap/damping assertions. |
| `tests/optim/test_accept_q_extrinsics.py` | `test_joint_q_and_camera_extrinsics_solve_through_public_block_api` now uses direct variables/static extrinsics and object residual dependencies. |
| `tests/optim/test_ad_strategies.py` | All current tests (`test_forced_ad_matches_analytic`, masked columns, both create-graph cases, parameterized batched parity, static-vector batch semantics, tensor weights, explicit FD failure, and graph release) were rewritten over object dependencies. |
| `tests/optim/test_implicit_diff.py` | All current implicit tests were rewritten to `optimize(differentiate="implicit")`, static variables, and variable geometry while preserving closed-form, batched, manifold, bounds, robust-kink, rank, mixed-status, and banded guards. |
| `tests/optim/test_linear_operators.py` | `test_new_solvers_preserve_structural_protocols` now includes the v2 solver family. |
| `tests/optim/test_linear_solvers.py` | Existing Cholesky tests remain; five LU tests cover arbitrary batches, ridge/no mutation, informative per-element failure, validation, and differentiability. |
| `tests/optim/test_manifolds.py` | The whole suite was rewritten around typed variables: round trips, Panda/floating parity, projection/unit coordinates, invalid group bounds, masks, scalar events, dtype, NaN bounds, and structural validation. |
| `tests/optim/test_panda_block_ad.py` | Both Panda pose AD/analytic tests and planar-neutral AD now use `RobotVariable`/object residuals. |
| `tests/optim/test_parametric_model_values.py` | `test_fake_betas_reach_batched_fk_ik_objective_and_rnea` uses an object-owned IK objective. |
| `tests/optim/test_problem_blocks.py` | The whole surviving suite covers line fit, multi-block layouts/batches, masks/scales/normals, offsets, weight/dtype validation, static-variable graph visibility, connected zeros, grouped dimensions, and nonfinite batch rows. |
| `tests/optim/test_problem_robust_gradient.py` | Both grouped-kernel objective/gradient and inactive-kernel tests use residual-owned weights/kernels. |
| `tests/optim/test_solver_lm_bounds.py` | All seven active-set/KKT/batch/robot-bound regressions were migrated to variables and private LM state without relaxing tolerances. |
| `tests/optim/test_solver_lm_damping.py` | Fourteen surviving tests retain exact damping multipliers/clamps, factorization-at-cap, termination, bounded status, scale, robust gain/grouping, nonfinite status, and warm-target refresh against private state/observable values. |
| `tests/optim/test_solver_lm_differentiation.py` | Raw-gradient KKT and exact private-step derivative remain; detached public behavior uses `optimize()`. |
| `tests/optim/test_solver_lm_pattern.py` | Seven surviving tests cover private-step purity/structure, fullgraph compile, leading batches, LM/GN convergence, per-element accept/reject, terminal ride-along, and factorization isolation. |
| `tests/optim/test_solver_lm_routing.py` | All five dense/banded parity, no-dense-call, fallback reason, and explicit linear-solver compatibility tests use the object graph. |
| `tests/optim/test_solver_lm_step_limits.py` | All five immutable/validated block-limit and normal/safeguard step tests use variable names and private LM state. |
| `tests/optim/test_solver_quality.py` | Panda bounded/block/facade, GN monotonicity, robust outlier, and batched-vs-sequential quality anchors retain their prior tolerances through migrated support. |
| `tests/optim/test_tangent_autograd.py` | All six singular-point, masked-gradient, higher-order, connected-zero, spherical, and Panda analytic-parity tests now use public variable/problem hooks. |
| `tests/optim/test_temporal_structure.py` | Temporal validation, cached zero-weight/numeric-block analysis, mask eligibility, dense-vs-structured normal/operator parity, and scaled/restricted band parity use temporal variables. |
| `tests/optim/test_vertical_slice.py` | All four guide/counting, batched parity, shared-parameter Jacobian, and two-phase convergence tests execute the node/object graph. |
| `tests/residuals/test_chamfer.py` | All five value/batch/shape/selected-gradient/NaN-mask tests now construct a source variable directly. |
| `tests/residuals/test_projection.py` | All ten value/batch/analytic/gradcheck/clamp/invalid/static-target/dtype/grouped-kernel tests use robot/static variables and shared RobotState. |
| `tests/residuals/test_rotation_prior.py` | All four weight/tangent/AD/finite-fp32 tests use `RobotVariable` and residual methods. |
| `tests/residuals/test_scene_sdf.py` | All five value, once-per-evaluation, selected-gradient, NaN-mask, and forward/reverse AD tests use `SceneSDFState` and Problem epochs. |
| `tests/residuals/test_smoothness.py` | Six tests cover zero/nonzero acceleration, arbitrary-batch velocity and weight, parameterized analytic-vs-FD, autograd, and constructor-owned trajectory validation. |
| `tests/residuals/test_swing_twist.py` | All five value/batch/tangent-FD/pi-convention/constructor tests use object-owned robot state. |
| `tests/residuals/test_temporal_structure.py` | Nine tests cover smoothness/reference, masks, create-graph, time indexing, contact dense/FD blocks, split horizon validation, and custom AD fallback through temporal variables/nodes. |
| `tests/tasks/test_contact_forces.py` | Node differentiability and force/torque smoothness tests were rewritten; solve quality, graph preservation, batches, validation, and gravity coverage remain. |
| `tests/tasks/test_ik_block_rebase.py` | Object protocol, shared RobotState, no-legacy import, static targets/rest, active bounds, and FD configuration tests were migrated. |
| `tests/tasks/test_ik_regression.py` | Floating-base joint-limit Jacobian construction moved to `RobotVariable`; all task quality/optimizer/batch/differentiable regressions remain. |
| `tests/tasks/test_solve_ik_fk_count.py` | `test_kinematic_residuals_share_one_fk_per_evaluation` replaces the context/provider wording and preserves exact FK call counts. |
| `tests/tasks/test_trajopt_named_blocks.py` | Factory ownership was added; dense/structured parity, bounds, leading batches, quaternion alignment, fallback, and B-spline rejection run through residual factories/trajectory variables. |
| `tests/tasks/test_trajopt_param.py` | Knot solve and three unsupported B-spline cases now pass residual factories instead of items. |

### Added test inventory

| Added file | Tests and purpose |
|---|---|
| `tests/optim/test_variables_v2.py` | Six tests for plain/typed batch ownership, geometry round trips, robot limits/temporal layout, constructor metadata validation, group-bound rejection, and auto names. |
| `tests/optim/test_residual_base_v2.py` | Seven tests for scale/diagonal row parity, Python-vs-tensor zero, both adapter forms, `Difference`, grouped dimensions, and working-type/shape validation. |
| `tests/optim/test_problem_v2.py` | Ten tests for harvesting/line fit, public geometry hooks, every AD strategy, one-time weights, static columns, atomic update, duplicate names, zero-weight laziness, exception restoration, and freeze. |
| `tests/optim/test_nodes_v2.py` | Seven tests for standalone freshness, once-per-epoch release, nested evaluation depth, inactive laziness, transitive Jacobian dependencies, explicit-identity sharing, and RobotState merging. |
| `tests/optim/test_optimizers_v2.py` | Eight tests for base defaults/info, Adam matrix-free solves, persistent per-element buffers, L-BFGS closure, reset, update-driven layout rebuild, and invalid controls/differentiation. |
| `tests/optim/test_torch_optimizer.py` | Two direct successors for generic optimizer classes and mask/bounds/group retraction. |
| `tests/optim/test_torch_optimizer_v2.py` | Five focused Adam/step/reset/update/L-BFGS/differentiation tests. |
| `tests/optim/test_lm_v2.py` | Six tests for the exact target line-fit example, one-step writeback/minimal info, independent batches, bound status, Problem update, and LU acceptance. |
| `tests/optim/test_implicit_v2.py` | Four tests for the public implicit module surface, static-target gradients, independent batches, and disconnected-static failure. |
| `tests/residuals/test_pose_limits_v2.py` | Six tests for pose/static references, analytic-vs-FD kinematics, knot adapters, position/velocity limits, and invalid frame/trainable target failures. |
| `tests/residuals/test_group_b_v2.py` | Eight tests for chamfer, swing/twist, projection, static targets, scene sharing/non-merging/AD, and contact analytic rows. |
| `tests/residuals/test_temporal_v2.py` | Five tests for smoothness hooks, static reference trajectories, temporal masks, rest references, and placeholder v2 types. |

Tests newly added inside existing files are also accounted for above:
generalized docstring/type-alias contracts, residual ABC contracts, retired
optim import guards, five LU tests, trajopt optimizer-factory ownership,
smoothness constructor validation, and split temporal horizon validation.

## Deviations and findings

1. **The hard size gate passed, but estimates and full-repository style gates
   did not.** `optim/` is 3,777 lines (23 below the 3,800 cap but 50 above the
   planned total); residuals/tasks meet their net-growth caps. All 93 changed
   Python files pass Ruff and formatting, while the repository still has the
   exact 140-finding Ruff baseline and 55 untouched unformatted files. Sphinx
   passed with four offline intersphinx DNS warnings.
2. **IK still rejects both L-BFGS spellings.** Generic `TorchOptimizer`
   implements and tests the summed-objective closure path, but the IK facade
   retains an actionable `NotImplementedError` because L-BFGS has one global
   history and couples batch elements. The diff contains no recorded failed
   IK-quality experiment, although T4 requested that evidence when keeping the
   rejection.
3. **`TimeIndexedResidual` is deliberately narrower than a generic adapter.**
   It wraps only residuals exposing the built-in private knot error/Jacobian
   primitives; custom residuals must accept `knot=` directly.
4. **`JointVelocityLimit` models velocity explicitly.** It accepts a
   velocity-valued `Variable` plus an explicit tensor/static-variable limit;
   it does not derive finite-difference velocity from a robot trajectory.
5. **The sparse benchmark schema moved from 1 to 2.** Retired public LM-state
   diagnostics were removed; route, status, cost, timing, and definition data
   remain. This is an intentional output-schema break caused by the private LM
   state migration.
6. **A private residual-variable helper layer was added.**
   `residuals/_variables.py` centralizes structural protocols and static/direct
   reference collection without creating a lower-layer import of `optim`;
   `Residual` identity-deduplicates the collected references. It is not public
   API.
7. **Shared initial/static tensor identity is now supported.** The obsolete
   role-collision guard was removed: optimizer ownership rebases/detaches the
   trainable value while preserving the static graph edge. The renamed
   implicit-diff regression proves the resulting gradient.
8. **Tolerance defaults intentionally differ.** The abstract `Optimizer`
   default is `1e-8`, while explicit LM/GN construction retains the previous
   `1e-6` default to preserve numerical behavior.
9. **The benchmark mutates `group_size` after residual construction.** The
   assigned values are valid divisors and tests pass, but this bypasses the
   constructor's normal validation and is a localized benchmark compromise.
10. **Vocabulary and global prose cleanup are scheduled for Orders 2/3.**
   Order 01 still contains iterative-optimizer uses of “solver” and stale
   named-block/provider prose in README/architecture/conventions/glossary and
   kinematics guidance. Deferring those sites preserves the user's sequential
   order, but is a deviation from Order 01's literal global truth-sweep wording;
   the sites are follow-up work, not accidental concurrent edits.
11. **Torch optimizer coverage is duplicated.** The mandated replacement
   `test_torch_optimizer.py` exists alongside newly added
   `test_optimizers_v2.py` and `test_torch_optimizer_v2.py`; several Adam,
   reset, update, and L-BFGS assertions overlap. This is not a correctness
   loss.
12. **Node standalone semantics needed a post-migration correction.** The
   initial node implementation retained its memo forever outside `Problem`.
   Active evaluation depth now separates safe within-epoch sharing from fresh
   standalone reads, with nested-scope tests. This is a finding fixed in the
   current diff, not an unresolved deviation.
13. **No parity tolerance or unauthorized layer-dependency contract change was
   observed.** Contract files changed are among those authorized by Order 01,
   and the three late stale-API test rewires above belong to this order.
