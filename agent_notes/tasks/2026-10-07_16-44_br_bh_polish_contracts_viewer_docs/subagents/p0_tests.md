# P0 — Test inventory and trim proposal (mars_aux, 2026-10-07 17:10)

**Recommendation: −128 test functions (−22 %), not −40 %.** Pinocchio parity tests skip without Pinocchio
(`tests/test_pinocchio/conftest.py:15`) and cover Panda only (revolute) for FK/integrate/difference; the per-kind
joint tests, finite-difference checks and validation tests are the only guard for spherical, free-flyer, planar,
helical joints.

**Baseline.** 586 `def test_` functions, 1131 items: 1131 passed, 3 skipped, 7 deselected, 185.4 s CPU.
Pinocchio `pin-3.9.0` in `.venv`. 15 slowest are all examples/docs snippets (`test_examples_run[01_ik.py]` 19.1 s,
`test_snippets[visualize.md]` 13.8 s, six examples ≈48 s, `optim/test_robot_terms::test_ik_with_restarts` 7.3 s,
`test_fk_compile::free_flyer…fullgraph` 4.3 s). Unit tests proposed for dropping run in milliseconds.

## Proposal (names without `test_`; "merge" = one parametrized test keeping every case and assertion)
- `io/test_build_model.py` (−13): drop `returns_model`, `joint_count`, `nq_nv`, `joint_placements_shape`,
  `body_inertias_shape`, `q_neutral`, `ir_schema_handshake_is_retired`, `idx_qs` (superseded by
  `idx_qs_dfs_order_on_branching_tree`); merge `revolute_z_mapped_to_jrz`, `revolute_unaligned`, `continuous_joint`,
  `spherical_joint`, `free_flyer_root_joint`, `free_flyer_shortcut_via_load` into one kind-mapping test; merge
  `root_body_inertia`+`child_body_inertia`; merge `ir_frame_added` into `body_frames_created`.
- `io/test_urdf.py` (−12): drop `parse_urdf_returns_irmodel`, `parse_urdf_has_bodies_and_joints`, `panda_njoints`,
  `panda_nbodies_equals_njoints`, `panda_joint_placements_shape`, `panda_body_inertias_shape`,
  `panda_root_joint_is_fixed`, `panda_topo_order_starts_at_universe`; drop duplicates
  `parse_urdf_accepts_yourdfpy_object`, `panda_model_is_frozen`, `panda_topology_parents_before_children`;
  merge `g1_nv` into `g1_nq`.
- `io/test_branching_tree.py` (−9): merge `model_builds`, `root_joint_is_free_flyer`, `has_23_spherical_joints`,
  `njoints`, `nq`, `nv`, `body_names_include_root`, `all_joint_names_present`, `topo_order_valid`,
  `q_neutral_shape` into one structure test. Keep `branching_ir_directly`, `branching_custom_height`.
- `io/test_kinematic_tree.py` (−10): merge 6 input-validation raises into 1 parametrized; merge 6 com/inertia
  mismatch raises into 1.
- `io/test_programmatic.py` (−3): drop `finalize_returns_irmodel`; merge `duplicate_body_raises`,
  `duplicate_joint_raises`, `multiple_root_bodies_raises`.
- `io/test_builder_helpers.py` (−2): drop `add_spherical`, `add_fixed`. `io/test_asset_resolver.py` (−1): drop
  `protocol_runtime_check`.
- `data_model/test_joint_models.py` (−14): drop `revolute_aligned_transform_shape`, `spherical_transform_shape`,
  `revolute_batch`, `spherical_batch`, `free_flyer_batch`; merge the 10 identity/neutral tests into one
  parametrized over joint kinds. KEEP every `*_integrate_difference` test.
- `data_model/test_topology.py` (−7): merge chain/tree pairs per function (`topo_sort`, `build_children`,
  `build_subtrees`, `build_supports`); drop `get_subtree`, `get_support` (superseded by `test_model_data`).
  Note: tests internal `data_model/topology.py` (no underscore).
- `data_model/test_model_data.py` (−1): drop `model_integrate_shape`.
- `lie/test_se3.py` (−6): drop `identity_shape`, `adjoint_shape`, `adjoint_batch`, `apply_base_shape`, `act`,
  `exp_log_roundtrip` (dup of singularities round trip); merge `identity_batch` into `identity`.
- `lie/test_so3.py` (−3): drop `identity`, `act_identity`, `exp_log_roundtrip`.
- `lie/test_types_se3.py`/`test_types_so3.py` (−2): drop `top_level_export_works`, `pose_alias_is_se3`.
- `lie/test_tangents.py` (−3): drop `hat_vee_so3_batch`, `hat_vee_se3_batch`, `right_jacobian_se3_fd`. Keep
  `exact_at_any_angle` (regression pin).
- `lie/test_interpolation.py` (−2): merge `slerp_identical` into `slerp_near_identical`; drop `sclerp_identity_path`.
- `lie/test_torch_backend_singularities.py` (−3): drop `so3_exp_at_zero_is_identity`, `se3_exp_at_zero_is_identity`;
  keep exactly one of the two se3 round trips.
- `spatial/test_spatial.py` (−9): drop `motion_zero`, `force_zero`, `inertia_zero`, `force_add`, `symmetric3_add`,
  `ops_ad_shape`, `inertia_apply_returns_force`; merge three se3-action identity tests.
- `spatial/test_mesh.py` (−2): merge the two bad-faces raises; drop `open_surface_also_rejected_by_reorientation`.
- `kinematics/test_fk.py` (−7): drop `fk_raw_shapes`, `fk_data_shapes`, `fk_batched_shapes`,
  `update_frame_placements_shape`, `panda_fk_neutral_shape`, `panda_fk_dtype_preserved`,
  `panda_fk_universe_is_identity`.
- `kinematics/test_jacobians.py` (−7): drop `compute_joint_jacobians_shape`, `get_frame_jacobian_shape`,
  `chain_jacobian_shape`, `panda_frame_jacobian_shape`, `panda_analytic_vs_finite_diff`,
  `get_frame_jacobian_batched`; merge the two `simple_arm` finite-difference tests.
- `kinematics/test_jacobian_time_variation.py` (−1): drop `batched_shapes`. `kinematics/test_fk_contracts.py` (−1):
  drop `use_compile_matches_eager` (dup of `test_fk_compile.py`).
- `test_skeleton_signatures.py` (−3): drop `classes_are_classes`, `callables_are_callable`,
  `model_has_frozen_dataclass_shape`; rest → `contract/`.
- `test_exceptions.py` (−2): drop `module_accessible_from_top_level`, `errors_carry_messages`.
- `test_exceptions_wired.py` (−5): merge 4 `model_inconsistency_*`; merge `shape_error_on_wrong_nq` with batched
  variant; drop `device_mismatch_error_is_subclass_of_valueerror`.
- Keep unchanged: `optim/*`, `collision/*`, `viewer/*`, `data_model/test_{model_structure,with_values,
  mimic_reduced_map,integrate_vectorized}`, `test_alignment`, `test_torch_backend_gradcheck`.

## Coverage safety net (114 `__all__` symbols grepped against tests + docs)
- No test and no doc mention: `CachedDownloadResolver`, `IRGeom` (io), `FrameType`, `JOINT_DIMENSIONS` (data_model).
- Only contract tests: `Cauchy`, `DARK`, `GemanMcClure`, `Layer`, `Manifold`, `PlotPanel`, `Result`, `Solver`, `State`,
  `Tukey`, `finite_difference`, `register_parser`, `joint_limits` (no docs). None touched by the trim. Re-run
  after applying.

## Structure
- `examples/test_examples_run` ≈48 s overlaps docs snippets → `slow` marker rather than drop.
- `test_types_so3.py`/`test_types_se3.py` → one file parametrized over the group. `test_skeleton_signatures.py` →
  `contract/`. Duplicated Panda fixtures across conftests not inspected.

## BetterHuman (187 functions; by name only)
Mostly substantive. Merge: `viz/test_studio_persistence.py` 5 validation raises → 1 (−4), two builtin
segmentation tests (−1); `io/test_schema.py` `mhr_schema_requires_*` pair (−1); `test_smoke.py` probably superseded
by `contract/test_br_imports.py` + `test_studio_app::imports_without_running_server` (unverified). Family files
(`smpl`, `smplh`, `mano`, `flame`, `smplx`) could share one parametrized family test.
