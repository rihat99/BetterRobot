# A2c state (kinematics + contract tests; stopped on budget)

Done: tests/kinematics/* ported (fk, fk_compile, fk_matrix, fk_regression, _generate_fk_reference, batched_values,
jacobians, jacobian_time_variation, mimic, state); test_public_api, test_skeleton_signatures expected lists = post-removal
surface; test_submodule_public_imports: dropped KinematicsLevel + StaleCacheError rows; test_exceptions_wired ShapeError /
DeviceMismatchError ported. Trashed tests/contract/test_cache_invariants.py (Data.q/v/a cache invalidation only).
Dropped Data-cache tests in test_jacobian_time_variation: test_compute_fills_joint_jacobians_byproduct,
test_velocity_reassignment_invalidates_dot, test_missing_velocity_raises (StaleCacheError). Dropped
`data.joint_pose_local` asserts (test_fk::test_fk_data_shapes, test_batched_values scalar-loop tuple).
Old `data.joint_jacobians(_dot)` -> stack of joint_jacobian(_time_variation)(..., reference="world") over joints.
Verify: 338 passed, 2 skipped (CPU, -m "not bench and not cuda"). Ruff: 9 left, all pre-existing PLC0415.

Owner/main-agent decision needed: QuaternionNormError opt-in (`check_quaternion_norm=True`) exists only on the Data
`forward_kinematics`; forward_kinematics_state has no such option. Removed the raise block in
test_fk_compile::test_free_flyer_quaternion_norm_check_is_opt_in (kept the "trusts input" half);
test_exceptions_wired::{test_quaternion_norm_error_on_floating_base, test_quaternion_norm_accepts_small_drift} still call
`forward_kinematics` -> fail after removal unless the option is added to the state API or the tests/exception are dropped.
test_fk_matrix::test_use_compile_matches_eager now compares torch.compile(forward_kinematics_state) vs eager (use_compile
lane gone).

Not done:
1. tests/contract/test_boundary_validation_count.py — still on rnea/aba/crba/ccrba/compute_centroidal_*/
   compute_generalized_gravity/forward_kinematics/create_data; needs re-counting validations per state entry point.
2. tests/contract/test_naming.py (scans for deprecated Data.<field> names; decide if still meaningful once Data is gone)
   and test_docstrings.py — not checked.
3. test_submodule_public_imports rows for *_raw kernels (joint_jacobians_raw, frame_jacobian_raw, rnea_raw, crba_raw,
   ccrba_raw, ...) and test_jacobians::test_joint_jacobian_plan_rebuilds_after_to (now on joint_jacobians_raw) depend on
   which raw kernels the removal agent trashes.
Backup of pre-port kinematics tests: scratchpad/kin_backup/.
