# A2e-3 state — partial port (budget ran out); tree still RED in tests/

## Done (green when run alone)
- contract/test_boundary_validation_count.py: ported. Patches `kinematics.state._validate_q`; counts
  forward_kinematics_state = 1, each dynamics.state fn on a precomputed state = 0,
  inverse_dynamics(forward_kinematics_state(q)) = 1; ModelValues.validate still 0. 1 passed.
- autograd/test_public_differentiability.py: dropped test_fk_and_frame_raw_blocks_reach_q_and_model_values,
  test_joint_jacobians_raw_reaches_q_and_model_values, test_raw_dynamics_reaches_q_and_body_inertias[4]
  (raw kernels gone; public equivalents already tested in the same file); test_frame_jacobian_raw_* ported to
  br.frame_jacobian (renamed test_frame_jacobian_reaches_q_and_model_values, still checks frame_placements grad). 9 passed.
- data_model/test_model_data.py: dropped test_data_batch_shape, test_data_reset, test_data_clone (Data gone). 11 passed.
- test_exceptions_wired.py: test_fixed_base_bypasses_quaternion_check now calls
  forward_kinematics_state(arm_model, q, check_quaternion_norm=True). Passes.
- kinematics/test_jacobian_time_variation.py: raw FK -> forward_kinematics_state(...).joint_pose; the reused
  joint_jacobians stack now built from joint_jacobian(..., reference="world") per joint. Passes.
- kinematics/test_fk_compile.py: import fixed; opt-in test passes.

## Remaining red (exact)
1. kinematics/test_fk_compile.py::test_free_flyer_fk_state_compiles_fullgraph FAILS: forward_kinematics_state does
   not compile with fullgraph=True (dynamo Unsupported). Subject (raw FK fullgraph) is gone. Options: compile
   `_fk_lean.lean_core` instead, or drop the test. NOT weakened to fullgraph=False.
2. io/test_builder_kinds.py: collection error (imports JointMimic). Dropping
   test_direct_joint_mimic_requires_a_concrete_target_kind + the import was DENIED by the auto-mode classifier
   ("Security Test Removal") — needs the user's/parent's call. Untouched.
3. kinematics/test_fk.py: imports fixed, raw calls ported to forward_kinematics_state/.joint_pose. Still references
   `joint_pose_local` (lines ~50, ~76: field does not exist on KinematicState) and
   test_frame_placements_raw_matches_public_fk (frame_placements_raw gone -> drop). Not run.
4. kinematics/test_jacobians.py: imports forward_kinematics_raw, joint_jacobians_raw, frame_jacobian_raw; tests
   test_joint_jacobians_raw_matches_workspace_pass, the plan-reuse test (~line 103), test_frame_jacobian_raw_matches_
   workspace_pass, test_frame_jacobian_raw_compiles_fullgraph target removed kernels. Untouched.
5. kinematics/test_fk_matrix.py: every test targets forward_kinematics_raw/_fk_matrix -> trash per brief. Untouched.
6. contract/test_submodule_public_imports.py lines ~136-176: expected lists still name forward_kinematics_raw,
   joint_jacobians_raw, frame_jacobian_raw; update to current kinematics/dynamics `__all__`. Untouched.
7. test_skeleton_signatures.py::{test_data_has_core_fields,test_data_legacy_aliases_are_removed}: drop. Untouched.
8. contract/test_roadmap_stub_inventory.py + docs/reference/roadmap.md JointMimic row. Untouched.
9. Grep-gate comment leftovers: contract/test_naming.py:4,59; kinematics/test_state.py:1; test_pinocchio/
   test_rnea_matches_pinocchio.py:119-120; test_crba_matches_pinocchio.py:5; bench/conftest.py:1. Untouched.
Gates: not run (full suite, ruff, mkdocs, grep, BetterHuman).
Note: `center_of_mass` is not re-exported from `better_robot.dynamics` (only from dynamics.state and root).
