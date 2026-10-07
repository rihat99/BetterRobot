# A2 state (stopped on budget) — read a2_inventory.md for the mapping

Done (no src removal yet; Data API still present so the tree stays green):
- Added to `kinematics/state.py`: `joint_jacobian`, `frame_jacobian_time_variation`, `joint_jacobian_time_variation`;
  `frame_jacobian` now shares private `_world_jacobian` + `_express` with `joint_jacobian` (same math, moved).
- Added to `dynamics/state.py`: `centroidal_momentum(model, state, v)`.
- Exported all of them from `kinematics/__init__`, `dynamics/__init__` and root `__init__` (additive).
- Ported (same tolerances, all green): `tests/test_pinocchio/{test_joint_jacobian, test_frame_jacobian,
  test_jacobian_time_variation, test_fk}_matches_pinocchio.py` (78 tests).
- Check run: `tests/test_pinocchio tests/kinematics tests/dynamics tests/contract tests/test_skeleton_signatures.py`
  -> 538 passed, 1 skipped. Ruff clean on touched files (pre-existing PLC0415/F401 in untouched test_pinocchio files).

Remaining, in order:
1. test_pinocchio: `test_aba_matches_pinocchio`, `test_crba_matches_pinocchio`, `test_rnea_matches_pinocchio`,
   `test_rnea_advanced_joints`, `test_rnea_missing_joint_kinds`, `test_centroidal`, `test_dynamics_derivatives`.
   Test functions that only assert Data cache population (`data.joint_forces`, `joint_velocity_local`,
   `joint_acceleration_local`, `joint_pose_local`, `data.tau`) are Data-cache tests -> drop and list.
   `data.com_velocity` vs Pinocchio vcom: port via `centroidal_momentum` linear rows / mass (verify row order).
2. Other tests: `test_exceptions_wired.py`, `test_skeleton_signatures.py`, `contract/{test_submodule_public_imports,
   test_cache_invariants (likely trash: Data cache), test_public_api, test_boundary_validation_count, test_naming,
   test_docstrings}`, `kinematics/{test_fk, test_fk_compile, test_fk_matrix, test_fk_regression, _generate_fk_reference,
   test_batched_values, test_jacobians, test_jacobian_time_variation, test_mimic, test_state}`, `dynamics/{test_batched_values,
   test_mimic}`, `data_model/test_model_data.py` (create_data shape checks -> trash those), `viewer/test_robot.py`,
   `io/{test_preserve_order, test_builder_kinds}`, `autograd/test_public_differentiability.py`,
   `bench/{conftest, bench_forward_kinematics, bench_jacobian, README.md}`.
3. Examples `07_branching_tree_body.py`, `08_dynamics.py`; `benchmarks/{run.py, m1_eager_cpu.py}`
   (`benchmarks/results/*.json` are historical outputs: leave).
4. Then remove (trash script): `data_model/data.py`, `_kinematics_level.py`, Data exports, Data entry points in
   `kinematics/{forward,jacobian}.py`, `dynamics/{rnea,crba,aba,centroidal}.py`, `StaleCacheError` (only raiser is
   `kinematics/jacobian.py:421`), unused raw kernels (list in inventory). Move `forward._validate_q` into `state.py`.
   Swap root `center_of_mass` to `dynamics.state.center_of_mass`; update root `__all__` comments + DAG docstring.
   `JointMimic`: `io/build_model.py:395` has an isinstance check — confirm nothing instantiates it before trashing.
5. CLAUDE.md updates: root `CLAUDE.md` still says "same `Model`/`Data` APIs" and "mutable `Data` belongs to one query";
   `src/better_robot/CLAUDE.md`, `{data_model,kinematics,dynamics}/CLAUDE.md`, `README.md` not touched yet.
6. Final gates: BR suite, BH suite, grep for `create_data|StaleCache|_kinematics_level|\bData\b`.
