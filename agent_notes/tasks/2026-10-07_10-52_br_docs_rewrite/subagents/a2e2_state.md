# A2e-2 state — src removal DONE, tests NOT ported (tree is RED in tests/)

## Done in src (no commits)
Trashed (to `<repo>/trash/`): `kinematics/forward.py`, `kinematics/_fk_matrix.py` (its only remaining user was
`_world.world_kinematics_from_q`, itself Data-only and removed), `dynamics/{rnea,crba,aba,centroidal,_execution}.py`,
`data_model/{data,_kinematics_level}.py`, `data_model/joint_models/mimic.py`.
Edited:
- `kinematics/jacobian.py`: kept only `JointJacobiansTimeVariationResult`, `joint_jacobians_time_variation_raw`,
  `_time_variation_in_reference`, `frame_jacobian_time_variation_raw` (+ `_ReferenceFrame`); dropped
  `JointJacobiansResult`, `joint_jacobians_raw`, `frame_jacobian_raw`, `compute_*`, `get_*`; docstrings fixed.
- `kinematics/__init__.py`, `dynamics/__init__.py`: export only the state API.
- `dynamics/_world.py`: removed `world_kinematics_from_q` + its `__all__` entry + `_fk_matrix` imports; docstring.
- `data_model/{__init__,model,frame}.py`, `joint_models/{__init__,base,composite}.py` (message now "A mimic joint cannot be
  nested ..."; `kind == "mimic"` check kept), `io/build_model.py` (JointMimic guard + import removed),
  `exceptions.py` (`StaleCacheError` + `__all__` entry removed), `_jacobian_columns.py` docstring.
- Root `__init__.py`: removed `Data`, `forward_kinematics`, `update_frame_placements`, `compute_joint_jacobians`,
  `compute_joint_jacobians_time_variation`, `get_{joint,frame}_jacobian{,_time_variation}`, `rnea`, `aba`, `crba`,
  `compute_centroidal_map`; `center_of_mass` now from `dynamics.state`. `__all__` = 21 names, comment counts fixed
  (data_model 6, kinematics 6, dynamics 6). `import better_robot` works. Ruff: no new errors in touched src files
  (model.py:233 PLC0415 is pre-existing).

## Remaining (tests that still reference removed names; grep gate hits per file)
test_public_differentiability (4: raw kernels all trashed -> drop those test fns unless portable), bench/baselines/
torch_filtered_smpl_b1_rtx6000_ada.json (4, check what string), bench/conftest.py (1), contract/test_boundary_validation_count
(5: port per brief), contract/test_naming (2), contract/test_submodule_public_imports (2), data_model/test_model_data (7),
io/test_builder_kinds (2: JointMimic guard test -> drop), kinematics/test_fk_compile (4), kinematics/test_fk_matrix (7:
`forward_kinematics_raw` in 5 tests -> port to `forward_kinematics_state`), kinematics/test_fk (10),
kinematics/test_jacobians (4), kinematics/test_jacobian_time_variation (5), kinematics/test_state (1),
test_exceptions_wired (2: `test_fixed_base_bypasses_quaternion_check` uses forward_kinematics_raw), test_pinocchio/
test_crba_matches_pinocchio (1), test_rnea_matches_pinocchio (2), test_skeleton_signatures (4).
Also: any test importing `better_robot.kinematics._fk_matrix` or `dynamics.{rnea,crba,aba,centroidal}` (grep).
Gates not run yet. Note: `src/better_robot/{data_model,dynamics,kinematics}/CLAUDE.md` show as modified in git — NOT by
A2e-2 (someone else is editing them concurrently); they mention `_fk_matrix` / `matrix_forward`.

## Gate run after src removal (before any test port)
`pytest tests -q -m "not bench and not cuda" --continue-on-collection-errors` ->
`15 failed, 1003 passed, 2 skipped, 7 deselected, 19 warnings, 10 errors in 159.35s`.
- 10 collection errors: autograd/test_public_differentiability, contract/test_boundary_validation_count,
  data_model/test_model_data, io/test_builder_kinds, kinematics/{test_fk,test_fk_compile,test_fk_matrix,
  test_jacobian_time_variation,test_jacobians}, test_exceptions_wired.
- 15 runtime failures: contract/test_submodule_public_imports (13 params: lists the removed raw names in
  kinematics/dynamics `__all__` -> update its expected lists to the state API), test_skeleton_signatures::
  {test_data_has_core_fields,test_data_legacy_aliases_are_removed} (subject gone -> drop + list),
  contract/test_roadmap_stub_inventory::test_roadmap_inventory_matches_explicit_not_implemented_raises (the JointMimic
  `NotImplementedError` in io/build_model.py was one of the inventoried raises -> remove its row from
  docs/reference/roadmap.md or the test's inventory; main agent's call since it touches docs).
- Not failing but still grep-gate hits: contract/test_naming, kinematics/test_state, test_pinocchio/test_{crba,rnea}
  _matches_pinocchio, bench/conftest.py, bench/baselines/*.json.
