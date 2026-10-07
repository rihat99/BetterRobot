# A2e state — Data API removal (PARTIAL: budget ran out; tree is green)

Gate: `CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider`
-> `1161 passed, 3 skipped, 7 deselected, 34 warnings in 161.64s`. Ruff on every touched file: all checks passed.
mkdocs / grep / BetterHuman gates NOT run (nothing in docs or BH touched yet; the grep gate fails until steps 1,2,4,6 are done).

## Done (no commits, nothing trashed)
- Step 3 (src): `kinematics/state.py` now owns `_QUAT_NORM_TOL`, `_validate_q`, `_validate_free_flyer_quaternion_norm`
  (moved verbatim from `forward.py`). `forward_kinematics_state(model, q, *, check_quaternion_norm=False)`; when True it
  calls the free-flyer check after broadcasting q (same place/semantics as old `forward_kinematics`).
  `kinematics/forward.py` re-imports the two validators from `.state` (unused imports removed); `dynamics/_execution.py`
  imports `_validate_q` from `kinematics.state`. `state.py` no longer imports `forward.py`, so `forward.py` can be trashed.
- Step 3 (tests): `tests/kinematics/test_fk_compile.py::test_free_flyer_quaternion_norm_check_is_opt_in` regained its
  `pytest.raises(QuaternionNormError, match="quaternion norm")` half via `forward_kinematics_state(..., check_quaternion_norm=True)`
  (+ `QuaternionNormError` import). `tests/test_exceptions_wired.py`: `test_quaternion_norm_error_on_floating_base` and
  `test_quaternion_norm_accepts_small_drift` now call `forward_kinematics_state` (they still existed at HEAD on the Data API;
  `forward_kinematics` import dropped). `test_fixed_base_bypasses_quaternion_check` still uses `forward_kinematics_raw`.
- Step 5: `tests/bench/{test_mem_watermark,bench_solve_ik}.py` use `from better_robot.terms import solve_ik` and pass the
  now-required batched inputs: `solve_ik(panda, {frame: target[None]}, panda.q_neutral[None])` (solve_ik needs q0 and a
  leading batch axis). Both pass (`2 passed`).

## Not done — steps 1, 2, 4, 6, 7 (findings for the next agent)
- Who imports the Data-era modules in src: `kinematics/__init__` (forward, jacobian), `kinematics/state.py` (only
  `jacobian.{_time_variation_in_reference, frame_jacobian_time_variation_raw, joint_jacobians_time_variation_raw}`),
  `dynamics/__init__` (aba, centroidal, crba, rnea). `dynamics/state.py` imports only `_world`; `_world` imports
  `kinematics._fk_matrix.{matrix_core_with_transforms, prepare_inputs}` and `_jacobian_columns`. So
  `dynamics/{rnea,crba,aba,centroidal,_execution}.py` have no src caller besides `dynamics/__init__` -> trash all five
  (check `_execution` has no other user; `rnea.py` also hosts `bias_forces`, `compute_generalized_gravity` — Data-era).
- Tests still on Data/raw kernels: `tests/contract/test_boundary_validation_count.py` (create_data, forward_kinematics,
  update_frame_placements, dynamics wrappers; monkeypatches `kinematics.forward._validate_q` and `dynamics._execution._validate_q`
  — port to patch `kinematics.state._validate_q` and the state functions); `tests/kinematics/test_jacobians.py` imports
  `kinematics.forward.forward_kinematics_raw` and `kinematics.jacobian.*`; `tests/test_exceptions_wired.py` imports
  `forward_kinematics_raw`; `tests/autograd/test_public_differentiability.py` imports the raw kernels (see a2d_state.md).
  Plus the step-4 list in the brief.
- `JointMimic`: users are `data_model/joint_models/__init__.py:17`, `io/build_model.py` isinstance guard (~line 395),
  `joint_models/{base,composite}.py` ("mimic" kind strings — keep the kind code, only drop the class).
- `data_model/model.py:18,268` (`Data` TYPE_CHECKING import + `create_data`), `data_model/__init__.py:12,14`.
