# A2b state (stopped on budget; 50k-token agent) — continue from here

Read a2_inventory.md for the Data -> state mapping. Nothing removed from src yet; tree green.

Done in A2b (all 7 remaining `tests/test_pinocchio` files ported, same tolerances):
- `test_aba_matches_pinocchio`, `test_crba_matches_pinocchio`, `test_rnea_matches_pinocchio`,
  `test_rnea_advanced_joints`, `test_rnea_missing_joint_kinds`, `test_dynamics_derivatives`, `test_centroidal`.
  `uv run pytest tests/test_pinocchio` -> 183 passed. Ruff clean on these (1 pre-existing PLC0415 in aba).
- Dropped Data-cache assertions (list for the report):
  - `test_rnea_matches_pinocchio.py::test_rnea_populates_data_fields` (whole test: only `data.tau`,
    `joint_velocity_local`, `joint_acceleration_local`, `joint_forces`, `joint_pose_world/local` populated).
  - `test_rnea_matches_pinocchio.py::test_rnea_bias_forces_matches_rnea_zero_accel`: 2 lines asserting
    `d1.bias_forces` cache == returned tau (the tau comparison at atol=1e-12 is kept; now near-tautological
    because `bias_forces` is `inverse_dynamics(..., zeros)`).
  - `test_rnea_advanced_joints.py` spherical-chain test: 3 lines asserting `data.joint_velocity_local/
    joint_acceleration_local/joint_forces` shapes (the `tau.shape == (4,)` check is kept).
- `test_centroidal::test_hg_linear_equals_mass_times_vcom`: `data.com_velocity` replaced by
  `torch.func.jvp` of `center_of_mass(model, forward_kinematics_state(model, q))` along v (Panda: dq/dt = v);
  same 1e-10 tolerance, passes. h_g from `centroidal_momentum` (linear rows first, as the old test assumed).
- Docstrings of the 3 earlier-ported Jacobian parity files renamed `get_*_jacobian*` -> `*_jacobian*`.
- Regex porting helper (reusable): `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/port.py`
  (rewrites `rnea/br.rnea/br_rnea`, `bias_forces`, `crba`, `aba` calls with `data=` kwargs into state calls,
  drops `x = m.create_data(...)` lines; run `ruff check --fix` only AFTER all manual edits, or it strips the
  new imports before they are used).

JointMimic finding: nothing in src instantiates it. Users: `joint_models/__init__.py` export,
`io/build_model.py:19,395` (isinstance guard that rejects it), `joint_models/composite.py:36` (nesting guard),
`tests/io/test_builder_kinds.py:16,234` (instantiates it only to assert build_model rejects it).
Removing it = trash `joint_models/mimic.py`, drop both guards and that test case (a guard test, not a cache test).

Remaining, in order (files still using the Data API, from grep):
1. tests: `test_exceptions_wired.py`, `test_skeleton_signatures.py`, `contract/{test_submodule_public_imports,
   test_cache_invariants (trash if cache-only), test_public_api, test_boundary_validation_count, test_naming}`,
   `contract/test_docstrings.py` (check), `kinematics/{test_fk, test_fk_compile, test_fk_matrix, test_fk_regression,
   _generate_fk_reference, test_batched_values, test_jacobians, test_jacobian_time_variation, test_mimic, test_state}`,
   `dynamics/{test_batched_values, test_mimic}`, `data_model/test_model_data.py` (trash create_data shape checks),
   `viewer/test_robot.py`, `io/{test_preserve_order, test_builder_kinds}`, `autograd/test_public_differentiability.py`,
   `bench/{conftest, bench_forward_kinematics, bench_jacobian, README.md}`.
2. `examples/07_branching_tree_body.py`, `examples/08_dynamics.py`, `examples/viewer/*` (check),
   `benchmarks/{run.py, m1_eager_cpu.py}`.
3. src removal (step 4 of a2_state.md), unchanged.
4. Root `CLAUDE.md`: DONE (the two "Model/Data APIs" / "mutable Data" sentences now describe
   `forward_kinematics_state` + frozen `KinematicState`). Note: its Commands block still runs
   `examples/01_ik.py` (legacy example per owner decision; check it still exists). Still to do: `README.md`,
   `src/better_robot/{,data_model,kinematics,dynamics}/CLAUDE.md`.
5. Gates (BR suite, BH suite, grep, ruff, `mkdocs build --strict`).
