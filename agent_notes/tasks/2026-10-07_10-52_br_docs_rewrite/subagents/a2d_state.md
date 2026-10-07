# A2d state — Data -> state port of dynamics/data_model/io/viewer/autograd/bench tests, examples, benchmarks, README

Done (no commits). Verify: `CUDA_VISIBLE_DEVICES="" uv run pytest tests/dynamics tests/data_model tests/io tests/viewer
tests/autograd tests/examples -q -m "not bench and not cuda"` -> 331 passed. Examples 07 (--no-viewer) and 08 run;
`benchmarks/run.py` smoke (cpu, panda, batch 1, all 5 algos) has no error rows; `m1_eager_cpu.py --samples 2` runs.
Ruff: no new errors in touched files (all remaining ones pre-existing at HEAD; no --fix applied).

Removed tests / assertions (Data cache or removed type):
- `tests/data_model/test_model_data.py::test_model_create_data`, `::test_model_create_data_batch` (create_data shapes).
- `tests/dynamics/test_batched_values.py::test_dynamics_wrappers_store_the_execution_batch_in_data` (asserted
  `data.q/v/a/batch_shape`; the tau batch shape is still covered by the loop-oracle test).
- `test_batched_values::test_ccrba_returns_frozen_dataclass_fields` -> `test_centroidal_map_and_momentum_shapes`
  (shape checks kept; `CCRBAResult` isinstance + FrozenInstanceError dropped: the type is gone).
- `tests/autograd/test_public_differentiability.py`: `_ccrba` entry dropped (same coverage as `_centroidal_map` +
  `_centroidal_momentum`); `data.joint_jacobians` (all-joint cache) dropped from the workspace FK gradient test.
- `tests/io/test_builder_kinds.py`: three `data.joint_pose_world is not None` lines (always true for the state).

Open items for the src-removal agent:
- `tests/autograd/test_public_differentiability.py` still imports `aba_raw, ccrba_raw, crba_raw, rnea_raw`
  (dynamics) and `forward_kinematics_raw, frame_placements_raw, joint_jacobians_raw, frame_jacobian_raw`
  (kinematics) for its `*_raw` gradient tests. Not Data API, so left alone; if src trashes those kernels,
  trash or port those tests (`test_joint_jacobians_raw_*`, `test_frame_jacobian_raw_*`, `test_raw_dynamics_*`).
- `tests/bench/bench_solve_ik.py` and `tests/bench/test_mem_watermark.py` call `br.solve_ik`, which does not exist
  at HEAD (pre-existing failure, marked bench/slow; fixture renamed `panda_data` -> `panda_state`).
- `bench_jacobian.py` now times `joint_jacobian(last joint, reference="world")` (no all-joints state function).
