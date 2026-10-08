# w1 C `br_kindyn` — report

All 5 items done. Nothing committed.

## What changed (file: change)
1. Compile-safe cache path
   - `src/better_robot/kinematics/_fk_lean.py` `placement_matrices`: when `torch.compiler.is_compiling()` is true,
     return `se3.to_matrix(...)` directly. This runs before the functorch probe, the inference-mode check and the
     `_version`/`object.__setattr__` cache. Docstring updated.
   - `src/better_robot/dynamics/_world.py` `spatial_inertias`: the same guard. Docstring updated.
   - `get_plan` was left alone. The fullgraph test passes without changing it.
2. ADR 0007: new `docs/contracts/decisions/0007-derivatives-contract.md` (Accepted). It covers tangent-space
   partials `(B..., n_out, nv)`, one `<quantity>_derivatives` entry per quantity, autograd first with analytic code
   as a later drop-in behind the same signature, batch first with compile-clean as the route to batch 1, and Warp
   reserved. Linked from `decisions/index.md` (table row) and from a new section at the end of
   `docs/contracts/04-differentiability.md`.
3. `_memo` keys: `docs/contracts/03-state-mutability.md` C-STATE-4 now has a table of the allowed keys
   (`joint_pose`, `frame_pose`, `jacobian_columns`, `world_geometry`) and the rule "add a key to the table in the
   same change". I also fixed a false claim in the old text ("filled only by the state's own methods"):
   `dynamics/state.py:43` and the `kinematics/state.py:268` helper also write to `_memo`. The section now names
   the model-side caches and says when they are bypassed.
4. `src/better_robot/lie/so3.py` `to_euler`: a gimbal-lock branch fires when `|sin_pitch| >= 1 - 4*eps(dtype)`.
   In that branch yaw = 0, pitch = copysign(pi/2) exactly, and roll = atan2(2 qx qw, qw^2 - qx^2) (the combined
   angle). The `torch.where` inputs are masked so gradients stay finite in both branches. Checked: finite
   gradients at the lock, on a random batch and at q=(0,0,1,0). Added `import math`. Docstring updated.
   Before the fix, the fp64 round trip at the lock was wrong (roll 0.245 / yaw -0.245 for a combined angle of 0.8).
5. Benchmarks
   - `pyproject.toml` `[tool.pytest.ini_options]`: added `python_files = ["test_*.py", "bench_*.py"]`, the smaller
     change than renaming. All 5 `bench_*.py` files already set `pytestmark = pytest.mark.bench`, so the default
     `-m "not bench"` run deselects them. `pytest tests/bench -m bench --collect-only` now collects 14 tests.
     Note: `pyproject.toml` is shared with other slices. My edit is one line.
   - `tests/bench/README.md`: removed the section about `bench_trajopt_sparse.py`, which sits in the repo `trash/`
     and no longer in `tests/bench/`. Added a note that `-m bench` collects these files and a `--threads 1` /
     `torch.set_num_threads(1)` note for shared cores.
   - `benchmarks/run.py` already runs on panda/fk/batch 1 (2.1 s), so the `p2_perf` break no longer reproduces
     there. The new smoke test stops it from coming back.

## Tests added
- `tests/kinematics/test_fk_compile.py::test_public_fk_rnea_crba_compile_fullgraph[fk|rnea|crba]`: Panda, float64,
  B=2. Runs `torch.compile(fullgraph=True)` on the public `forward_kinematics_state`, `inverse_dynamics` and
  `mass_matrix` and checks the result with `assert_close` against eager.
- `tests/lie/test_euler_matrix_interop.py`: `test_to_euler_gimbal_lock_puts_combined_angle_in_roll` (fp32/fp64,
  pitch +/-pi/2: pitch and yaw exact, combined roll, quaternion round trip) and
  `test_to_euler_off_gimbal_lock_matches_regular_formula_bitwise` (`torch.equal` on 4096 random quaternions).
- `tests/bench/test_run_smoke.py`: runs `benchmarks/run.py` in a subprocess on one tiny config (cpu, panda, batch
  1, fk, 1 repeat, `--out tmp`). It has no bench marker, so it runs in the default suite.

## Docs touched
`docs/contracts/decisions/0007-derivatives-contract.md` (new), `docs/contracts/decisions/index.md`,
`docs/contracts/04-differentiability.md`, `docs/contracts/03-state-mutability.md`, `tests/bench/README.md`.
`docs/concepts/kinematics-and-jacobians.md` is unchanged because nothing there needed it.

## Gate output (tails)
- Full suite: `1629 passed, 12 skipped, 20 deselected, 33 warnings in 203.20s` (run with the other slices' edits
  in the tree, before the lint-only test edits). The slice tests passed again after those edits: `141 passed,
  1 skipped`.
- `uv run ruff check src tests`: `Found 82 errors.` (the pre-existing count; none added).
- `uv run mkdocs build --strict`: `Documentation built in 3.41 seconds`, no warnings.

## Left / outside the slice
- `mkdocs.yml` nav (not my file) needs one line after the 0006 entry:
  `- 0007 Derivatives contract: contracts/decisions/0007-derivatives-contract.md`. The strict build passes
  without it because a page missing from nav is only INFO, but the page will not appear in the nav.
- I did not re-run the probe with the guards reverted, so I have not confirmed the new fullgraph test fails
  without them. The evidence that it would is p1 F1 and p3c item 2.
- BH: no BH call sites break. `to_euler` returns different values only at the lock, and those were wrong before.
