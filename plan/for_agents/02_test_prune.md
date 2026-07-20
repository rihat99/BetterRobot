# Order 02 — Test prune: delete verified duplicates, drop the `_v2` suffix

Read `plan/README.md` and `DESIGN_RULES.md` first. Assumes Order 01 landed
(its feature deletions already removed their attached tests). Every row below
was verified test-by-test in the 2026-07-20 audit (both sides of each
duplicate pair read in full); re-confirm the pair still matches before
deleting — Order 01 may have shifted line numbers.

**Protected — do not touch:** `tests/test_pinocchio/` and the 135 parity
cases; `tests/optim/test_solver_lm_*` numerics; `test_implicit_diff.py`;
all gradcheck/FD-parity tests in kinematics/dynamics/lie/residuals;
`test_layer_dependencies.py`; `test_hot_path_lint.py`;
`test_boundary_validation_count.py`; `tests/warp/`;
`test_vertical_slice.py` + `slice_support.py` (integration value, stays).

## Deletions (verified zero coverage loss)

| File | Action | ~LOC |
|---|---|---:|
| `tests/residuals/test_group_b_v2.py` | Delete whole — every block re-tests a dedicated file (`test_chamfer.py`, `test_swing_twist.py`, `test_projection.py`, `test_scene_sdf.py`, `test_temporal_structure.py`) with identical or superset assertions | 322 |
| `tests/optim/test_torch_optimizer_v2.py` | Delete whole — subsumed by `test_optimizers_v2.py` + `test_torch_optimizer.py` | 97 |
| `tests/optim/test_implicit_v2.py` | Delete whole — 3 of 4 tests subsumed by `test_implicit_diff.py`; the 4th asserts `__all__` trivia | 64 |
| `tests/contract/test_pluggable_protocols.py` | Delete whole — isinstance checks identical to `test_protocols.py`; rho-shape covered by `test_kernel_rho_weight_consistency.py` | 47 |
| `tests/residuals/test_temporal_v2.py` | Delete after folding its two unique tests (ReferenceTrajectory-static-Variable, RestResidual) into `test_temporal_structure.py` | ~65 |
| `tests/optim/test_lm_v2.py` | Delete the three redundant tests (third line-fit copy, bounds→status, batched-converge — covered by `test_solver_lm_*`); keep step/update/LU-era smoke minus what Order 01 removed | ~45 |
| `tests/optim/test_optimizers_v2.py` | Delete `test_adam_is_matrix_free_...` (duplicate of `test_torch_optimizer.py` Adam case, same constants and monkeypatch) | ~24 |
| `tests/optim/test_problem_v2.py` | Delete the duplicate static-variable-harvest test and one line-fit copy (both live in `test_problem_blocks.py`) | ~20 |
| `tests/contract/test_docstrings.py` / `test_public_api.py` | Remove the duplicated top-level `__all__`-docstring check from one of the two | ~15 |
| `tests/contract/test_no_legacy_strings.py` | Delete — `pytest.skip`s unless `BR_STRICT=1`, so it guards nothing in any gate, and the strings it polices are already gone | 37 |

`test_roadmap_stub_inventory.py` **stays** — it keeps the roadmap's
NotImplemented inventory honest for one cheap test.

## Renames (no `_v2` without a v1)

`test_variables_v2.py` → `test_variables.py`; `test_nodes_v2.py` →
`test_nodes.py`; `test_residual_base_v2.py` → `test_residual_base.py`;
`test_pose_limits_v2.py` → `test_pose_limits.py`; `test_optimizers_v2.py` →
`test_optimizers.py`; `test_problem_v2.py` → `test_problem.py`. Rename
`*_v2`-suffixed test identifiers inside surviving files at the same time
(e.g. `test_root_se3_remains_the_lie_type_after_optim_v2`). These files carry
genuine unique coverage — rename, don't delete.

## Acceptance

- Full gate green; report the exact before/after passed-test count and test
  LOC (`wc -l`) in `02_results.md`, with a row per deleted file naming where
  its coverage lives now.
- No new tests added in this order.
- Target: ≥ 700 test LOC removed beyond Order 01's feature-attached tests.
