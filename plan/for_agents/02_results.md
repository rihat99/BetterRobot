# Order 02 results — test prune

Status: complete on `dev` (2026-07-20).

## Outcome

- Python test LOC: **22,922 → 22,188** (**−734**, target ≥700).
- Full CPU gate: **1,537 → 1,502 passed** (**−35 verified duplicate
  cases**); the same 2 tests remain skipped and 16 CUDA cases deselected.
- No new test semantics were added. The two unique temporal tests were moved
  into `test_temporal_structure.py` with their assertions preserved.
- All surviving `_v2` test filenames, identifiers, fixture labels, and test
  documentation were renamed.
- Protected parity, solver-numerics, implicit-diff, gradcheck, architecture,
  Warp, and vertical-slice files have zero diff from Order 01.

## Deletion and coverage ledger

| Deleted or pruned content | Net LOC | Cases | Coverage retained in |
|---|---:|---:|---|
| `residuals/test_group_b_v2.py` | −322 | −8 | Dedicated Chamfer, swing/twist, projection, scene-SDF, temporal-structure, and contact tests |
| `optim/test_torch_optimizer_v2.py` | −97 | −5 | `test_torch_optimizer.py` and surviving optimizer contracts |
| `optim/test_implicit_v2.py` | −64 | −4 | `test_implicit_diff.py`; the fourth case was export-list trivia |
| `contract/test_pluggable_protocols.py` | −46 | −8 | `test_protocols.py`, kernel consistency tests, and objective tests |
| `residuals/test_temporal_v2.py` plus folded unique cases | −53 | −2 | Unique static-ReferenceTrajectory and RestResidual cases moved to `test_temporal_structure.py`; duplicate smoothness cases removed |
| `optim/test_lm_v2.py` partial prune | −41 | −3 | `test_problem_blocks.py`, `test_solver_lm_bounds.py`, and LM pattern tests |
| `optim/test_optimizers_v2.py` partial prune | −26 | −1 | Adam matrix-free behavior in `test_torch_optimizer.py` |
| `optim/test_problem_v2.py` partial prune | −37 | −2 | Line fit and static-variable harvesting in `test_problem_blocks.py` |
| `contract/test_public_api.py` duplicate check | −11 | −1 | Richer centralized checks in `test_docstrings.py` |
| `contract/test_no_legacy_strings.py` | −37 | −1 | Guard was non-failing on violations outside `BR_STRICT`; scoped source grep is clean |
| **Total** | **−734** | **−35** | |

Suite LOC deltas reconcile exactly: optimizer tests −265, residual tests −375,
and contract tests −94.

## Renames

- `test_variables_v2.py` → `test_variables.py`
- `test_nodes_v2.py` → `test_nodes.py`
- `test_residual_base_v2.py` → `test_residual_base.py`
- `test_pose_limits_v2.py` → `test_pose_limits.py`
- `test_optimizers_v2.py` → `test_optimizers.py`
- `test_problem_v2.py` → `test_problem.py`
- `test_lm_v2.py` → `test_lm.py`

## Verification

- Full CPU gate: **1,502 passed, 2 skipped, 16 deselected**.
- CUDA gate on GPU 2: **15 passed, 1,505 deselected**.
- Pinocchio parity: **135 passed**.
- Contract suite: **334 passed**.
- Documentation tests: **27 passed**; Sphinx doctest: **30 passed**.
- Changed test files pass Ruff and format checks; `git diff --check` passes.

Sphinx's four warnings were the same unreachable external intersphinx
inventories seen in Order 01; no documentation test failed.

## Deviations and findings

- The explicit rename list omitted the surviving `test_lm_v2.py`, while the
  order title and rule require no `_v2` suffix without a v1. It was renamed to
  `test_lm.py`; this is the only plan deviation.
- The temporal-file net reduction was 53 rather than the approximate 65 lines
  because preserving the two unique cases required 27 lines in their existing
  destination. The overall hard target was still exceeded by 34 lines.
