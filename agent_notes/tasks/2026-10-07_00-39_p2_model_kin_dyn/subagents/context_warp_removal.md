# Context: remove the Warp lanes from BetterRobot (code and tests)

Repo: `/home/rikhat.akizhanov/better/dev/BetterRobot` (git worktree, branch `dev`). Do not `cd` elsewhere.
The owner decided to remove NVIDIA Warp entirely: one torch implementation, differentiable everywhere.
The torch FK was rewritten (`src/better_robot/kinematics/_fk_lean.py`) and is the only lane that stays.

## Rules
- Never delete with `rm`. Remove files with `git rm` (they stay in git history) or the `trash` helper.
- Do not commit, push, or touch `agent_notes/`. Leave the working tree for the main agent to review and commit.
- Do not weaken, skip or xfail any test that is not about Warp. Tests that only test Warp are removed with it.
- Surgical: touch only what Warp removal requires. Match the surrounding style. Ruff line length 120.
- Do not edit `docs/` (another agent does the docs) and do not edit anything under `src/better_robot/optim2`,
  `src/better_robot/terms`, `src/better_robot/kinematics/_fk_lean.py`, `state.py`.
- No new dependencies, no pip installs. Run Python through `uv run`. CPU only: prefix `CUDA_VISIBLE_DEVICES=""`.

## What to remove
1. Files: `src/better_robot/kinematics/_warp_bridge.py`, `_warp_kernels.py`,
   `src/better_robot/dynamics/_warp_bridge.py`, `_warp_kernels.py`, the directory `tests/warp/`,
   `tests/bench/test_m1_warp_benchmark_schema.py`, `tests/bench/baselines/warp_fk_cuda_*.json`,
   `benchmarks/m1_warp_fk_vs_compiled_torch.py`.
2. The `use_warp` parameter and every Warp branch, fallback warning and helper in
   `src/better_robot/kinematics/forward.py` (`forward_kinematics`, `_warn_warp_fallback`, ...) and
   `src/better_robot/dynamics/rnea.py`. After removal `forward_kinematics(model, q, compute_frames=, check_quaternion_norm=, use_compile=)`.
   Remove structure tensors that existed only for Warp, if any (check `src/better_robot/data_model/model_structure.py`
   and `data.py` mentions of warp; keep anything torch code still reads).
3. The `warp` extra in `pyproject.toml` (and any reference to it in other extras such as `all`). Do not run `uv lock`
   or `uv sync`; if `uv.lock` would need regenerating, say so in your report instead.
4. Mentions in tests that are not Warp tests: `tests/contract/test_optional_imports.py`,
   `tests/data_model/test_layout_aliasing.py`, `benchmarks/baseline.py`, `tests/bench/README.md`,
   `tests/bench/definitions.md`, `tests/bench/external/definition.md` — remove or reword only the Warp parts.
   Leave `tests/optim2/test_problem.py` alone (its mention is a comment about custom kernels in general).
5. `src/better_robot/kinematics/CLAUDE.md`, `src/better_robot/lie/CLAUDE.md`, and the top-level `CLAUDE.md`
   (in this repo root): remove the Warp sentences only.

## Checks before you report
- `grep -rni warp src tests benchmarks pyproject.toml CLAUDE.md` shows only what you deliberately left; list it.
- `uv run ruff check src/ tests/`
- `CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"`
  Expected: exactly one failure, which exists before your change:
  `tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity`.
  The contract tests (`tests/contract/`) include an inventory of `NotImplementedError` raises, a layer-dependency
  check and an optional-import check; they must pass. If a contract test encodes "Warp exists", update that entry.

## Report (final message, short)
Files removed, files edited (one line each: what changed), the grep leftovers, the test summary line,
anything you were unsure about and left alone.
