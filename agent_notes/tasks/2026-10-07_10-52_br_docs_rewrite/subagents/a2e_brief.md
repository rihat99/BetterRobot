# A2e brief — remove the `Data` API from `src/better_robot`

Read `context.md` first (rules: no commits, never `rm` — use `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>`,
CPU only: `CUDA_VISIBLE_DEVICES=""`, never skip/xfail/weaken a test). Then `a2_inventory.md` (the Data -> state mapping and
the kernel keep-list), `a2_state.md` step 4–6, `a2d_state.md` (open items from the test port).

Repo: `/home/rikhat.akizhanov/better/dev/BetterRobot` (git worktree, branch `dev`, HEAD 0e81bbc). All tests, examples,
benchmarks and README already use the state API (`forward_kinematics_state`, `kinematics.state.*`, `dynamics.state.*`).
Nothing outside `src/` still calls `create_data`/`Data`/`rnea`/`crba`/`aba`/`ccrba`/`get_*_jacobian*`. BetterHuman does not
use the Data API either. So this is pure removal inside `src/` plus one small addition.

## Do, in order
1. Trash: `src/better_robot/data_model/data.py`, `data_model/_kinematics_level.py`, `data_model/joint_models/mimic.py`
   (`JointMimic` placeholder — first confirm nothing instantiates it: the only users are `isinstance` guards in
   `io/build_model.py` and `joint_models/{base,composite,__init__}.py`; remove the guards and the guard test, list the test).
2. Remove the Data entry points, keeping the raw kernels the state API uses (keep-list in `a2_inventory.md`):
   `kinematics/forward.py` (move `_validate_q` into `kinematics/state.py`; trash the file if nothing else remains),
   `kinematics/jacobian.py` (keep `*_time_variation_raw` + `_time_variation_in_reference` and their deps; drop
   `compute_joint_jacobians*`, `get_*`), `dynamics/{rnea,crba,aba,centroidal}.py` (drop Data wrappers; keep raw kernels
   that `dynamics/_world.py` / `dynamics/state.py` call — read them first), `exceptions.StaleCacheError`, every Data export
   from `data_model/__init__.py`, `kinematics/__init__.py`, `dynamics/__init__.py`, root `__init__.py` (root
   `center_of_mass` -> `dynamics.state.center_of_mass`). `Model.create_data` and any `Data` mentions in
   `data_model/{model,frame}.py` and `dynamics/_world.py` go too.
   Raw kernels that end up with no caller in `src/`: trash them. `tests/autograd/test_public_differentiability.py` imports
   `rnea/crba/aba/ccrba_raw`, `forward_kinematics_raw`, `frame_placements_raw`, `joint_jacobians_raw`, `frame_jacobian_raw`:
   for a kernel you keep, keep its test; for a kernel you trash, drop that test function and list it (do not keep a dead
   kernel just to keep a test).
3. Add `check_quaternion_norm: bool = False` to `forward_kinematics_state(model, q, *, check_quaternion_norm=False)`
   (same semantics the old `forward_kinematics` had: when True, raise the same exception for a non-unit free-flyer
   quaternion; read the old code before trashing it). Then restore the `pytest.raises` half of
   `tests/kinematics/test_fk_compile.py::test_free_flyer_quaternion_norm_check_is_opt_in` (see git diff of that file at
   HEAD vs `git show 672c9bc^:tests/kinematics/test_fk_compile.py`) and port the two quaternion tests that were dropped
   from `tests/test_exceptions_wired.py` (compare with `git show 672c9bc:tests/test_exceptions_wired.py`).
4. Port the remaining tests that still mention Data: `tests/contract/test_boundary_validation_count.py`,
   `tests/contract/test_naming.py`, `tests/data_model/test_model_data.py`, `tests/io/test_builder_kinds.py`,
   `tests/kinematics/test_state.py`, `tests/test_skeleton_signatures.py`, `tests/bench/conftest.py` (grep for
   `\bData\b|create_data|KinematicsLevel|JointMimic`). Same assertions, same tolerances; a test whose subject no longer
   exists is dropped and listed.
5. `tests/bench/{test_mem_watermark,bench_solve_ik}.py` call `br.solve_ik`, which lives in `better_robot.terms` now:
   fix the import (pre-existing breakage, in scope because you touch bench).
6. Update `src/better_robot/CLAUDE.md`, `src/better_robot/{data_model,kinematics,dynamics}/CLAUDE.md` and
   `README.md`: remove every Data/`create_data`/`rnea`/`crba`/`aba` sentence, describe the state API in the same voice.
   Keep each guide short. Root `__init__` docstring/`__all__` comments: adjust.
7. `docs/concepts/architecture.md` has a `<!-- TODO(api) ... -->` HTML comment about the root `__all__`: resolve it
   against the final `__all__` and delete the comment.

## Gates (all must pass; report the exact tail lines)
```bash
CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider
uv run ruff check src tests   # no NEW errors in files you touched (compare against `git stash`-free: use `git diff --name-only` + ruff on HEAD copies)
uv run mkdocs build --strict
grep -rn "create_data\|StaleCache\|_kinematics_level\|KinematicsLevel\|JointMimic\|\bData\b" src tests examples benchmarks README.md CLAUDE.md docs   # must be empty (docs/concepts/design-decisions.md may mention them as history — that file only)
cd /home/rikhat.akizhanov/better/dev/BetterHuman && CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider   # 1 known pre-existing failure: tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest
```

## Final message (≤ 30 lines)
Files trashed, files edited, tests dropped (name + reason), gate tails, anything left undone. Write the same to
`a2e_state.md` in this folder if you run low on budget before finishing.
