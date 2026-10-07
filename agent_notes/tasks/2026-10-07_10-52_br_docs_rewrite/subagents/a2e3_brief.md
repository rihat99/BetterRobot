# A2e-3 brief — port the tests to the post-removal src (tree is RED now)

Read `context.md` (no commits; never `rm` — `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>`; CPU only
`CUDA_VISIBLE_DEVICES=""`; never skip/xfail/weaken a test — same assertions, same tolerances; a test whose subject no
longer exists is dropped and listed). Then `a2e2_state.md` (what was removed from src and which tests break),
`a2_inventory.md` (old -> new mapping). Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`, branch `dev`.
src is final: the state API only (`kinematics.{forward_kinematics_state, KinematicState, frame_jacobian, joint_jacobian,
frame_jacobian_time_variation, joint_jacobian_time_variation}`, `dynamics.{inverse_dynamics, mass_matrix,
forward_dynamics, center_of_mass, centroidal_map, centroidal_momentum}`), raw kernels left: `kinematics/_fk_lean.py`,
`_jacobian_columns.py`, `jacobian.py::{joint_jacobians_time_variation_raw, frame_jacobian_time_variation_raw}`,
`dynamics/_world.py`. Do NOT edit src except to fix a real bug you find (say so) — and do not edit CLAUDE.md/README/docs.

## Work list (from a2e2_state.md), in order; run each file as you go
1. Collection errors (10 modules): `tests/autograd/test_public_differentiability.py`,
   `tests/contract/test_boundary_validation_count.py`, `tests/data_model/test_model_data.py`,
   `tests/io/test_builder_kinds.py`, `tests/kinematics/{test_fk,test_fk_compile,test_fk_matrix,
   test_jacobian_time_variation,test_jacobians}.py`, `tests/test_exceptions_wired.py`.
   Most of these were ALREADY ported to the state API and only import a removed raw kernel for one or two tests —
   drop those imports and the tests that targeted the kernel (or re-express through the state API with the same
   tolerance when that is a one-line change). `test_fk_matrix.py`: if every test targets `_fk_matrix`, trash the file.
   `test_boundary_validation_count.py`: port to patch `kinematics.state._validate_q` and count validations through
   `forward_kinematics_state` + the `dynamics.state` functions, keeping the "validated exactly once" assertions.
2. Failing tests (15): `tests/contract/test_submodule_public_imports.py` (expected lists -> current `__all__` of each
   subpackage; read the `__init__` files), `tests/test_skeleton_signatures.py::{test_data_has_core_fields,
   test_data_legacy_aliases_are_removed}` (subject gone -> drop), `tests/contract/test_roadmap_stub_inventory.py`
   (it counts NotImplementedError stubs against `docs/reference/roadmap.md`; the JointMimic stub is gone -> update
   the test's inventory AND the roadmap page's matching line so they agree; read both).
3. Grep-gate leftovers in passing tests: `tests/contract/test_naming.py`, `tests/kinematics/test_state.py`,
   `tests/test_pinocchio/test_{crba,rnea}_matches_pinocchio.py`, `tests/bench/conftest.py` — remove the stale
   mentions (comments/docstrings/names). `tests/bench/baselines/*.json` are historical data: leave them.

## Gates (report exact tails)
```bash
CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider
uv run ruff check src tests            # no NEW errors in touched files
uv run mkdocs build --strict
grep -rn "create_data\|StaleCache\|_kinematics_level\|KinematicsLevel\|JointMimic\|\bData\b\|forward_kinematics_raw\|_fk_matrix\|bias_forces\|compute_generalized_gravity" src tests examples benchmarks --include=*.py --include=*.md   # must be empty
cd /home/rikhat.akizhanov/better/dev/BetterHuman && CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider --deselect tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest
```

Final message (≤ 25 lines): files edited/trashed, tests dropped (name + one-line reason), gate tails, anything
undone. If budget runs low: write `a2e3_state.md` here with the exact remaining red list and stop.
