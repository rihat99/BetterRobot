# A2e-2 brief — the Data-API removal itself (steps 1, 2, 4 of `a2e_brief.md`)

Read `context.md` (rules: no commits, never `rm` — use `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>`,
CPU only `CUDA_VISIBLE_DEVICES=""`, never skip/xfail/weaken a test), then `a2e_brief.md` steps 1, 2 and 4, then
`a2e_state.md` (a caller map written by the previous agent, who did steps 3 and 5; that work is committed at HEAD 52245da).
`a2_inventory.md` has the Data -> state mapping. Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`, branch `dev`.

Scope: ONLY the removal (steps 1, 2, 4) plus the tests that reference what you remove. Do NOT edit CLAUDE.md, README.md
or docs — the main agent does those after you. Budget is tight: do not re-derive what `a2e_state.md` already lists; trust
it after a quick grep. Work file by file and run the focused tests of each package as you go
(`uv run pytest tests/kinematics tests/dynamics tests/contract tests/data_model tests/io tests/autograd
tests/test_exceptions_wired.py tests/test_skeleton_signatures.py -q -p no:cacheprovider -x`).

Decisions already taken:
- Raw kernels with no caller left in `src/` are trashed with their modules; the test functions that targeted only those
  kernels (in `tests/autograd/test_public_differentiability.py`, `tests/kinematics/test_jacobians.py`,
  `tests/test_exceptions_wired.py::test_fixed_base_bypasses_quaternion_check`) are dropped and listed, unless the same
  check can be expressed through the state API with the same tolerance — then port it.
- `tests/contract/test_boundary_validation_count.py`: port to the state API (patch `kinematics.state._validate_q` and
  the `dynamics/state.py` entry points; keep the "validated exactly once" assertions).
- `JointMimic` class goes; the `"mimic"` kind strings in `joint_models/{base,composite}.py` stay.
- Root `__init__.py`: `center_of_mass` -> `dynamics.state.center_of_mass`; every Data-era export removed; keep `__all__`
  sorted the way it is now.

Gates (report exact tails):
```bash
CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider
uv run ruff check src tests          # no NEW errors in touched files
grep -rn "create_data\|StaleCache\|_kinematics_level\|KinematicsLevel\|JointMimic\|\bData\b\|forward_kinematics_raw\|bias_forces\|compute_generalized_gravity" src tests examples benchmarks   # must be empty
cd /home/rikhat.akizhanov/better/dev/BetterHuman && CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider -x --deselect tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest
```
(The deselected BH test is a known pre-existing 1-ulp failure; do not touch it.)

Final message (≤ 25 lines): files trashed, files edited, tests dropped (name + reason), gate tails, final root
`__all__` additions/removals, anything undone. If budget runs low, write `a2e2_state.md` here and stop with the tree
green (or say exactly what is red).
