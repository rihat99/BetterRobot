# w3 rename: JointModel.joint_transform -> joint_placement

Outcome: renamed, all checks green. Not committed.

## What changed (BetterRobot worktree)
- `src/better_robot/data_model/joint_models/{base,fixed(x2),revolute(x2),prismatic,spherical,free_flyer,planar,
  translation,helical,composite}.py`: `def joint_transform` -> `def joint_placement` (12 defs); composite's
  inner call `jm.joint_transform(qj)` -> `jm.joint_placement(qj)`.
- `src/better_robot/data_model/joint_dispatch.py`: dispatch function, its fallback call and `__all__` renamed (13th def).
- `src/better_robot/kinematics/_fk_lean.py:22,271`: import and call renamed.
- `tests/data_model/test_joint_models.py`: module docstring + every `.joint_transform(` call (pytest ids like
  `fixed_transform` left as is).
- `docs/contributing/extending.md:17` (protocol table), `src/better_robot/kinematics/CLAUDE.md:60`.
- `docs/reference/changelog.md` Unreleased: one bullet "Breaking: `JointModel.joint_transform` is `joint_placement` ...".
- NOT touched: `KinematicState.joint_transform` / `frame_transform` and all `state.joint_transform` uses.
- No edit needed: C-EXT-2 in `docs/contracts/08-extension-points.md` names no protocol method; `data_model/CLAUDE.md`
  and `docs/reference/glossary.md` do not list the method.
- Private helpers `_revolute_transform`, `_prismatic_transform`, `_helical_transform` (return 7-vectors) keep
  their names (private, out of brief).

## Verification
- `grep -rnE "joint_transform\(" --include=*.py src tests examples benchmarks scripts` -> no hits.
- `grep -c "def joint_placement"` over joint_models + joint_dispatch -> 13.
- `uv run ruff check src tests` -> `Found 80 errors.` (changed files alone: `All checks passed!`).
- `uv run mkdocs build --strict` -> exit 0, "Documentation built in 3.47 seconds".
- Full suite tail: `1650 passed, 13 skipped, 20 deselected, 33 warnings in 213.95s (0:03:33)`.
- BetterHuman: `grep -rnE "joint_transform\(|joint_dispatch|joint_placement\(" --include=*.py src tests examples scripts`
  in `/home/rikhat.akizhanov/better/dev/BetterHuman` -> no hits (exit 1). Its only `joint_transform` use is the
  KinematicState attribute `self.kinematics.joint_transform.index_select(...)` (`core/body.py:264`), unaffected.

## Open question for the owner
- C-NAME-5 (`docs/contracts/01-naming.md:82`) and the glossary define "placement" as a *model constant*.
  `joint_placement(q_slice)` is q-dependent (local child-in-joint pose), and `Frame.joint_placement` /
  `model.joint_placements` already use the word for constants. The new name is local-not-world (fits C-NAME-5's
  intent) but stretches "constant". Either widen the placement row to "local, child relative to parent" or
  pick a different name (e.g. `joint_motion`). Not changed here.
