# w1 br_api — slice A report

Status: items 1, 2, 3 (code + docstring), 6 done. Items 3 (docs page), 4 (Trajectory), 5 (naming contract), 7 (MJCF
experimental note) NOT done — token budget ran out. Not committed.

## What changed
- `src/better_robot/kinematics/state.py`: `joint_jacobian`, `joint_jacobian_time_variation` take `joint: int | str`,
  resolved via `model.joint_id`. Ids were already keyword-only everywhere (existing `test_keyword_only.py`, MAX_POSITIONAL=2,
  empty allowlist), so no call site changed.
- `src/better_robot/data_model/model_structure.py`, `model.py`: `get_subtree` / `get_support` accept `int | str`;
  `joint_id` / `frame_id` / `body_id` raise the new `UnknownNameError` on an unknown name (was a bare `KeyError`).
- `src/better_robot/exceptions.py`: new `UnknownNameError(BetterRobotError, KeyError)` (KeyError kept so existing
  `except KeyError` callers still work), in `__all__`. `BackendNotAvailableError` docstring: reserved for a future kernel brick.
- `docs/contracts/06-validation-errors.md` (C-ERR-2): one line on `UnknownNameError` and `int | str` ids.
- `docs/contracts/07-dependencies-layering.md`: one line, `BackendNotAvailableError` reserved for a future kernel brick.
- `src/better_robot/io/ir.py`: `IRJoint.damping`, `friction`, `armature` (float, default 0).
- `src/better_robot/io/parsers/urdf.py`: reads `<dynamics damping friction>` (yourdfpy `joint.dynamics`).
- `src/better_robot/io/parsers/mjcf.py`: reads `damping`, `frictionloss`, `armature` through a new `_scalar_attr` helper.
- `src/better_robot/io/build_model.py`: the full-width `armature/friction/damping` tables (were zeros) are filled per
  velocity coordinate from the IR joint (0 for joints with no IR record, e.g. a free-flyer root); the existing mimic
  reduction then maps them to `ModelValues`.
- `src/better_robot/io/__init__.py`: public `parse(source, *, format=...) -> IRModel` (the old parse block of `load`,
  moved); `load` = `build_model(parse(...))`; `parse` added to `io.__all__`. Docstring says composition lives on the IR.

## Tests added
- `tests/contract/test_keyword_only.py::test_kinematics_dynamics_ids_are_keyword_only` (frame/joint/body ids keyword-only).
- `tests/kinematics/test_joint_names.py`: names == ids for both joint Jacobians and subtree/support; unknown
  joint/frame/body name raises `UnknownNameError`.
- `tests/io/test_joint_dynamics.py`: URDF dynamics reach `ModelValues`; MJCF version (`importorskip("mujoco")`, skipped
  here: mujoco is not installed, so the MJCF path is UNVERIFIED); `parse` returns an IR that `build_model` consumes.

## Gates
- `pytest tests/io tests/contract tests/docs`: `986 passed, 9 skipped`; `tests/kinematics` + new tests: `401 passed`;
  `tests/io tests/dynamics`: `139 passed, 1 skipped`. The full suite was NOT run (budget).
- ruff: the whole tree shows 87 findings (baseline 82; other agents are editing the same tree). In my changed files the only
  findings are 2 PLC0415 in `io/__init__.py`. Both are lazy imports that already existed (yourdfpy, JointFreeFlyer); one
  moved from `load` into `parse`. I did not check where the other 5 come from.
- `mkdocs build --strict`: NOT run.

## Left / for the orchestrator
- Undone: io docs page entry for `parse`; item 4 `Trajectory`; item 5 SE(3) vocabulary in `01-naming.md` + glossary +
  regex contract test; item 7 MJCF "experimental" note (docstring + io page); docs for `UnknownNameError` in the reference.
- Check that `docs/reference` API pages pick up `io.parse` / `UnknownNameError` (auto-generated? not verified).
- BH: code that catches `KeyError` from `joint_id`/`frame_id` still works (subclass). No known BH break.
