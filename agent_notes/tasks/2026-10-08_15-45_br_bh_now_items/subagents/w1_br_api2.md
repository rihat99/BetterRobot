# w1 br_api (part 2) — slice A remainder

Status: items 4, 5, 7 and the io docs entry for `io.parse` done (items 1, 2, 3 code, 6 were done in `w1_br_api.md`).
Not committed. Full suite NOT run (budget); slice dirs run, see Gates.

## What changed
- `src/better_robot/data_model/trajectory.py` (new): `Trajectory` frozen dataclass, `q (T, nq)`, `times (T,)` seconds,
  `v (T, nv) | None = None` (field order `q, times, v` so `v` can default). `__post_init__` raises `ShapeError` on
  bad shapes (no value scans), `__len__` = T, `at(step) -> (q (nq,), v (nv,) | None)`. Nothing else uses it.
- `src/better_robot/data_model/__init__.py`, `src/better_robot/__init__.py`: export `Trajectory` (`__all__` both).
- `docs/contracts/01-naming.md`: new section `C-NAME-5 SE(3) vocabulary` appended at the end (table placement / pose /
  transform, forbidden synonyms, enforcement line). Lines 62-63 (time axis, agent E) untouched.
- `docs/contracts/00-index.md`: matrix row `C-NAME-5 | tests/contract/test_se3_vocabulary.py` (required by C-DOC-3
  matrix test; one line outside the listed slice files, needed for the contract to be consistent).
- `docs/reference/glossary.md`: `pose` row now says "computed in the world frame"; new rows `placement`, `transform`
  (links C-NAME-5), `Trajectory`.
- `src/better_robot/io/parsers/mjcf.py`: module docstring "Experimental" paragraph listing the known gaps (jointed body
  pos/quat dropped, degree ranges, `limited="auto"`, iquat/fullinertia/geom mass, euler/axisangle/xyaxes/zaxis).
- `docs/concepts/io.md`: paragraph + runnable snippet for `io.parse` ("composition lives on the IR",
  `load = build_model(parse(...))`), and a `!!! warning "MJCF support is experimental"` admonition with the same gaps.
- `src/better_robot/io/__init__.py`: the 2 lazy imports marked `# noqa: PLC0415 - <reason>` (inspect.py style).

## Tests added / changed
- `tests/data_model/test_trajectory.py`: len, `at`, `v=None`, frozen, three `ShapeError` cases.
- `tests/contract/test_se3_vocabulary.py`: regex over `__all__` of 10 public packages + parameter names of exported
  callables + annotated fields of exported classes (707 names today, 0 offenders); rejects `tf`/`xform`/`homogeneous`/
  `se3_matrix`, `pose|placement` + `matrix|matrices|mat`, and `world` + `placement`. Plus parametrized self-checks of
  the regex (known-bad rejected, vocabulary names accepted).
- `tests/contract/test_public_api.py`: `Trajectory` added to the REQUIRED top-level set.

## Gates
- `pytest tests/io tests/contract tests/docs tests/data_model -q`: `1107 passed, 9 skipped`.
- `ruff check src tests`: `Found 80 errors` (baseline 82). All files slice A touched: no new findings vs HEAD
  (io/__init__.py now clean; mjcf.py 6, urdf.py 8, model.py 1 PLC/PLR/F841/E741 are pre-existing at HEAD).
- `mkdocs build --strict`: "Documentation built"; only INFO "not in nav" lines for ADRs 0007/0008 (agents C/D).

## Left / for the orchestrator
- Vocabulary inconsistency NOT renamed (scope/budget): `JointModel.joint_transform(q_slice)` in
  `data_model/joint_models/*.py` and `data_model/joint_dispatch.py` returns a LOCAL 7-vector, but C-NAME-5 says
  "transform" is a world 3x4 matrix. Not caught by the test (protocol method, not exported). Rename candidate:
  `joint_motion_pose` or `joint_placement_delta`; owner decision.
- `kinematics/_fk_lean.py:204 placement_matrices` (private, agent C's file) breaks C-NAME-5's "no placement matrix"
  wording; private so the test ignores it.
- `docs/reference` API pages for `io.parse`, `Trajectory`, `UnknownNameError`: not checked whether auto-generated.
- MJCF tests still skip (mujoco not installed in the worktree venv); MJCF dynamics parsing from part 1 is unverified.
