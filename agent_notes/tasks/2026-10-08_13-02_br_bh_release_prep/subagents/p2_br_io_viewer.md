# P2 slice br_io_viewer — BR `io`, `viewer`, `examples/viewer`, top-level modules

Scope note: this agent ran with a 50k-token budget, so the slice (5562 lines) was swept by risk, not read end
to end. Files below are marked **full**, **targeted** (structure grep + the risky region) or **not read**.
A follow-up sweep of the "not read" files is still owed.

## Files read
- full: `io/parsers/urdf.py` lines 160-295 (body/joint/root extraction), `io/parsers/mjcf.py` lines 60-184 (body tree visit).
- targeted: `io/parsers/urdf.py` 1-160 (geoms, mesh scale), `io/parsers/mjcf.py` rest, `io/__init__.py` (`load` signature),
  `io/ir.py` (`IRBody`), `viewer/viewer.py` 1-72, `viewer/capture.py` (connect error path), `examples/viewer/04_record.py`.
- run, not read line by line: `examples/viewer/01..03 --check` (all rc=0), `04_record.py` (no `--check`; run headless, see below),
  `viewer/layers/points.py` + `timeline.py` via probes (T=1 works; T=0 raises `ValueError: nsteps must be at least 1`, acceptable).
- not read: `io/build_model.py` (1165), `io/parsers/programmatic.py`, `io/builders/kinematic_tree.py`, `io/assets.py`,
  `viewer/{robot,_math,theme,layer}.py`, `viewer/layers/{mesh,skeleton,skinned,arrows,frames,ground,plot}.py`,
  `viewer/tools/{gizmo,picking}.py`, `exceptions.py`, `_typing.py`, `_validation.py`, `__init__.py`.

## Fixed (what, where, test)
1. **URDF inertia ignored the `<inertial><origin rpy>` rotation** (`io/parsers/urdf.py`, body loop). The URDF spec gives
   the inertia in the inertial frame; the parser copied it as-is into the link frame. Probe vs Pinocchio on rpy=(0.3,-0.5,1.1):
   BR `diag(1,2,3)+ixy` vs Pinocchio `[[1.77,-0.40,-0.18],...]`. Fix: `inertia = R @ I @ R.T` with R from the inertial origin.
   Test: `tests/io/test_urdf.py::test_inertia_is_rotated_from_inertial_frame_into_link_frame` (Pinocchio parity).
   Effect: dynamics of any URDF with a rotated inertial frame change (now correct). Robots with rpy=0 are unchanged.
2. **`Viewer.url`/`port` reported the requested port, not the bound one** (`viewer/viewer.py` `__init__`). viser silently
   moves to the next free port when the port is taken; `record()` then pointed Chromium at the wrong server and failed after
   30 s (`examples/viewer/04_record.py` failed here because 0.0.0.0:8080 was busy). Fix: `port = server.get_port()` when
   the Viewer creates the server. Test: `tests/viewer/test_server_smoke.py::test_url_follows_viser_when_requested_port_is_taken`.
   After the fix, `04_record.py` (port 8095 copy) wrote `panda.gif` (8 frames, 17.1 s) and `panda.viser` (1.5 MB); server stopped.

## Reported, not changed (ranked)
1. **CRITICAL — MJCF parser gives wrong kinematics for every body that has a joint** (`io/parsers/mjcf.py:131-132`).
   Joint origin = `jnt.pos` only; the body's own `pos`/`quat` is dropped (it is used only for joint-less bodies, line 172).
   Repro (`scratchpad/p2io/t.xml`, `probe_fk.py`): MuJoCo puts body `a` at (0,0,1) rot z 90°, `b` at (0,1,1); BR puts both
   at the origin with identity rotation. Needs a design decision (a nonzero `jnt.pos` is a joint axis off the body origin,
   which needs a body-offset frame); suggested: compile the spec (`spec.compile()`) and read `body_pos/quat`, `jnt_pos/axis`,
   `jnt_range`, `body_ipos/iquat/inertia/mass` from the compiled `MjModel`, which also fixes items 2-5.
2. **HIGH — MJCF joint ranges are not converted from degrees** (`mjcf.py:135-141`). MJCF default `compiler angle="degree"`;
   `MjSpec` keeps raw units. Repro: `range="-90 45"` → BR limits `[-90, 45]`, MuJoCo `[-1.5708, 0.7854]`.
3. **HIGH — `limited="auto"` (MuJoCo 3 default) is read as `True`** (`mjcf.py:136`, `bool(enum 2)`), so a joint without
   `range` gets limits `[0, 0]` (locked). Repro: joint `j2` → BR `[0,0]`, MuJoCo `jnt_limited=0`.
4. **MEDIUM — MJCF inertia ignores `iquat`/`fullinertia`, and geom-derived mass (`inertiafromgeom`)** (`mjcf.py:88-91`).
   Repro: `quat="0.707 0.707 0 0" diaginertia="1 2 3"` → BR keeps `diag(1,2,3)` unrotated.
5. **MEDIUM — MJCF `euler`/`axisangle`/`xyaxes`/`zaxis` orientations are ignored** (spec keeps them in `alt`, `quat` stays
   identity) for joint-less bodies, geoms and sites; and a body with several joints emits several joints with the same child.
6. **GATE GAP — MJCF has no coverage in the dev env**: `mujoco` (extra `io-mjcf`) is not installed in the worktree venv, so
   MJCF tests skip. Probes above used the main checkout's venv (`mujoco 3.8.0`) read-only with `PYTHONPATH` = worktree src.
   Suggest installing the extra in CI and adding a MuJoCo FK parity test.
7. LOW — `Viewer(server=...)` with a user server keeps the default `host/port` in `url` (unchanged; fakes lack `get_port`).
8. LOW — `examples/viewer/04_record.py` has no `--check` and no `--port`; it hard-codes the default 8080.
9. LOW — URDF root selection takes the first body that is never a child (`urdf.py:275-280`); a URDF with two roots is not
   rejected here (may be validated in `build_model`, not checked).

## Performance notes (measured)
None measured (budget). `record`: 8 frames at 320x180 took 17.1 s on CPU headless Chromium (~2.1 s/frame), dominated by
screenshots; not profiled.

## Gate results
- Slice tests: `tests/io tests/test_pinocchio` 309 passed; `tests/viewer` 30 passed, 1 skipped (`test_capture.py:80`, no
  Chromium on PATH in pytest env); earlier `tests/io tests/viewer tests/test_exceptions*` 170 passed, 2 skipped.
- `examples/viewer/01..03 --check`: rc=0. `04_record.py`: works after fix 2 (port 8095).
- Full suite (concurrent edits by other agents in the worktree): 3 failed, 1584 passed, 11 skipped. Two failures are
  `tests/docs/test_snippets.py` for `docs/getting-started/visualize.md` (execute + outputs_fresh); third not captured.
  NOT diagnosed (budget). Suspect: fix 2 — with 8080 held by another process the snippet's printed `url` is now the
  real port (8081), not 8080. Check before merge: rerun with 8080 free.
- `ruff check src tests`: 82 findings = baseline, no new ones (17 pre-existing in the 4 touched files).
