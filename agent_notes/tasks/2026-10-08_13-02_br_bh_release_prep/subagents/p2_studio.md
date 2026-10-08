# P2 studio — owner-chosen HumanStudio changes (BH)

## What changed
- `src/better_human/viz/studio/gizmo.py`: transform controls get `depth_test=False` (draw through the mesh) for every
  joint; default `scale` 0.15 -> 0.20. BR `viewer/tools/gizmo.py` untouched (`**options` already reaches viser).
- `src/better_human/viz/studio/app.py`: above the tab group (built once in `_build_sidebar`, so a model switch keeps it
  on top) a one-line markdown `model · joint: <name|—> · frame: <name|—>` and a viser button group (no label) with
  `Reset pose`, `Deselect joint`, `Deselect frame` (`_ACTIONS`, `run_action`). `deselect_frame` reuses what
  `FramesEditor.delete` does (`frames.active = None`, `layers.set_edit_vertices([])`). Label refresh: `select_joint`,
  `set_status` (the frames editor reports every frame selection through it), `_on_change` for body/frames kinds.
  The two old buttons are removed from the Body tab.
- `docs/studio.md` lines 47-48: Pose bullet no longer lists the moved buttons.
- Tests: `tests/viz/test_studio_gizmo.py::test_root_is_six_dof` asserts `depth_test is False`;
  new `tests/viz/test_studio_app.py::test_selection_row_buttons_and_label` (buttons gone from Body, label follows
  joint and frame selection, the three actions clear joint/frame and reset `q` to `q_neutral`).

## Real-server check (127.0.0.1:8082, stopped afterwards; port closed)
Driver: `subagents/drive_studio_p2.py` (adapted from the previous task's drive_studio.py). Log lines:
- after selecting joint 18 + adding a frame: `smplx neutral · joint: left_elbow · frame: frame_0 | depth_test: False | scale: 0.2`
- `Deselect frame` -> `frame: —`; `Deselect joint` -> `joint: —`, gizmo removed; `Reset pose` ok.
Screenshots (`vis/studio/`): `studio_before_select.png`, `studio_joint_selected.png` (elbow gizmo rings visible over
the arm), `studio_after_deselect.png`.

## Owner should look at
1. LAYOUT BUG, not fixed (out of budget): viser's button group does not wrap, so at the default sidebar width the
   third button `Deselect frame` is clipped off the right edge (seen in `studio_joint_selected.png`, with and without
   the group label). Clicks still work server-side. Options: three plain `gui.add_button` rows (always visible, less
   compact) or shorter labels (`Reset`, `Joint ×`, `Frame ×`). Recommend plain buttons in a small folder.
2. The selection label wraps to two lines at the default sidebar width (long joint/frame names).
3. After a model switch `frames.active` keeps the old name; the label hides it if the frame is not in the new set,
   but the Frames tab may still act on it.

## Gate results
See the final lines appended below.
- `tests/viz/test_studio_app.py` + `test_studio_gizmo.py`: 24 passed (before the group-label tweak; all of `tests/viz`
  71 tests passed apart from the new test's first draft, since fixed). `ruff check src tests`: All checks passed.
- Full BH suite: started, still running at handback; output in
  `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/tasks/b62j3c3xd.output`
  (same command also prints `git ls-files studio_data` and `git status --short`). NOT verified by me.
