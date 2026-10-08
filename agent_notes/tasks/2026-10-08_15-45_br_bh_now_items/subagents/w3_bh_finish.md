# w3_bh_finish — finish of w2_bh (BH worktree /home/rikhat.akizhanov/better/dev/BetterHuman, uncommitted)

All five items done. Full BH suite green, ruff clean, mkdocs strict clean. Nothing committed.

## 1. Test: studio resets `frames.active` on a model switch
- `tests/viz/test_studio_app.py::test_model_switch_clears_active_frame` (fake server): add a frame by a
  pointer click (active = `frame_0`), `load_model("smpl", gender="female")`, assert key `smpl_female` and
  `frames.active is None`. A different model is used so the old name cannot survive by accident.
- Mutation check: with `app.py:189` (`self.frames.active = None`) removed, the test fails
  (`assert 'frame_0' is None`). Source restored byte-identical (cmp).

## 2. Test: MHR `build_model` device-cache hit
- `tests/families/test_mhr.py::test_build_model_reuses_cached_tree_for_bare_cuda` (CPU only, no weights):
  monkeypatches `torch.cuda.current_device -> 1` and `mhr_rig.build_mhr_tree` to raise; a fake cached robot
  whose `joint_placements` claims `cuda:1`. Asserts `build_model(device=torch.device("cuda"))` and
  `"cuda:1"` return the cached object, `"cuda:0"` rebuilds (hits the raising stub), plus two `_same_device`
  cpu/cuda checks.
- Mutation check: with `_same_device` returning False for bare `cuda`, the test fails. Restored (cmp).

## 3. `docs/reference/roadmap.md` (BH)
- Vision, "Relationship to BetterRobot": the sentence now reads "Humanoid *robots* need almost nothing
  here — BetterRobot already loads URDF — and human↔robot retargeting is a separate package built on both,
  not part of BetterHuman." (dropped "limited to human↔robot tools (retargeting), with a namespace reserved").
- "Landed (now)" checked against w2_bh.md and the diff. Kept: `corrective_features(identity=)` + memo
  (`_memo` init=False), cached `vertices`/`frame_poses` properties, `META_KEYS` + test, `(B, T, nq)` time
  semantics in addendum, studio `on_pointer`/`clear_pointer`/`nearest_on_ray` + frame reset + MHR cache.
  DROPPED: "Call sites follow BetterRobot's keyword-only identifiers." — no such change is in w2's done list
  or in the BH diff (w2: "No other BR-API fallout found").
- Open: Vision item 4 still says "Conversion between body models and retargeting to humanoid robots are
  modules over the same core." Left as is (brief named one sentence; it is still true of a separate package
  over BR), but the orchestrator may want it reworded for consistency with the owner decision.

## 4. Live studio check (real viser on 127.0.0.1:8082, headless Chromium)
- Driver: `subagents/drive_studio_p3.py` = copy of release_prep `drive_studio_p2.py` plus one scene click
  dispatched through the callback that `Viewer.on_pointer` registered in the real server
  (`server.scene._scene_pointer_cb`; the websocket path needs a client handle, so the entry is called directly).
- Log `vis/studio/drive.log`, PNGs `vis/studio/studio_{before_select,pointer_click,joint_selected,after_deselect}.png`.
- Results: one click slot registered (`('click', 'run')`). Pointer click on joint 18 ->
  `smplx neutral · joint: left_elbow · frame: —`, status `posing 'left_elbow' (spherical)`, gizmo drawn on the
  left elbow in the PNG. Buttons: Deselect frame -> frame `—`, gizmo kept; Deselect joint -> joint `—`,
  gizmo gone; Reset pose -> both `—`. Exit 0, "server stopped"; `ss -ltn` shows nothing on 8082 (or 8097).
- Note (pre-existing, not changed): the status line is not updated by the three buttons; after them it still
  shows the last message (`frame 'frame_0': vertex 3000, joint 15`).

## 5. Gates (BH worktree)
```
CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider
339 passed, 1 skipped, 13 warnings in 294.74s (0:04:54)
uv run ruff check src tests
All checks passed!
uv run mkdocs build --strict
INFO    -  Documentation built in 0.36 seconds
```
- `test_corrective_slice_is_exact_when_dropped_joint_is_at_rest` passes (w2's tolerance change; owner
  decision in PLAN.md says float32 ulp comparison, matching).
- For BR: one of the 13 warnings comes from BR `Viewer.on_pointer` (`viewer.py:278`), which calls viser's
  `scene.on_pointer_event`, deprecated in the installed viser ("Use on_click or on_rect_select instead").
  Not fixed (BR side, out of this brief).

## Files touched
- `/home/rikhat.akizhanov/better/dev/BetterHuman/tests/viz/test_studio_app.py` (+1 test)
- `/home/rikhat.akizhanov/better/dev/BetterHuman/tests/families/test_mhr.py` (+1 test, +2 imports)
- `/home/rikhat.akizhanov/better/dev/BetterHuman/docs/reference/roadmap.md` (1 sentence, 1 bullet dropped)
- this task: `subagents/drive_studio_p3.py`, `vis/studio/*` (4 PNGs + drive.log), this file.
- Scratch studio data dir (not in repo): `<scratchpad>/w3_studio_data`.
