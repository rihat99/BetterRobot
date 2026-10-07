# P4-2a brief — studio: model chooser, frames layer, saved-sets dropdown, session lock

Read `context.md`, `p4_studio_brief.md` (target shell; owner decisions), `p4_state.md` (what P4 already fixed:
viewer-owned visibility, stale-rebuild, Display tab, LIGHT theme, data-dir default; exact remaining list), then
BH `src/better_human/viz/studio/{app.py,layers.py,state.py,frames_editor.py,persistence.py,gizmo.py}` and BR
`src/better_robot/viewer/{viewer.py,layer.py,layers/frames.py,layers/points.py}` (read-only). Repo
`/home/rikhat.akizhanov/better/dev/BetterHuman` (BR is at commit 31a37ed, do not edit it). Scope:
`src/better_human/viz/studio/**`, `tests/viz/**`, `docs/studio.md`.

Do exactly these four, in order, running `tests/viz` after each:
1. **Model tab** (first tab): family dropdown (smpl, smplx, mhr), gender dropdown (neutral/male/female; hidden or
   disabled for mhr), LOD slider (mhr only), "Load" button. Load → `session.load_body(...)` (exists) → rebuild
   the body-dependent widgets: shape-slider groups, density, pose dropdown, `model_key` used by persistence,
   the editors' built-in map. Rebuild by removing and re-adding the affected tab/folder (viser `remove()` on the
   handles). The gizmo must survive a model switch (re-attach or recreate). Status line reports the loaded key.
   Test: switch smpl → smplx → mhr with the fake server: no exceptions, layer names unique, `session.body`
   family matches, shape-slider count matches the new body.
2. **Frames layer from `session.frame_set`**: a layer (BR `FramesLayer` if it takes `(T, N, 7)` poses, else
   `PointsLayer` + small triads) showing every authored frame (vertex position + parent-joint orientation) as
   soon as a set is loaded/edited (`state.py` frame_set mutations call a `studio_layers.refresh_frames()` hook).
   The old model-frames layer (`body.contact_frames`) stays only if it is distinct; otherwise replace it. Test:
   load `studio_data/frames/smplx_neutral/contacts.json` → layer shows `len(frame_set)` markers; add a frame →
   `+1`.
3. **Frames tab saved-sets dropdown** (like the Segment tab's): lists `persistence` saved sets for the current
   `model_key`, refreshed after Save and after a model switch; free-text box only for the Save name.
4. **Session lock**: one `threading.Lock` on the session (or `HumanStudio`) wrapping every GUI callback and the
   gizmo callback (the gizmo runs on BR's worker thread). Document it in a one-line comment at the lock.

Gates: `CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests/viz -q -p no:cacheprovider`
(all green, no test weakened; dropped tests listed with reason), `uv run ruff check src tests`. Append
"P4-2a" to `p4_state.md` (what changed, remaining list for P4-2b: colour modes, in-place opacity/radius/edit,
drive on 8095, screenshots via BR `viewer.capture`, docs). Final message ≤ 20 lines. Do not spawn agents. Do
not commit. No servers needed.
