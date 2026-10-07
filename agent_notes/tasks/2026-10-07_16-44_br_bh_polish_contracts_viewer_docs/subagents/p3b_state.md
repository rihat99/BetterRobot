# P3b state (budget ran out; item 3 NOT started, tree consistent)

Done: items 1, 2, 4, 5.
- `src/better_robot/viewer/capture.py` (new): find_chrome, find_ffmpeg, fit_camera, HeadlessClient, write_frames.
- `viewer.py::Viewer.record(path, *, steps, size, fps, gpu, camera)` + `_points(steps)`; old record test moved out of
  `tests/viewer/test_core.py` (copy in scratchpad/old_record_test.py); unused `render` import dropped there.
- `tests/viewer/test_capture.py` (3 unit + 1 slow integration).
- ffmpeg: frames go as JPEG via `-f image2pipe -c:v mjpeg -i pipe:0` (Playwright's ffmpeg has no rawvideo/png/"-").

Remaining (exact):
1. `Viewer._points` still uses `layer.geometry()` for framing. Item 3 needs a replacement source of bounds
   (e.g. a small `Layer.points(step) -> (N,3)` per layer, or the layer arrays) BEFORE geometry() is removed.
2. Item 3: trash `viewer/render/`, `tests/viewer/test_render.py`; remove `Geometry` (layer.py:13), `Layer.geometry`
   (layer.py:69), geometry() in layers/{arrows,points,skinned,mesh(x3),frames,ground,skeleton}.py, `_Shifted` +
   `Geometry` import in viewer.py; update viewer/__init__.py:7-8 docstring and viewer/CLAUDE.md (render/ lines).
3. `examples/viewer/04_record.py` rewrite; docs `docs/concepts/viewer.md` + `docs/getting-started/visualize.md`
   record snippets (`no-run`).
4. `scripts/chrome_headless.sh` not added (the agents' wrapper hard-codes scratchpad paths).
5. Gates not run: full suite, mkdocs --strict, tests/docs, BH grep.

## P3b-2
- Done: `Layer.world_points(step)` (named so: `PointsLayer.points` array would shadow `points()`, and BH studio writes `edit.points[...]`); implemented on mesh x3, skinned, skeleton, frames, points, arrows; `Viewer._points` uses it + offsets. Removed Geometry, geometry(), _Shifted, fits_camera, orphan helpers; trashed render/ + test_render.py (BetterRobot/trash/). New offset unit test in test_capture.py. Ground: cell 0.25/section 1.0, lighter lines; theme.robot 0.62; fit margin 1.15. Re-recorded vis/p3b (white bg OK).
- Gates run: tests/viewer 27 pass; tests/docs 91 pass; ruff viewer clean (pre-existing FBT001/TID251 only).
- NOT done: full suite, mkdocs --strict, BH grep; docs/concepts/viewer.md lines ~125-140 still mention render.frame_camera/render.record; panda fills ~45% of height (Panda meshes are white from URDF, theme.robot unused there) — fit by projected bbox needed.
