# P3b brief — drop the rasterizer; `Viewer.record` via headless Chromium

Read `context.md`, then `p0_viewer.md` §1 (what exists only for the rasterizer), §2 (`get_render`), §3 (measured
headless path, numbers, the libasound caveat), the "P3a-3" section of `p3a_state.md` (current viewer API), and the
scripts `subagents/p0_assets/{headless.py,headless2.py,chrome.sh}` and the scratchpad script
`/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/shot_p3a3.py`
(its camera was set before the scene arrived → empty frames; fix that). Repo
`/home/rikhat.akizhanov/better/dev/BetterRobot`, clean at `2f7c6e2`. Scope: `src/better_robot/viewer/**`,
`tests/viewer/**`, `examples/viewer/04_record.py`, `pyproject.toml` (optional-extra text only), `docs/concepts/viewer.md`
+ `docs/getting-started/visualize.md` (only the `record` paragraphs/snippets). Another agent edits non-viewer src.

Owner decision: one rendering path. The own rasterizer goes; `record` screenshots the real viser scene in a headless
Chromium. No Python dependency is added (no Playwright): the browser binary is found via `$BETTER_ROBOT_CHROME`,
else `shutil.which("chromium")/("google-chrome")/("chrome-headless-shell")`, else a clear `RuntimeError` with the
install hint. `libasound.so.2` absence is the machine's problem: document the `LD_LIBRARY_PATH` workaround in the
docstring, do not ship a wrapper in `src`. Put the chrome wrapper the agents used under `scripts/chrome_headless.sh`
(dev helper, documented) if it is useful for tests.

Work:
1. New `viewer/capture.py`: `HeadlessClient` context manager — launches Chromium headless (`--no-sandbox
   --window-size=W,H [--use-angle=gl-egl --enable-gpu --ignore-gpu-blocklist when gpu=True]`) on `viewer.url`, waits
   for `server.get_clients()` (timeout → error), exposes `render(height, width) -> np.ndarray (H, W, 3) uint8` via
   `client.get_render`, `set_camera(position, look_at, up)`; kills the process on exit. Frame the camera from the
   listed layers' bounds (reuse/relocate the fitting logic that `render/__init__.py::frame_camera` has today —
   compute bounds from the layers' current positions, NOT from `geometry()`), set it AFTER the first render
   succeeds (the scene must have arrived), then render.
2. `Viewer.record(path, *, steps=None, size=(960, 540), fps=None, gpu=False, camera=None)`: for each step
   `show(step)`, render, write PNG frames to a directory, or WebM when `path` ends in `.webm` (ffmpeg found via
   `$FFMPEG` / `shutil.which("ffmpeg")`, VP8 command from `p0_docs_refs.md` Part 1b; error with hint if absent);
   `.gif` via Pillow (already a dependency through imageio? check `uv run python -c "import PIL"`; if absent, no
   GIF). Returns the list of frame arrays or the output path.
3. Remove the rasterizer: trash `viewer/render/` (trash.py), `Geometry`, `Layer.geometry` and every `geometry()`
   method, `_Shifted` (group offsets must still render correctly — they are real viser node offsets, so headless
   capture sees them), `tests/viewer/test_render.py`. Rewrite `examples/viewer/04_record.py` and the two docs
   snippets to the new API (`no-run` tag on snippets that need a browser; `tests/docs/test_snippets.py` shows the
   tag syntax). `make_screenshots.py`/`make_panda_ik.py` in the older task folder used the rasterizer — leave them,
   note it.
4. Tests: `tests/viewer/test_capture.py` — unit test with a fake client (no browser) for the frame/camera/output
   logic; one `@pytest.mark.slow` integration test that runs only if a Chromium binary is found
   (`pytest.skip` otherwise is allowed here because it is an environment capability, not a weakened assertion).
5. Verify for real on this machine: record 20 steps of the Panda trajectory (`examples/viewer/02_play_trajectory.py`
   logic) to `agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/vis/p3b/panda.webm` and one
   PNG `panda_000.png` (CPU unless `python3 ~/.claude/skills/autoresearch/scripts/gpu_free.py --top 1 --need-gb 2`
   says FREE). Port 8094 only; stop everything you start. Report ms/frame.

Gates: `tests/viewer`, full suite, `uv run ruff check src/better_robot/viewer tests/viewer examples`, `uv run mkdocs
build --strict`, `uv run pytest tests/docs -q`. Final message ≤ 25 lines: API old→new, files trashed, timing,
artifact paths, gate tails, BH impact (grep `/home/rikhat.akizhanov/better/dev/BetterHuman/src` for `record(`,
`geometry(`, `render`). Do not spawn agents. Do not commit. If budget runs low: do not leave the tree half-removed —
finish item 3 or don't start it; write `p3b_state.md` with the exact remaining list.
