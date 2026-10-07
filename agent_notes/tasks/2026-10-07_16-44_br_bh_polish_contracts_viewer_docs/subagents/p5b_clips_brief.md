# P5b-1 brief — docs clips and screenshots via `Viewer.record`

Read `context.md`, `p0_docs_refs.md` Part 1b (clip budget: 4–10 s, 960 px, 24 fps, ≤ 1.5 MB per WebM, poster
PNG), `p3b_state.md` (capture API, env vars, known loose framing), `src/better_robot/viewer/capture.py`,
`src/better_robot/viewer/viewer.py::record`, examples `examples/viewer/*.py`, `examples/0{1,2,3,6,8}_*.py`.
Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`. Scope: `src/better_robot/viewer/capture.py` (framing fix
only), `docs/clips/**` (new), `docs/images/**` (replace the 4 low-quality rasterizer PNGs), one generator script
`agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/subagents/make_clips.py` (committed with
the notes, so the clips are reproducible). Env for Chromium/ffmpeg:
`BETTER_ROBOT_CHROME=~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell`,
`LD_LIBRARY_PATH=/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/libasound/x/usr/lib/x86_64-linux-gnu`,
`FFMPEG=~/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux`. GPU only if `python3 ~/.claude/skills/autoresearch/scripts/gpu_free.py --top 1 --need-gb 2`
says FREE (then `gpu=True`); CPU is ≈1.6 s/frame, budget accordingly (≤ 120 frames per clip, 24 fps playback
of 12 fps capture is fine — set `fps` so the clip plays at real time). Ports 8094 only; stop everything.

1. Framing fix in `fit_camera`: fit the projected bounding box of `world_points` over the recorded steps (not the
   bounding sphere) so the subject fills ≈ 70 % of frame height; 3/4 view (azimuth ≈ 35°, elevation ≈ 25°),
   up = theme up. Unit test the fill ratio with synthetic points in `tests/viewer/test_capture.py`.
2. Produce, 960×540, white background, LIGHT theme, GroundLayer sized to the subject:
   - `docs/images/panda_rest.png`, `panda_ik.png` (solved pose with target marker), `g1_floating.png`
     (G1 standing, free-flyer triad visible), `trajectory_strip.png` (4 poses of the 02 trajectory side by side
     using viewer groups with offsets, one frame).
   - `docs/clips/panda_trajectory.webm` + `.png` poster (examples/02_trajectory: 60 knots, torque-coloured skeleton).
   - `docs/clips/batched_ik.webm` + poster (8 Panda copies in a grid via groups, each converging to its own
     target — record the LM iterations as steps).
   - `docs/clips/g1_clip_fit.webm` + poster (examples/03_clip_fit_phases result, floating base walking/moving).
   - `docs/clips/viewer_layers.webm` + poster (one scene: mesh → +skeleton → +frames → +ground toggled on over
     the steps, via `set_visible` per step).
   - `docs/clips/obstacle_trajectory.webm` + poster (examples/06).
   Each clip ≤ 1.5 MB; report sizes and frame counts. Look at every poster PNG with the Read tool; re-shoot if
   framing/colours are off.
3. Trash the old `docs/images/*.png` you replaced (trash.py) — same filenames, so no doc links change; the
   `banded_sparsity.png` (matplotlib) stays.
4. `make_clips.py`: one script, `--out docs/`, reproduces everything; docstring lists the env vars.

Gates: `tests/viewer`, `uv run pytest tests/docs -q` (image existence checks), `uv run ruff check src/better_robot/viewer tests/viewer`.
Final message ≤ 20 lines: files produced with sizes, framing before/after, timing, anything that failed.
Do not spawn agents. Do not commit.
