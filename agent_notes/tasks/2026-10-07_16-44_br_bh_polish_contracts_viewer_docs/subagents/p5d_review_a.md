# P5d-1 review — index, getting-started, concepts (mars_aux, read-only)

11 findings, most severe first (paths relative to the BR worktree):

1. `docs/figures/viewer_layers.svg` box "render (offline) · PNG · GIF · MP4, no server" — `Viewer.record` screenshots
   the live viser server in headless Chromium and writes .webm/.gif/PNG only (viewer.py:157-176, capture.py:27-36).
   Fix: relabel "record (headless Chromium) → webm · gif · png".
2. `docs/concepts/viewer.md:3-4` — "renders … to a GIF, MP4 or PNG frames without one [a browser]"; "offline render".
   Fix: "records the live scene through headless Chromium to .webm/.gif/PNG".
3. `docs/concepts/viewer.md:19` — figure alt "render offline without a server" contradicts viewer.md:131. Same fix.
4. `docs/getting-started/visualize.md:6` vs `docs/concepts/viewer.md:10` — same clip `viewer_layers.webm`, two
   incompatible captions (layers toggled on with set_visible / robot + point layer share one timeline). Check poster.
5. `lie_spatial.svg`, `jacobians.svg`, `concepts/lie-and-spatial.md:105-106` — LOCAL / WORLD / LOCAL_WORLD_ALIGNED
   read like an enum; API takes strings `"world" | "local" | "local_world_aligned"` (kinematics/state.py:190).
6. `concepts/kinematics-and-jacobians.md:53-54` — says `joint_jacobian` also takes a name; it takes `joint: int` only
   (state.py:212).
7. `concepts/optimization.md:149,153` — `solve(problem, solver, iters=50, x=None)` hides the `*`; same for
   `solve_layer` (`*, iters=50, x=None, damping=1e-6`) and `run(*, x=None)`.
8. `concepts/optimization.md:150` — Result fields listed without `result["q"]` indexing (phases.py:47).
9. `concepts/model-and-state.md:109` — `frame_pose_of(frame_id)`; parameter is `frame: int | str` (state.py:147).
10. Stale "frame" for the time axis: `visualize.md:47,51`, `concepts/viewer.md:64`, `viewer_layers.svg` ("scrub
    frames"). Code says step (`Viewer.show(step)`, `Layer.set_step`).
11. `getting-started/gpu-and-batching.md:6` — markdown backticks inside a raw-HTML figcaption; use `<code>`.

Checked and matching: architecture.svg ranks; pipeline_overview/ik_loop/optim_problem/banded_normal/time_structure
figures; Problem keywords and solver defaults; dynamics signatures; every dotted `better_robot.*` ref imports;
every backticked function has a def; extras table; viewer calls used.

Not done: clip posters vs captions (20-knot, obstacle); "8 tangent coordinates per knot" (time-and-structure.md:50);
index, first-model, floating-base, inverse-kinematics, trajectories, collision, io, joints-bodies-frames, rest of
dynamics checked by grep only.
