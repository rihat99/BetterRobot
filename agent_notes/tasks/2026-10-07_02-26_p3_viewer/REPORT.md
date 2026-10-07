# REPORT — P3 viewer (BetterRobot core, robot adapter, offline recording)

Task `2026-10-07_02-26_p3_viewer`, branch `dev`, commits `54ce744` … `1306ed5` (none pushed). Owner was asleep: no
questions asked; assumptions are listed at the end. Plan: `PLAN.md` in this folder; design intent:
`../2026-10-06_20-13_architecture_plan/subagents/study_viewer.md` §3.

## Outcome in one paragraph

The three old viewers' job in BetterRobot is now one package, `src/better_robot/viewer/` (2,160 lines including the rasterizer and tools; the old one
was 2,463 without recording): a `Viewer` on viser with groups of `Layer`s and a `Timeline`; layers hold preloaded `(T, ...)` numpy
arrays and push one frame at a time into batched scene nodes; `RobotView`/`robot_layers` turn a `Model` and a
clip `q (T, nq)` into mesh, skeleton (torque-coloured), frame and collision-sphere layers with one FK call;
tools (`PoseGizmo`, picking) sit on top; a torch rasterizer records GIF/PNG/MP4 without a browser and
`Viewer.save` writes `.viser` files for the static client. The old viewer, its tests and the `tasks`
dependency are gone; examples run on the new API; docs and contract tests are updated; a clean-context review
found one major bug (rasterizer coverage) which is fixed and pinned by a test.

## What was built (per milestone)

| M | Files | Check |
|---|---|---|
| M1 core | `viewer.py`, `timeline.py`, `layer.py`, `theme.py`, `_math.py`, `layers/{skeleton,arrows,points,frames,ground}.py` | fake-server tests (pushes once per frame in one `atomic()`, hidden layers skipped and caught up, group teardown, timeline clamp/loop/GUI); loopback smoke: 50-frame skeleton+arrows+frames+1000 points at **0.23 ms/frame** Python |
| M2 robot | `robot.py` (`RobotView`, `robot_layers`, `robot_meshes`), `layers/mesh.py` (`MeshLayer`, `RigidMeshLayer`, `SpheresLayer`) | Panda (11 DAE meshes, vertex colours kept) and G1 (36 STL) play; FK once per clip; mesh world pose = `joint_pose ∘ origin` matches `lie.se3.compose` to 4e-8; `RobotView.update(q)` rewrites arrays in place |
| M3 tools | `tools/gizmo.py`, `tools/picking.py` (BH studio picking ported to numpy), `layers/plot.py` (`PlotPanel`), `examples/viewer/01–03`, old examples 01/02/04/05 ported | all examples build their scenes on a loopback server; IK gizmo example reaches a moved target to 7e-12 m with `terms.solve_ik` |
| M4 recording | `render/rasterizer.py` (tiled z-buffer, perspective-correct depth, flat shading, size-classed blocks), `render/__init__.py` (`render_frames`, `record`, `frame_camera`), `Viewer.record/save`, `examples/viewer/04_record.py` | **160 ms** per 640×360 Panda frame (136k faces) on CPU with all threads, 487 ms on one thread, 203 ms at 1080p; GIF of 300 frames in 45 s; `.viser` of 300 frames **1.7 MB** (11 GLB messages once + ~24 pose messages per frame, decoded by the reviewer); coverage test against a brute-force edge test |
| M5 | `layers/skinned.py` (`SkinnedMeshLayer`: client bones, numpy LBS for offline; `CameraLayer`: frustum + per-frame or lazy images) | synthetic two-bone mesh bends at the second bone; top-4 weights renormalised |
| M6 | old viewer trashed to `trash/`; `docs/concepts/viewer.md`, `docs/guides/visualize.md` (doctest passes), `docs/conventions/extension.md`, autodoc pages; `viewer/CLAUDE.md`; contract tests (viser only in `viewer.py`, trimesh only in `robot.py`, no pyrender/imageio) | full suite **1743 passed**, known optim failure deselected; `grep render_modes\|RendererBackend` empty outside `trash/` |
| M7 | `subagents/context_review_viewer.md`, review fixes `1306ed5` | see below |

Artifacts (gitignored): `artifacts/panda.gif`, `artifacts/panda.viser` (absolute path
`/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_02-26_p3_viewer/artifacts/`).

## Numbers that matter

| What | Value |
|---|---|
| Frame push, 5 batched layers (24-joint skeleton, 4 arrows, 24 frames, 1000 points) | 0.23 ms Python, ~10 messages |
| Frame push, Panda + G1, visual + collision + skeleton (94 mesh nodes) | 4.2 ms, ~190 messages |
| Rasterizer, Panda 640×360 / 1920×1080 (CPU) | 160 ms / 203 ms (peak RSS ≈ 0.8 GB) |
| `.viser`, 300 Panda frames at 30 fps | 1.7 MB (9.5 MB uncompressed) |

Per-mesh pose writes are the cost for mesh-heavy robots (2 messages per mesh per frame). A per-body frame node
with meshes as children would cut G1 from 72 to ~60 messages; not worth it yet.

## Review (mars_aux, clean context) and what changed

Confirmed: quaternion conventions everywhere (`compose_poses` vs `lie.se3.compose` 7e-7, `rotate` 2e-15),
body index == joint index on Panda and G1, `.viser` structure, layer-dependency contracts, perspective-correct
double-sided depth. Found and fixed in `1306ed5`:

1. **Major** — anchored raster blocks lost the last pixel row/column of small triangles when `frac(lo) > 0.5`
   (19 holes in a 1080p Panda frame). Blocks are now one pixel larger than their class bound; regression test
   compares four triangles with a brute-force edge test pixel for pixel.
2. Minor — pixels left of/above the image were clamped onto row/column 0; now rejected.
3. Minor — no lock between the playback thread and `add`/`remove_group`; `Viewer` now holds an `RLock`, and
   `Timeline.play` cannot start two threads.
4. Minor — joints without a finite effort limit (free-flyer) were drawn at full torque colour; now base colour;
   `tau` frame count is validated.
5. Nit — `Camera.view_matrix` NaN when looking along `up`; fallback axis added.

Not reviewed by the reviewer: `SkinnedMeshLayer`, `CameraLayer`, picking, `PlotPanel` (covered by unit tests
only); no browser was available to compare the rendered image with the client view.

## Assumptions (made off-hours)

- Default host `127.0.0.1` (session rule), `host="0.0.0.0"` exposes.
- No new dependencies: offline rendering is our torch rasterizer + PIL; MP4 only through an `ffmpeg` binary.
- Old viewer deleted per the approved roadmap; old examples 01/02/04/05 keep the old `tasks.solve_ik` but use the
  new viewer (they go when the old optim goes). `(S, T, nq)` side-by-side batches are not a feature: use groups
  with `offset=`.
- `PlotPanel` marks the current frame with a point per series (uPlot has no programmatic vertical cursor).
- Rigid meshes are one node each; batched instancing only for spheres/skeleton/arrows/frames/points.
- Triangles partly behind the camera are dropped (no clipping); the ground is split into 1 m quads so only the
  quad under the camera can vanish.

## Open for the owner

- MP4 needs `ffmpeg` on PATH (not installed here); `.viser` files were parsed but not opened in the static client.
- Message count per frame for mesh-heavy robots (see numbers) if 60 fps with several robots is a goal.
- P4 (BetterHuman) will add `human_layers` on `SkinnedMeshLayer` and move the studio onto this `Viewer`.
- GPU addendum for P2 landed meanwhile (`../2026-10-07_00-39_p2_model_kin_dyn/REPORT.md`, commit `8f8cfed`).
