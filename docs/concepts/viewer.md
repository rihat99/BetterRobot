# Viewer

Visualization must not become a dependency of robot math, so the viewer is the
top layer of BetterRobot: it reads `Model` and kinematic states; no kinematics,
dynamics, term, or optimizer module imports it.

Interactive rendering uses [viser](https://viser.studio/main/), installed
through the `viewer` extra. viser is imported only when a `Viewer` is created;
mesh files go through `trimesh` only in the robot adapter.

## Viewer, timeline, layers

The viewer has three parts and a frame loop:

- A `Layer` owns preloaded `(T, ...)` numpy arrays and the scene handles it
  creates once in `build(scene, root)`. `set_frame(k)` writes slice `k` into
  those handles; a static layer has `T = 1`.
- The `Timeline` holds the current frame, `fps`, play/pause/loop/speed, and
  advances by wall clock, so slow pushes drop frames instead of slowing
  playback.
- The `Viewer` owns the viser server, groups of layers (`/<group>/<layer>` in
  the scene tree, one GUI folder per group) and the sidebar. `show(k)` is the
  only place frames are pushed: every visible layer receives
  `min(k, T - 1)` inside one `server.atomic()` block; layers already at that
  frame are skipped, hidden layers are skipped and catch up when shown again.
  `show()` with no argument re-pushes the current frame after an in-place edit.

Built-in layers (all batched: one or two scene nodes each, per frame only
array writes):

| Layer | Data | Draws |
|---|---|---|
| `SkeletonLayer` | joints `(T, J, 3)`, `parents (J,)` | spheres at joints, cylinders along bones; colours per frame for torque |
| `ArrowsLayer` | origins and vectors `(T, N, 3)` | forces, contacts; zero vectors are hidden so `N` stays fixed |
| `FramesLayer` | poses `(T, F, 7)` xyzw | coordinate axes |
| `PointsLayer` | points `(T, N, 3)` or `(N, 3)` | point cloud |
| `MeshLayer` | vertices `(T, V, 3)` or `(V, 3)`, faces | one static or deforming mesh |
| `RigidMeshLayer` | mesh specs + body poses `(T, B, 7)` | URDF/MJCF geometry following bodies |
| `SpheresLayer` | centers `(T, S, 3)`, radii | `RobotSpheres` collision models |
| `GroundLayer` | — | reference grid |
| `PlotPanel` | series `(T, K)` | sidebar chart with a marker at the current frame |

Layers never see a `Model`. The robot adapter in `viewer/robot.py` is the
only file that does: `RobotView(model, q)` runs forward kinematics once for
the whole clip `q (T, nq)` and returns mesh, skeleton, frame and sphere layers
(`robot_layers(...)` is the list form). `RobotView.update(q)` rewrites the
same arrays in place for live tools; `tau (T, nv)` colours the skeleton joints
by `|tau| / effort_limit`.

## Tools

Tools hold a `Viewer` and callbacks; they are not layers and sit outside the
frame loop. `PoseGizmo` wraps transform controls and reports poses in
BetterRobot order. Dragging never runs IK on its own: the callback decides, as
in `examples/viewer/03_ik_gizmo.py`. `viewer.tools.picking` has ray, sphere
and screen-rectangle selection over vertices with viser's camera conventions.

## Recording without a browser

Every layer also exposes `geometry(frame)`: the same arrays as world
triangles. `viewer.render` rasterizes them with a small tiled z-buffer in
plain torch (CPU or CUDA), so `Viewer.record("clip.gif")` and `.png` frames
need no browser or GPU library; `.mp4` uses `ffmpeg` when it is on `PATH`.
`Viewer.save("clip.viser")` records every frame into viser's file format,
which plays in the static client without Python.

## Theme

One `Theme` dataclass carries the palette (robot, joint, link, force, torque,
target, grid) and the lighting the viewer applies at start-up; `LIGHT` and
`DARK` are the defaults. Layers take colours as arguments, so a theme is a
set of defaults, not a global.

## Dependency boundary

The viewer may depend on every lower layer; nothing may depend on the viewer.
The core (`viewer.py`, `timeline.py`, `layer.py`, `layers/*`) imports only
numpy, torch and viser handles; viser itself is imported in `viewer.py` only,
trimesh in `robot.py` only, and neither pyrender nor imageio anywhere.
Contract tests enforce this. The default host is loopback (`127.0.0.1`); pass
`host="0.0.0.0"` to expose a server.

For a complete setup, use {doc}`/guides/visualize`. For how visual geometry
reaches a model, read {doc}`parsers_and_ir`.
