# Viewer

The viewer shows robots, bodies and data clips in a browser, and records the same scene through a headless
Chromium to `.webm`, `.gif` or PNG frames. After this page you will know its four ideas (layers, groups, one timeline, headless record)
and which call to reach for: `show`, `play`, `save` or `record`.

It lives in `better_robot.viewer`, is built on [viser](https://viser.studio), and is the top of the import stack:
nothing else in the library imports it. viser is imported only when you create a `Viewer`.

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/viewer_layers.png" width="960"><source src="../../clips/viewer_layers.webm" type="video/webm"></video><figcaption>One scene built up layer by layer: mesh, then skeleton, frames and ground, each switched on with <code>set_visible</code>.</figcaption></figure>

## A layer holds a whole clip

A `Layer` is one named thing in the scene: a point cloud, a skeleton, rigid meshes, coordinate frames, arrows.
You give it every frame up front as numpy arrays with a leading time axis `(T, ...)`. Static input counts as
`T = 1`. The layer keeps the arrays **by reference**, so editing them in place and calling `viewer.show()` updates
the scene without rebuilding anything.

![The Viewer shows layers that each hold a whole clip; `record` screenshots the same scene through headless Chromium.](../figures/viewer_layers.svg)

```python
import numpy as np
from better_robot.viewer import GroundLayer, PointsLayer, Viewer

viewer = Viewer()                        # viser server on http://127.0.0.1:8080
cloud = np.random.rand(100, 50, 3)       # T = 100 frames of 50 points
points = viewer.add(PointsLayer("cloud", cloud))
viewer.add(GroundLayer(size=3.0))
print(viewer.url, viewer.timeline.nsteps)
```
```output
http://127.0.0.1:8080 100
```

The built-in layers are `PointsLayer`, `SkeletonLayer`, `FramesLayer`, `ArrowsLayer`, `GroundLayer`,
`RigidMeshLayer`, `MeshLayer`, `SpheresLayer`, `SkinnedMeshLayer` and `CameraLayer`; `viewer.add_plot` adds a
`PlotPanel` of time series to the GUI, and `ImagePanel(name, images)` puts an image per step there, such as the input
video (`(T, H, W, 3)` uint8, or a callable `step -> image` with `nsteps=`). `PointsLayer`, `SkeletonLayer`,
`SpheresLayer` and `ArrowsLayer` take one colour, one per element `(N, 3)`, or one per step `(T, N, 3)`, such as a
contact turning red or green. `ArrowsLayer` draws line shafts `line_width` pixels wide by default; with
`shaft_radius=` (metres) its arrows are solid, a cylinder and a cone of length `head_size` and radius
`head_radius`, so forces keep their thickness at any zoom; a head takes at most 35 % of its arrow, so a short solid
arrow is the long one scaled down, never a lone head. An arrow no longer than `min_length` metres (default
`1e-6`: exact zeros and float noise) draws nothing in either mode, neither a dot nor a lone head; raise it to hide
forces too small to read. `SkeletonLayer` takes a joint `radius` per joint `(J,)`
or per step `(T, J)`, a `thickness` `(J,)` or `(T, J)` that multiplies each marker's cross-section but not its
length, `offsets (J, 3)` in each joint's frame that move a marker off its joint origin (bones then join the drawn
centres), and a `bone_radius` per bone `(J,)`, entry `j` for the bone into joint `j` (thin finger bones); a radius of
0 hides that marker or bone. Given joint `kinds`, a bone with an axis marker at either end bends at right angles
(see the robot skeleton below); bones between axis-less joints, such as every bone of a positions-only skeleton,
are straight. A new layer type subclasses `Layer` and implements
`build` (create scene nodes once), `set_step` (move them to step `k`) and optionally `world_points` (positions
that `record`'s camera fits).

## A group is the unit you toggle and remove

`viewer.add(layer, group="...")` puts the layer under the scene node `/<group>/<layer.name>`. The default group is
`"scene"`. Each group gets a GUI folder with one visibility checkbox per layer, and `remove_group` tears the whole
group down and shrinks the timeline to the longest remaining layer, so a clip browser can swap one clip's group for
the next. Layer names must be unique inside a group.

```python
viewer.add(PointsLayer("noise", 0.01 * np.random.randn(1, 200, 3)), group="debug")
viewer.remove_group("debug")
```

`add(..., offset=(x, y, z))` moves the whole group, in the browser and in `record`, which is how you put two
clips side by side; `set_offset(group, (x, y, z))` moves it again later, such as from a spacing slider.

## One timeline drives every layer

The `Viewer` owns one `Timeline` of integer steps at `fps`. Its length follows the longest layer; a layer
shorter than the timeline holds its last step. `show(k)` is the only place steps reach the layers: every visible
layer gets its sample for step `k` inside one atomic server update, and layers already on that sample are skipped.
`show()` with no argument re-pushes the current step, which is what you call after editing arrays in place.

A layer recorded at another rate passes its sample times in seconds: `viewer.add(layer, times=t)` with `t` of
shape `(T,)`, non-decreasing. Step `k` then shows the latest sample at or before `k / fps`, so a 10 Hz layer and a
30 Hz layer play in sync on a 30 fps timeline. Without `times` (the default) sample `k` is shown at step `k`. The
layer itself still receives a plain integer sample index in `set_step`. `viewer.set_fps(f)` changes the rate of the
timeline itself, keeping one step per frame of the clip on screen; a clip browser calls it when it switches clips.

```python { .no-run }
viewer.add(PointsLayer("mocap", markers_10hz), times=np.arange(len(markers_10hz)) / 10.0)
```

`viewer.add(layer, valid=v)` with a bool `v` of shape `(T,)` hides the layer on the samples where `v` is false, such
as a person who is not tracked in that frame; its sidebar switch stays on. A `(T, N)` mask hides single spheres of a
`SpheresLayer` or arrows of an `ArrowsLayer`. Like the layer's arrays, `v` is kept by reference.

```python
viewer.show(10)
cloud[:, :, 2] += 0.5   # edit the clip in place ...
viewer.show()           # ... and re-push step 10
```

The other three calls consume the timeline:

| Call | What it does | Needs a browser |
|---|---|---|
| `play()` | plays by wall clock at `fps` and blocks until Ctrl-C; `play(block=False)` returns | yes, to watch |
| `save(path)` | writes a `.viser` recording for viser's static web client | no |
| `record(path)` | screenshots the scene in a headless Chromium to `.webm`, `.gif` or `.png` frames | no |

```python { .no-run }
viewer.play()  # blocks until Ctrl-C
```

## A robot becomes layers with one kinematics pass

`robot_layers(model, q)` turns a `Model` and a clip `q (T, nq)` into rigid-mesh, skeleton, frame and sphere layers.
It runs forward kinematics once for the whole clip and hands the joint poses to the layers.
`RobotView` holds the same layers and adds `update(q)`, which rewrites them in place for a new clip of the
same length; follow it with `viewer.show()`.

```python
import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot.viewer import RobotView, robot_layers

model = br.load(panda_description.URDF_PATH)
phase = torch.linspace(0.0, 2 * torch.pi, 30)[:, None]
mid = (model.lower_pos_limit + model.upper_pos_limit) / 2
half = (model.upper_pos_limit - model.lower_pos_limit) / 2
q = mid + 0.5 * half * torch.sin(phase + torch.arange(model.nq) * 0.7)   # q (30, nq)

for layer in robot_layers(model, q, skeleton=True, frames=["body_panda_hand"]):
    viewer.add(layer, group="panda")

robot = RobotView(model, q, name="ghost", visual=False, skeleton=True)
for layer in robot:
    viewer.add(layer, group="ghost", offset=(0.0, 1.0, 0.0))
robot.update(q.flip(0))   # same length, played backwards
viewer.show()
```

Options of both: `visual` and `collision` meshes, `skeleton`, `frames` (names or ids), `tau` (the joint load on the
skeleton), `spheres` (collision spheres), `theme`, `opacity`. The skeleton draws each revolute joint as a
cylinder along its axis (radius 25 mm, length 60 mm) and each prismatic joint as a thin box along it, with the axes
from `model.structure.joint_axes`. Joints that share a point, such as a hip's pitch, roll and yaw, are drawn apart
as in a textbook kinematic diagram: `spread_offsets(model)` slides each marker that would overlap another along its
own axis (so it stays on the true axis line), once at `model.q_neutral` and for markers of the plain thickness, and
the bones (8 mm) connect the drawn markers in chain order. A bone bends at right angles instead of running
diagonally: it is up to three legs along the axes of its parent joint's frame, first along the parent's axis, last
along the child's (so it enters a cylinder along its axis or across it, never at a slant), the rest between,
joined by spheres; a leg under 1 cm folds into the longest. A bone over 15 cm from a joint with one child, such as a
humanoid's thigh, stays straight, since a long bone's small offsets would otherwise draw a long jog. The routing is
chosen once; each step the legs split the bone's current centre difference, so they always meet the child.

When you already have the joint poses `(T, njoints, 7)` (`KinematicState.joint_pose`),
`robot_skeleton(model, joint_poses, bone_radius=...)` builds the skeleton from them without another kinematics
pass; a `bone_radius` of 0 hides a bone, such as the ones to massless sensor mounts
(`model.body_inertias[1:, 0] == 0`). With `tau` `(T, nv)` each joint marker shows the joint's load
`|tau| / effort_limit` in two ways: its colour runs through the theme's `torque_ramp` (navy, teal, green, yellow)
up to the limit and turns `over_limit` (magenta) past it, and its cross-section grows from 0.8 to 1.5 times the
default at the limit (its length stays, and no marker moves; the layout is spaced for the plain thickness, so a
heavily loaded marker may touch its neighbour). Joints
without a finite effort limit keep the plain look. `end_points` maps frames (names or ids) to a point in each
frame; a bone runs from the frame's parent joint to that point, as thick as the bone into that joint, such as a
hand past the last wrist joint out to where its contact forces act.

```python
from better_robot.viewer import robot_skeleton

poses = br.forward_kinematics_state(model, q).joint_pose.numpy()   # (30, njoints, 7)
tau = 40.0 * torch.sin(phase + torch.arange(model.nv))              # (30, nv) N·m
fingers = {"body_panda_hand": (0.0, 0.0, 0.1)}                       # a bone out to the fingertips
viewer.add(robot_skeleton(model, poses, name="panda load", tau=tau, end_points=fingers), group="panda")
```

## Tools: a gizmo and numpy picking

`better_robot.viewer.tools` holds two things. `PoseGizmo(viewer, name, pose, on_change=...)` is a draggable
transform handle that calls `on_change(pose)` on every drag; `examples/viewer/03_ik_gizmo.py` uses it to re-solve
IK and call `robot.update(q)` live. Picking is plain numpy over vertices, so it also works on data you never
showed: `ray_hit`, `nearest_on_ray` (the point nearest the camera within a radius of a click ray),
`vertices_in_sphere` and `vertices_in_rect`.

viser has a single scene-pointer slot, so the viewer owns it: a tool calls `viewer.on_pointer("click", callback)`
(or `"rect-select"`) and `viewer.clear_pointer()` when it is done. A new registration replaces the previous one;
never call `viewer.server.scene.on_pointer_event` directly.

```python
from better_robot.viewer.tools import vertices_in_sphere

vertices = np.random.rand(1000, 3)
picked = vertices_in_sphere(vertices, center=np.array([0.5, 0.5, 0.5]), radius=0.2)
print(picked.shape)
```
```output
(37,)
```

## Recording the scene

`viewer.record(path)` opens the scene in a headless Chromium (`better_robot.viewer.capture`), shows each
step and screenshots it. The output format follows the suffix: `.webm` (needs `ffmpeg`), `.gif`, `name.png`
for numbered frames, or a directory of frames. Options are `steps`, `size` (width, height), `fps`, `gpu`
and `camera`.

```python { .no-run }
import tempfile
from pathlib import Path

out = Path(tempfile.mkdtemp()) / "panda.webm"
viewer.record(out, size=(960, 540), steps=range(0, 60, 2))
print(out.exists())
```

Without a `camera`, `record` fits a 3/4 view to the visible layers' `Layer.world_points` over the recorded
steps, upright along the theme's `up` axis. Pass `camera=(position, look_at)` for a fixed view. The browser
binary comes from `$BETTER_ROBOT_CHROME` or the `PATH`; the module docstring of `capture` explains the
one-time setup.

`viewer.aim(steps=...)` points the live browsers with the same 3/4 view: the connected ones and every new one
(viser's initial camera, which "Reset View" returns to). It fits the bounding box of what is visible at each step,
so a clip of robot meshes frames in milliseconds. It frames the current step by default; pass `steps`, such as
`range(0, viewer.timeline.nsteps, 10)`, to frame a whole clip, and `azimuth=`/`elevation=` to look from another
side.

## Themes and the up axis

`Viewer(theme=...)` takes a `Theme`: GUI and scene colours, lights and the `up` axis. `LIGHT` (the default) and
`DARK` are built in. `Theme(up="+y")` is for data in a y-up convention, such as SMPL bodies; it sets the up
direction of the browser scene and of the `record` camera, and a `GroundLayer` added without an explicit
`plane` lies in the plane perpendicular to it (`"xz"` for `+y`). `GroundLayer(origin=p, normal=n, fill=0.3)`
instead tilts the grid onto the plane through `p` with normal `n`, such as a fitted floor, over a translucent fill
(`fill` is its opacity; 0, the default, draws lines only).

```python
from better_robot.viewer import DARK, Theme

dark_viewer = Viewer(theme=DARK, port=8081)
y_up_viewer = Viewer(theme=Theme(up="+y"), port=8082)
print(y_up_viewer.theme.up_vector)
```
```output
(0.0, 1.0, 0.0)
```

`Viewer(server=...)` reuses an existing `viser.ViserServer` (or a fake one in tests), and `viewer.server` is the
escape hatch for custom GUI. The default host is `127.0.0.1`, so the server is reachable only from your machine;
forward the port over SSH to view it remotely.

## Lazy inputs

The convention for large per-step data is a callable `step -> array` evaluated only when that step is shown.
Today `CameraLayer(image=...)` and `ImagePanel` accept it (decoding video frames on demand); the other layers take eager
`(T, ...)` arrays, because they need every sample up front for `nsteps` and for `world_points`.

## Limits

- One integer timeline per viewer. Per-layer `times` resample onto it; there is no continuous time axis.
- viser is the only backend ([ADR 0004](../contracts/decisions/0004-viser-viewer.md)).
- No GUI brick: `viewer.server.gui` is the sanctioned API for studio-style panels built on the viewer.
- `record` needs a headless Chromium (`BETTER_ROBOT_CHROME`); `save` does not.

## Where next

- [Visualize a robot](../getting-started/visualize.md) walks through a first scene and a recording.
- `examples/viewer/` has runnable scripts: a robot pose, an IK gizmo, recording a clip.
- [API reference: viewer](../reference/api/viewer.md).
