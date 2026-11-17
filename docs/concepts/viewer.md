# Viewer

The viewer shows robots, bodies and data clips in a browser, and records the same scene through a headless
Chromium to an `.mp4` or `.webm` video or a `.png` still. After this page you will know its four ideas (layers, tags, one timeline, headless record)
and which call to reach for: `show`, `serve`, `save` or `record`.

It lives in `better_robot.viewer`, is built on [viser](https://viser.studio), and is the top of the import stack:
nothing else in the library imports it. viser is imported only when you create a `Viewer`.

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/viewer_layers.png" width="960"><source src="../../clips/viewer_layers.webm" type="video/webm"></video><figcaption>One scene built up layer by layer: mesh, then skeleton, frames and ground, each switched on with <code>set_visible</code>.</figcaption></figure>

## A layer holds a whole clip

A `Layer` is one named thing in the scene: a point cloud, a skeleton, rigid meshes, coordinate frames, arrows.
You give it every frame up front as numpy arrays with a leading time axis `(T, ...)`. Static input counts as
`T = 1`. The layer keeps the arrays **by reference**, so editing them in place and calling `layer.redraw()` updates
the scene without rebuilding anything.

![The Viewer shows layers that each hold a whole clip; `record` draws the same scene in a headless Chromium.](../figures/viewer_layers.svg)

```python
import numpy as np
from better_robot import Viewer
from better_robot.viewer import PointsLayer

viewer = Viewer()                        # viser server on http://127.0.0.1:8080, with a ground grid
cloud = np.random.rand(100, 50, 3)       # T = 100 frames of 50 points
points = viewer.add(PointsLayer("cloud", cloud))   # one layer, or an iterable of layers
print(viewer.url, viewer.timeline.nsteps)
```
```output
http://127.0.0.1:8080 100
```

The built-in layers are `RobotVisual` (a whole robot, below), `PointsLayer`, `LinesLayer`, `FramesLayer`,
`ArrowsLayer`, `GroundLayer`, `RigidMeshLayer`, `MeshLayer`, `SpheresLayer`, `SkinnedMeshLayer` and `CameraLayer`. Each layer of data takes its name and
its main array by position and every other argument by keyword, as in `SpheresLayer("contacts", centers, radii=radii)`.
`viewer.add_plot` adds a
`PlotPanel` of time series to the GUI, and `ImagePanel(name, images)` puts an image per step there, such as the input
video (`(T, H, W, 3)` uint8, or a callable `step -> image` with `nsteps=`). `PointsLayer`, `LinesLayer`,
`SpheresLayer` and `ArrowsLayer` take one colour, one per element `(N, 3)`, or one per step `(T, N, 3)`, such as a
contact turning red or green. A list of `(N_t, 3)` arrays makes a `PointsLayer` whose size changes per step
(detections, a growing map); a step without points hides it. `LinesLayer(name, points)` draws segments
`(T, N, 2, 3)` (start, end) `width` pixels wide, in the theme's text colour unless given `colors`.
`MeshLayer(name, vertices, faces=, colors=(V, 3))` colours each vertex: a static mesh blends them across each face,
a deforming one draws one scene node per distinct colour (at most 64), each face in its first vertex's colour.
`CameraLayer(name, poses, intrinsics=K, size=(W, H))` draws a pinhole camera's frustum from its intrinsics, an
off-centre principal point included, cut `scale` metres in front of the camera, with an optional image on the cut.
`ArrowsLayer` draws viser's solid arrows, a cylinder of `shaft_radius` and a cone of `head_radius` and `head_length`
(metres, the same for every arrow), so forces keep their thickness at any zoom; an arrow shorter than its head is all
head. An arrow of at most `min_length` metres (default `1e-6`: exact zeros and float noise) is left out of the step;
raise it to hide forces too small to read. A new layer type subclasses `Layer` and implements
`build` (create scene nodes once), `set_step` (move them to step `k`) and optionally `bounds` (the box
`aim` frames), `apply_theme` and `build_gui` (its controls in the sidebar).

## A tag is the unit you toggle, move and remove

Every layer carries tags, by default its own name; `viewer.add(layers, tags="...")` gives every layer of the call
those tags instead. A layer lives under the scene node `/<first tag>/<layer.name>`. `remove`, `set_visible` and
`redraw` take a layer or a tag (every layer carrying it), and once there are two tags the sidebar's Layers section
has one checkbox per tag, labelled in sentence case (the ground's tag `ground` and tags that hold only sidebar
charts or images are not listed). `remove` shrinks the timeline to the longest
remaining layer, so a clip browser can remove one clip's tag before adding the next. Layer names must be unique
under a first tag, and a tag that `set_offset` or `remove` does not know raises `UnknownNameError`.

A layer is changed through its own methods, never its fields: `viewer.set_visible(layer, False)` hides it,
`layer.set_opacity(0.4)` fades a mesh layer (`RigidMeshLayer`, `MeshLayer`, `SpheresLayer`, `SkinnedMeshLayer`; a
layer without surfaces raises `ArgumentError`), and `viewer.replace(old, new)` swaps the data a layer draws, such
as a body switched in a studio: `new` takes `old`'s tags, place, sidebar controls and visibility, the current step is
pushed and the camera stays.

```python
viewer.add(PointsLayer("noise", 0.01 * np.random.randn(1, 200, 3)), tags="debug")
viewer.remove("debug")
```

`add(..., offset=(x, y, z))` moves the layers' first tags, in the browser and in `record`, which is how you put two
clips side by side; `set_offset(tag, (x, y, z))` moves a tag again later, such as from a spacing slider.

## The sidebar, the camera and the theme

`viewer.sidebar` holds the controls in sections with a fixed order: Timeline, Robot, Layers, Plots, then the sections
an application creates, Info last. The Timeline section, shown once the timeline has more than one step, holds a
Play box, the Step slider, a Speed choice (0.25x to 2x, taken while playing) and Loop. `sidebar.checkbox`, `dropdown`, `slider` and `button` take a section and a label
and return viser's control; asking again for the same section and label returns the same control and adds the
callback, so one control drives every layer that asked for it, and it goes when the last of them is removed. Every
callback runs under `viewer.lock`, the one lock of adding, removing and pushing steps. A callback that raises a
`BetterRobotError` shows it as a red notification in the browser and the control returns to its previous value.

Until you call `aim`, the viewer points the browsers' camera at the scene after each `add`: the boxes of the visible
layers' `Layer.bounds` over the clip (at most 32 steps are read), centred in a 3/4 view at the nearest distance where
every corner of every box lies inside four fifths of a 16:9 view. `aim(steps=, azimuth=, elevation=, fov=)` chooses
the view yourself.

## One timeline drives every layer

The `Viewer` owns one `Timeline` of integer steps at `fps`. Its length follows the longest layer; a layer
shorter than the timeline holds its last step. `show(k)` is the only place steps reach the layers: every visible
layer gets its sample for step `k` inside one atomic server update, and layers already on that sample are skipped.
`show()` with no argument re-pushes the current step to every layer; `layer.redraw()` re-pushes one layer, which is
what you call after editing its arrays in place.

A layer recorded at another rate passes its sample times in seconds: `viewer.add([layer], times=t)` with `t` of
shape `(T,)`, non-decreasing. Step `k` then shows the latest sample at or before `k / fps`, so a 10 Hz layer and a
30 Hz layer play in sync on a 30 fps timeline. Without `times` (the default) sample `k` is shown at step `k`. The
layer itself still receives a plain integer sample index in `set_step`. `viewer.set_fps(f)` changes the rate of the
timeline itself, keeping one step per frame of the clip on screen; a clip browser calls it when it switches clips.

```python { .no-run data-reason="names a clip, markers_10hz, that this page does not build" }
viewer.add([PointsLayer("mocap", markers_10hz)], times=np.arange(len(markers_10hz)) / 10.0)
```

`viewer.add([layer], valid=v)` with a bool `v` of shape `(T,)` hides the layer on the samples where `v` is false, such
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
| `serve()` | plays by wall clock at `fps` and blocks until Ctrl-C | yes, to watch |
| `save(path)` | writes a `.viser` recording for viser's static web client | no |
| `record(path)` | draws the scene in a headless Chromium into an `.mp4` or `.webm` video or a `.png` still | no |

```python { .no-run data-reason="blocks until Ctrl-C" }
viewer.serve()  # plays and blocks until Ctrl-C
```

## A robot is one layer

`RobotVisual(model, q)` is a whole robot: it runs forward kinematics once over `q`, one configuration `(nq,)` or a
clip `(T, nq)` (`None` is `model.q_neutral`), and draws the result in one of three styles.

| Style | Draws | Needs |
|---|---|---|
| `"meshes"` | the display shapes of a `Geometry` (`io.load_geometry`), or a `Skin` | `geometry=` with display shapes, or `skin=` |
| `"maximal"` | a skeleton with one glyph per joint, bones, body hulls and end blocks | nothing but the model |
| `"minimal"` | a stick figure: one ball per cluster of close joints, sticks between | nothing but the model |

`style=None` picks `"meshes"` when there is something to show as meshes and `"maximal"` otherwise. Mesh files are
looked up under `root`, the description file or its folder, and under each folder above it, so `root=` is the path
you loaded. `collision=True` overlays the collision model, the `spheres` (a `RobotSpheres`) when given, otherwise
the collision shapes of the geometry; `frames=True` draws a triad at every moving joint and `frames=[...]` at
the frames listed, by id or name.

```python
import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot.viewer import RobotVisual

URDF = panda_description.URDF_PATH
model = br.load(URDF)
geometry = br.io.load_geometry(URDF, model)
phase = torch.linspace(0.0, 2 * torch.pi, 30)[:, None]
mid = (model.lower_position_limit + model.upper_position_limit) / 2
half = (model.upper_position_limit - model.lower_position_limit) / 2
q = mid + 0.5 * half * torch.sin(phase + torch.arange(model.nq) * 0.7)   # q (30, nq)

robot = viewer.add(RobotVisual(model, q, geometry=geometry, root=URDF, frames=["panda_hand"]))
ghost = viewer.add(RobotVisual(model, q, name="ghost", style="minimal"), offset=(0.0, 1.0, 0.0))
ghost.update(q.flip(0))            # the same clip played backwards: one kinematics pass, then a redraw
robot.set_style("maximal")         # built now, the first time it is shown; "meshes" stays in the scene, hidden
print(robot.style, robot.styles, ghost.styles)
```
```output
maximal ('meshes', 'minimal', 'maximal') ('minimal', 'maximal')
```

Each part (a style, the collision overlay, the frames) is built the first time it is shown and only hidden after
that, so switching back creates nothing, and a step is written only to the parts on screen. `set_style`,
`set_collision` and `set_frames` switch them from code; the sidebar's Robot section has the same three controls,
shared by every robot in the viewer, so a switch from code does not move them. `update(q)` shows new configurations of the same number of steps, for a live
solver; a clip of another length is a new visual (`viewer.replace(old, new)`). Each robot takes the theme's next
hue, in the order the robots are added, unless `hue=` names one. The name defaults to the model's, so two robots of
one model need a `name=` each.

### The skeleton

The two skeleton styles are laid out by a `SkeletonRecipe`; `MAXIMAL` and `MINIMAL` are the presets and
`RobotVisual(maximal=, minimal=)` takes your own. Sizes come from the robot: every radius is a fraction of the
robot's size, so a hand and a humanoid read the same on screen. The layout is computed once, at `model.q_neutral`,
and every block is rigid in one joint frame, so a step only moves blocks.

`MAXIMAL` draws what the joints are:

- a revolute joint is a drum along its axis, with dark end faces that show which way the axis points; a prismatic
  joint is a sleeve with a rail sliding through it; a spherical joint is a ball; any other kind is a small ball,
  and a free base is a cube;
- joints that share a point, such as a hip's pitch, roll and yaw, are drawn apart: each drum that would overlap
  another slides along its own axis, so it stays on its true axis line, and only then shrinks;
- a bone joins each joint to its parent, thinner than either glyph (`bones`, the ratio; 0 draws none);
- a hull, a rounded box, stands for the root body and for every body where the tree branches, and an end block
  for every body that ends a chain (hands, feet, fingers). Their sizes come from the sphere set when the visual
  has one, else from the collision shapes of the geometry, else from the inertias. `hulls="all"` also boxes the
  bodies with one child joint, `"none"` draws neither; `body_hull={frame: bool}` decides one body.

`MINIMAL` draws the least that still reads as the robot: joints closer than `cluster` times the robot's size share
one ball (a three-joint shoulder is one ball), a joint that only twists its limb gets none (`twists=False`), and
sticks join the balls.

Both take `limits=True` (an arc over each bounded revolute joint's range with a tick at the current angle),
`sides=True` (the left side lighter, the right darker), `size` (a multiplier of every glyph), `joint_size={joint:
multiplier}` (0 hides a glyph) and `color={joint or frame: rgb}`.

```python
from better_robot.viewer import MAXIMAL, SkeletonRecipe

recipe = SkeletonRecipe(limits=True, sides=True, joint_size={"panda_finger_joint2": 0.0})
viewer.add(RobotVisual(model, q, name="limits", maximal=recipe), offset=(0.0, -1.0, 0.0))
print(MAXIMAL.glyphs, MAXIMAL.hulls, recipe.limits)
```
```output
kind auto True
```

A skeleton is at most three batched nodes, one per unit mesh (cylinder, ball, box), plus one line node for the
limit arcs; with the sphere overlay and the frames a robot is at most six drawing nodes, and a step writes
positions and rotations to each node on screen (the sticks of a minimal skeleton also their lengths). `bounds` is
a box per joint and per block of the maximal skeleton in every style, so `Viewer.aim` frames a robot the same way
whatever it is drawn as, and by what it covers: one box round a moving robot is about twice that on screen.

### A skinned surface

`Skin(vertices, faces, weights=, bind=)` is a surface that deforms with the joints: `vertices (V, 3)` in the world at
the configuration `bind (nq,)` (`None` is `model.q_neutral`) and `weights (V, njoints)`. `RobotVisual(model, q,
skin=skin)` shows it as the `"meshes"` style, skinned in the browser.

## Tools: a gizmo and numpy picking

`better_robot.viewer.tools` holds two things. `PoseGizmo(viewer, name, pose=, on_change=...)` is a draggable
transform handle that calls `on_change(pose)` on every drag; `examples/viewer/03_ik_gizmo.py` uses it to re-solve
IK and call `robot.update(q)` live. A callback that raises stops the gizmo, and `gizmo.wait()` raises its error in
your thread. Picking is plain numpy over vertices, so it also works on data you never
showed: `ray_hit`, `nearest_on_ray` (the point nearest the camera within a radius of a click ray),
`vertices_in_sphere` and `vertices_in_rect`.

The viewer holds one pointer callback so that tools never fight over the scene: a tool calls
`viewer.on_pointer("click", callback)` (viser's `SceneClickEvent`: `ray_origin`, `ray_direction`, `screen_pos`) or
`viewer.on_pointer("rect-select", callback)` (`SceneRectSelectEvent`: `screen_min`, `screen_max`), and
`viewer.clear_pointer()` when it is done. While one callback is held, a new registration raises `ArgumentError`:
the tool taking the pointer over calls `clear_pointer()` first. Never register on `viewer.server.scene` directly.

```python
from better_robot.viewer.tools import vertices_in_sphere

vertices = np.random.rand(1000, 3)
picked = vertices_in_sphere(vertices, center=np.array([0.5, 0.5, 0.5]), radius=0.2)
print(picked.shape)
```
```output
(24,)
```

## Recording replays the timeline headless

`record(path)` steps the same timeline in a headless Chromium and draws each step once, so a recording shows
exactly what a browser would; `aim` points the live browsers at the view `record` would choose.
[Record a video of a scene](../guides/record-a-video.md) gives the steps and the options.

## Themes and the up axis

`Viewer(theme=...)` takes a `Theme`, the one source of colours: the backdrop gradient, the grid, the shadow on the
ground, the light rig (ambient, hemisphere, a key light that casts shadows, a fill; no environment map, so meshes and solid
colours share one rig), the collision overlay's colour and opacity, the robots' hues (`theme.hue(k)`
for the `k`-th robot) and the shades of a hue (`theme.shade(hue, "body")`), the GUI accent and the up axis. `LIGHT`
(the default) and `DARK` are built in; `viewer.set_theme(theme)` switches the page and every layer. `Theme(up="+y")` is for data in a y-up convention, such as SMPL bodies; it sets the up
direction of the browser scene and of the `record` camera, and a `GroundLayer` added without an explicit
`plane` lies in the plane perpendicular to it (`"xz"` for `+y`). The default grid has no edges and fades out with
the distance from the camera; `GroundLayer(size=4.0)` draws a square 4 m wide. `GroundLayer(origin=p, normal=n, fill=0.3)`
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

`Viewer(server=...)` reuses an existing `viser.ViserServer`, or takes `better_robot.testing.FakeServer`, the
stand-in that records scene calls without a port or a browser (the tests and the examples' `--check` run on it). It
declares each viser call the viewer makes, and the GUI calls of an app on it (dropdowns, text and number fields,
markdown, tabs), with viser's parameters, and a test of the full tier binds them against viser, so a call viser would refuse fails on the stand-in too; `viewer.server` is the escape hatch for custom GUI. The default host is `127.0.0.1`, so the server is reachable only from your machine;
forward the port over SSH to view it remotely.

## Lazy inputs

The convention for large per-step data is a callable `step -> array` evaluated only when that step is shown.
Today `CameraLayer(image=...)` and `ImagePanel` accept it (decoding video frames on demand); the other layers take eager
`(T, ...)` arrays, because they need every sample up front for `nsteps` and for `bounds`.

## Limits

- One integer timeline per viewer. Per-layer `times` resample onto it; there is no continuous time axis.
- viser is the only backend ([ADR 0004](../contracts/decisions/0004-viser-viewer.md)).
- No GUI brick beyond the sidebar: `viewer.sidebar.gui` (viser's GUI API) inside `with viewer.sidebar.section(name):`
  is the sanctioned API for studio-style panels built on the viewer.
- `record` needs a headless Chromium (`BETTER_ROBOT_CHROME`); `save` does not.

## Where next

- [Visualize a robot](../getting-started/visualize.md) walks through a first scene and a recording.
- [Record a video of a scene](../guides/record-a-video.md): `record`'s formats, options and camera.
- `examples/viewer/` has runnable scripts: a robot pose, an IK gizmo, recording a clip, and five demos of
  avoidance while tracking a dragged target ([Try the avoidance demos](../guides/avoidance-demos.md)).
- [API reference: viewer](../reference/api/viewer.md).
