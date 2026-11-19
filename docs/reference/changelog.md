# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each release groups its entries by change type;
every removal names its replacement. The [migration page](migration-0.3.0.md) maps every public name of the 0.2 line
that 0.3.0 renames or removes, one table row each.

## Unreleased

### Added
- `viewer.Sidebar`: the viewer's controls in ordered sections (Timeline, Robot, Layers, Plots, Images, an
  application's sections in creation order, Info). A control is shared by `(section, label)`, so one control drives every layer
  that asks for it; every callback runs under `Viewer.lock`, and a `BetterRobotError` raised in one returns the
  control to its previous value and shows a red notification in the browser. `Viewer.sidebar` is the viewer's one.
- `Sidebar.set_value(section, label, *, value)` shows a value on a control without running its callbacks;
  `RobotVisual.set_style`, `set_collision` and `set_frames` called from code move the Robot section's shared control
  with it and leave the other robots as they are.
- Tags: `Layer(name, nsteps, *, tags=)` (default: the layer's name; the first tag is the scene parent
  `/<tag>/<name>`) and `Viewer.tags`. `Viewer.remove`, `Viewer.set_visible`, `Viewer.set_pose` and
  `Viewer.redraw` address every layer of a tag, and the sidebar's Layers section has one checkbox per tag once there
  are two.
- `Layer.bounds(step)`: the world box `(lo, hi)` of what a layer draws, or one box per part `(lo (N, 3), hi (N, 3))`
  for a layer of thin, spread-out parts, which `Viewer.aim` and `Viewer.record` frame
  (over at most 32 sampled steps, by default 32 samples of each layer's clip; `Viewer.aim(fov=)`). Until `aim` is
  called, the viewer aims the camera after each `add`; it keeps each sample's boxes until `redraw`, `show()` or
  `set_visible` (a step change, playback and a longer timeline keep them) and fits the boxes without listing their
  corners, so the sixteenth G1 added takes about what the first does (80 and 59 ms, maximal style, 300 steps, also
  with a step shown between adds or while playing; the sixteenth took 1.7 to 2.5 s).
- `Layer.redraw()` and `Viewer.redraw(target)` push the current step again after a layer's arrays changed in place;
  `Layer.viewer` is the viewer the layer was added to.
- `Viewer.serve()` plays the timeline and serves the page until interrupted; `Viewer.set_theme(theme)` applies a
  theme to the page and to every layer; `Viewer(ground=True)` adds the ground grid under the tag `ground`.
- `Theme` tokens `backdrop` (the page's vertical gradient), `grid`, `shadow`, `lights` (ambient, hemisphere, key,
  fill), `hues` with `Theme.hue(index)`, `Theme.shade(hue, role, *, side=0)` (`side=1` a left part, lighter, `-1`
  a right one, darker), `accent`, `collision` and `text`, all colours RGB in `[0, 1]`, and `collision_opacity`.
  `Theme.apply` switches viser's default lights and environment map off for an ambient, a hemisphere, a key light
  that casts shadows and a fill light, one rig for meshes and solid colours.
- `Theme(sidebar_width="medium")`: viser's `"small"`, `"medium"` or `"large"` sidebar, which `Viewer.set_theme`
  keeps; it put `"medium"` back over an application's own `configure_theme(control_width=)`.
- `testing.FakeServer` keeps the lights, the background image, arrows, button groups, folders and dropdown options,
  and its clients have a camera and notifications; its plain and skinned meshes take `cast_shadow` and
  `receive_shadow`. It keeps a copy of each array it is sent, as viser serialises one, so a test sees an edit in
  place only after the push that sends it.
- `viewer.RobotVisual(model, q, *, name, geometry, root, spheres, skin, style, minimal, medium, maximal, hue,
  collision, frames, tags)`: a robot as one layer. It runs forward kinematics over `q (nq,)` or `(T, nq)` and draws
  the style `"meshes"` (the display shapes of a `Geometry`, or a `Skin`), `"minimal"`, `"medium"` or `"maximal"`
  (skeletons), with a collision overlay (the sphere set, else the collision shapes, a pale see-through shell over any
  style) and frame triads. Each part is built the first time
  it is shown; `set_style`, `set_collision`, `set_frames` and the sidebar's Robot section switch them while the
  viewer runs, and `update(q)` shows new configurations. A robot takes the first of the theme's hues that no other
  robot of the viewer is drawn in, so one that replaces another keeps its hue; `RobotVisual.hue` is the colour it
  is drawn in (record 0058).
- `viewer.SkeletonRecipe` with the presets `viewer.MAXIMAL` (a glyph per joint kind, slim tapered limbs as wide as
  their link up to the size its mass gives, trunks for the bodies that branch, and where a chain ends a sole, a
  finger block or a limb that tapers to the end of its body, with no ball at its tip, in two tones; no ball but a
  spherical joint's among the drums, and a head only above a trunk; the drums of joints that share a point, such
  as a hip, cross as a universal joint's do, and any other glyph that would overlap slides along its own axis, at
  most half its length, then shrinks), `viewer.MEDIUM` (the same glyphs, laid out by the same rules, on thin pale
  rods instead of bodies, a rod on to the end of each chain closed by a round cap of its width, and a drum outside
  a compound at most 0.35 of its shortest rod: `hulls="none"` with `glyphs="kind"`, `bones=0.35`, `size=0.7`) and
  `viewer.MINIMAL` (one
  ball per cluster of close joints, sticks between, a trunk's head as a small ball on a stick, and from each leaf
  joint a stick on to the far end of its body with a small ball, so a quadruped stands on its feet and a humanoid
  has feet and hands; the styles share one size rule, a glyph as large as the mass it carries, 26 mm times the
  cube root of its subtree's mass in kg, kept between 0.5 and 1.2 of a reference radius between 2 % and 3 % of the
  robot's size, so a chain narrows towards its tip, and size bodies from the collision shapes before a sphere set,
  leaving out with a warning a collision mesh file that cannot be read): `glyphs` (`"kind"` or `"ball"`), `bones`
  (0.6 of the glyph radius by default), `hulls`, `limits`, `cluster`, `twists`, `sides`, `size`, `joint_size`,
  `body_hull` and `color`; a `bones`, `cluster` or `joint_size` below 0, a `size` not above 0, or any of them not
  finite raises `ArgumentError`. A skeleton is three batched nodes. `Theme.shade` takes the role `bone` for
  the maximal skeleton's pale bodies and the medium one's rods.
- `viewer.Skin(vertices, faces, *, weights, bind=None)`: a skinned surface a `RobotVisual` shows as its meshes.
- `viewer.load_meshes(geometry, *, root=None, collision=False)`: the display (or collision) shapes of a `Geometry`
  as `MeshSpec` arrays, `vertices (V, 3)`, `faces (F, 3)`, `body` and `origin`, for code without a viewer; mesh files
  are found as `RobotVisual(root=)` finds them, by the one lookup the viewer uses. A `geometry` that is not a
  `Geometry` (a `Model`) or a `collision` that is not a bool raises `ArgumentError`.
- `viewer.LinesLayer(name, points, *, colors=None, width=2.0, tags=())`: segments `(T, N, 2, 3)` or `(N, 2, 3)`,
  one colour, one per segment or one per step, in `Theme.text` unless given.
- `MeshLayer(colors=(V, 3))`: one colour per vertex. A static mesh is one GLB node that blends them across each face;
  a deforming one is one node per distinct colour (at most 64), each face in its first vertex's colour.
- `PointsLayer` takes a list of `(N_t, 3)` arrays, a cloud whose size changes per step (`PointsLayer.counts`), with
  one colour or a list of `(N_t, 3)` colours; a step without points hides the cloud.
- `testing.FakeServer.scene.add_glb`, and `testing.FakeServer.get_host()` (loopback), as viser's server has.
- `examples/viewer/00_showcase.py`: one page that shows the G1, the Panda and the Go2 as meshes or as the minimal,
  medium or maximal skeleton, with the collision overlay, the frame triads, the light or dark theme and a looping
  motion.
- `better_robot.viewer.tools` is in the public tree (`viewer.__all__`): the API snapshot, the keyword-only and
  vocabulary tests and the API page cover `PoseGizmo`, `CameraFrustum`, `ray_hit`, `nearest_on_ray`, `project`,
  `vertices_in_sphere` and `vertices_in_rect`; `PoseGizmo` and `CameraFrustum` have a `repr`.

### Changed
- `Viewer.add(layers, *, tags=None, pose=, times=, valid=)` takes one layer or an iterable of layers and returns
  its argument; `tags` replaces `group`, also in `Viewer.add_plot`. `Viewer.set_visible(target, visible)` renames
  its first parameter.
- `Viewer.set_pose(tag, pose)` and `Viewer.add(pose=)` place a tag's frame at a pose `(7,)` `[tx, ty, tz, qx, qy,
  qz, qw]`, or a translation `(3,)` (were `set_offset(group, offset)` and `add(offset=)`, a translation only), so a
  y-up body stands beside a z-up robot without turning its arrays; `aim`, `record` and `save` show the tag placed.
  `Layer.bounds` is the box in the frame of the layer's first tag. A pose must be finite numbers, and a quaternion
  whose norm is within `1e-3` of 1 is normalised (one further off raises). `Viewer.pose(tag)` reads the pose back,
  and the picking functions `ray_hit`, `nearest_on_ray`, `vertices_in_sphere`, `vertices_in_rect` and `project`
  take it as `pose=` to pick on the arrays of a placed tag.
- `Viewer(theme=LIGHT)` (was `None`); `Viewer.replace(old, new)` keeps `old`'s tags and sidebar rows.
- `Layer.build_gui(sidebar)` takes the viewer's `Sidebar` (was viser's GUI API); `PlotPanel` and `ImagePanel` sit in
  the sidebar's Plots and Images sections.
- `GroundLayer(cell_color=None, section_color=None, fill_color=None)`: the grid and its fill take `Theme.grid`
  unless given (the fill its cell colour; was a fixed light grey) and follow `Viewer.set_theme`.
- `Viewer.record(path, *, size=(1920, 1080), fps=None, steps=None, camera=None, fov=radians(40), gpu=None,
  bitrate=None) -> Path` draws each step once in a headless Chromium that runs viser's own client, with the canvas
  pinned to `size` at a pixel ratio of 1, and encodes in the page: `.mp4` is H.264 that ffmpeg only puts in its
  container, `.webm` is VP9 (VP8 where VP9 is refused) and needs no ffmpeg, `.png` is one still; a path without a
  suffix takes `.mp4` when ffmpeg is found, else `.webm`, and the path written is returned. Frame `k` shows
  `steps[k]` (a layer's colours and points too, which arrived one frame late), a camera may move per step, and the
  background is the page's, so a dark theme records a dark clip. `gpu=<nvidia-smi index>` draws on that GPU
  through Vulkan and raises `ArgumentError` when Chromium does not; `gpu=None` draws on the CPU (SwiftShader).
  A 1920x1080 Panda clip of 300 frames takes 42 s on the CPU and 5 s on a GPU, where screenshots took 2.6 to 5 s
  per frame (`docs/guides/record-a-video.md`).
- `Viewer.record` finds ffmpeg through `$BETTER_ROBOT_FFMPEG` (was `$FFMPEG`) or the `PATH`.
- The `viewer` extra pins `viser>=1.1.1,<1.2`: `Viewer.record` drives the client of viser 1.1, and a test names
  the viser version when the internals it uses change. On viser 1.1.1 a connected browser goes blank when a skinned
  mesh is removed or replaced (its client's bug, fixed upstream in viser pull request 791); reload the page.
- `GroundLayer(size=None)` (was `10.0`): the default grid has no edges and fades out with the distance from the
  camera; shadows fall on it at `Theme.shadow`. A number keeps a square grid of that width.
- `Timeline.build_gui(sidebar)` takes the viewer's `Sidebar` (was viser's GUI API): a Play box, the Step slider, a
  Speed dropdown (0.25x to 2x, taken while playing, no restart) and Loop, run under `Viewer.lock`. A step set while
  playing becomes where playback goes on.
- `Timeline(nsteps, *, fps=30.0, loop=True, lock=None)`: the playback thread advances under `lock` (the viewer's
  `Viewer.lock`) and never waits for it without looking whether playback stopped, so `Timeline.pause`,
  `Viewer.set_fps` and `Viewer.record` may run in a sidebar callback; they hung there.
- The Layers section does not list tags that hold only sidebar panels (`PlotPanel`, `ImagePanel`) and labels its
  checkboxes in sentence case; a `PlotPanel`'s time axis shows seconds instead of a date.
- `Viewer.aim` and `Viewer.record` centre the boxes of the bounds in the view, at the nearest distance where every
  corner of every box lies inside 80 % of a 16:9 view (was the bounding sphere with a 15 % margin, which left the
  robot a fifth of the height), seen from azimuth 60 degrees (was 35): a front three-quarter view of a robot facing
  `+x`. `RobotVisual.bounds` is a box per joint, per collision sphere and collision shape that is not a mesh (exact),
  and per block of the maximal skeleton on a body without one, so a clip of a G1 sweeping its joints fills 73 % of
  the frame's height (one box round the clip left it at 42 %) and the lowest point of a G1 or a Go2 is its soles'
  (within 1 mm of the meshes; the blocks reached 3 to 7 cm below).
- `CameraLayer(name, poses, *, intrinsics, size, scale=0.3, color=None, image=None)` (was `fov=`, `aspect=`): the
  frustum's edges from the intrinsics `K` of a `(W, H)` image, an off-centre principal point included, and the image
  on its cut; viser's own frustum has no principal point. It frames `Viewer.aim` too (`bounds`).
- Colours: every layer draws the sRGB colour it is given. The skeleton, `SpheresLayer`, `ArrowsLayer`, `LinesLayer`
  and `CameraLayer` drew about 100 levels paler than a `MeshLayer` of the same colour, because viser reads their
  colours as linear values. With the true colours `Theme.hues` are lighter mid-tones (green `(116, 190, 140)`, was
  `(82, 160, 104)`) and `Theme.shade` mixes less: `joint` 10 % and `face` 32 % towards black (were 22 % and 58 %),
  `stick` 10 % towards white (was 25 %). `DARK.collision` is a pale glass `(236, 239, 245)` at opacity 0.16: the
  light theme's `(140, 150, 165)` at 0.18 was a dim ghost on the dark backdrop, and a darker or denser grey turns
  to smoke that dims the skeleton it holds.
- A mesh layer below opacity 1 (`MeshLayer`, `RigidMeshLayer`, `SpheresLayer`, `SkinnedMeshLayer`, the collision
  overlay) casts and catches no shadow: the shadow map ignores opacity, so a glass at 0.2 threw a solid shadow and
  wore the shadows of what it holds. `Layer._surface()`, a subclass hook beside `_track`, `_store_opacity` and
  `_fade`, gives the settings a mesh node is drawn with at the layer's opacity.
- `ArrowsLayer` draws viser's arrows (`scene.add_arrows`): `shaft_radius=0.006`, `head_radius=0.016` and
  `head_length=0.04` metres for every arrow, the arrows not drawn at a step left out of it.
- `ArrowsLayer.vectors` holds the vectors as given (was multiplied by `scale`) and `ArrowsLayer.scale` applies at
  each push, so an in-place edit is in the units of the input; a float32 input is no longer copied.
- `tools.PoseGizmo(translation=True, on_top=False)`: `translation=False` leaves only the rings that turn the pose
  (was `disable_sliders=True, disable_axes=True`), `on_top=True` draws the handle in front of every mesh (was
  `depth_test=False`).
- `tools.CameraFrustum(pose, *, fov, aspect)` takes the camera's pose `(7,)` `[x, y, z, qx, qy, qz, qw]`, quaternion
  scalar last like every pose of the library (was `position` and the scalar-first `wxyz`); `from_camera` reorders
  viser's camera, and a `pose` that is not `(7,)` raises `ShapeError`.

### Removed
- `Viewer.add(group=)`, `Viewer.add_plot(group=)` and `Viewer.remove_group`: use `Viewer.add(tags=)` and
  `Viewer.remove(tag)`.
- `Viewer.play`: use `Viewer.serve()`, or `viewer.timeline.play()` to play without blocking.
- `Viewer.rebuild`: use `Layer.redraw()` after an in-place edit, or `Viewer.replace(old, new)` for a new layer.
- `Layer.world_points` and its overrides in the layers: use `Layer.bounds`.
- `Layer.listed` and `GroundLayer.listed`: use `Layer(tags=)`; the Layers section lists tags, never the ground.
- `Theme.brand_color`: use `Theme.accent`.
- `Theme.grid_cell` and `Theme.grid_section`: use `Theme.grid`.
- `Theme.robot` and `Theme.human`: use `Theme.hue(index)`.
- `Theme.point`, `Theme.force`, `Theme.torque`, `Theme.contact` and `Theme.target`: no replacement; no layer read
  the colours, and an application passes its own.
- `Theme.control_width`: use `Theme.sidebar_width`.
- `Viewer.set_offset` and `Viewer.add(offset=)`: use `Viewer.set_pose` and `Viewer.add(pose=)`, which take the same
  translation `(3,)` or a pose `(7,)`.
- `Viewer.record` to `.gif`, to numbered `.png` frames and to a directory of frames: use `Viewer.record` to an
  `.mp4` or a `.webm`, or to one `.png` still per call.
- `viewer.capture` (`HeadlessClient`, `write_frames`, `find_chrome`, `find_ffmpeg`): use `Viewer.record`.
- `viewer.robot_skeleton`, `viewer.robot_meshes` and `viewer.SkeletonLayer`: use `viewer.RobotVisual`, which draws
  the meshes and both skeletons of a model and runs the forward kinematics itself, and `viewer.load_meshes(geometry,
  root=)` for the meshes as arrays. Colouring joints by torque
  (`robot_skeleton(tau=)`) and `end_points=` have no replacement.
- `Theme.joint`, `Theme.link`, `Theme.torque_ramp` and `Theme.over_limit`: use `Theme.hue` and `Theme.shade`; a
  skeleton takes its robot's hue and the shades of it.
- `ArrowsLayer(head_size=, line_width=)` and `ArrowsLayer.HEAD_FRACTION`: use `head_length=`; every arrow is solid,
  and a short one is no longer scaled down whole.
- `CameraLayer(fov=, aspect=)`: use `CameraLayer(intrinsics=, size=)`.
- `viewer.theme.colormap`: no replacement; nothing called it once the torque colours went, and an application maps
  values to colours with numpy.
- `CameraLayer` from `viewer.layers.skinned`, a second path to it: use `better_robot.viewer.CameraLayer`.
- The names `better_robot.viewer.layers` re-exported, seven of the twelve layers, a second path to them: use
  `better_robot.viewer`, which holds every layer.
- `tools.PoseGizmo(disable_sliders=, disable_axes=, depth_test=)`: use `translation=False` and `on_top=True`.
- `tools.CameraFrustum.position` and `tools.CameraFrustum.wxyz`: use `CameraFrustum.pose`.

### Fixed
- `Viewer(server=s)` takes `host`, `port` and so `url` from `s` (was the defaults, `127.0.0.1:8080`, so `record`
  opened whatever page listened there).
- `Viewer(server=s, host=, port=)` raises `ArgumentError` (the host and port were dropped silently: `url` is the
  server's); `host` and `port` default to `None`, which is `127.0.0.1` and 8080 for the viewer's own server. A
  `PoseGizmo.pose` that is not `(7,)` raises `ShapeError` when set, as it does at construction.
- `Viewer.url` puts an IPv6 host in brackets (`http://[::1]:8080`; was `http://::1:8080`), so `record` reaches such a
  server, and a page Chromium cannot load raises `ArgumentError` naming its address at once (it was blamed on the
  viser version after a minute).
- The docs and `Viewer.add` said a layer keeps the caller's arrays by reference, and the docs' example edited a
  float64 cloud that the layer had copied. A layer keeps its arrays as float32 attributes (`layer.points`,
  `layer.poses`, `layer.valid`, ...), the input itself only when it already is C-contiguous float32: edit the
  attribute in place and call `redraw()` or `Viewer.show()`.
- `SkinnedMeshLayer.bounds` skinned every vertex against every bone, 32.5 ms per step at SMPL-X size and 1.07 s per
  `Viewer.add`. It is a box per bone now, the vertices the bone carries at least a quarter of, boxed once in its bind
  frame and turned with it: 0.1 ms per step, 5 ms per add; on SMPL and SMPL-X it holds the mesh within 2 to 5 cm in
  the median.
- `GroundLayer(plane=None)` follows `Viewer.set_theme` to another up axis (it kept the first theme's plane, and
  wrote it into `plane`, which now stays as given).
- `Viewer(fps=nan)` and `Timeline(fps=nan)` raise `ArgumentError`, as `Viewer.set_fps` does; `Viewer.show(True)`
  is refused (a bool showed step 1).
- A sidebar dropdown shared by several layers drops the options of a layer that is removed and that no other
  offers: the Style dropdown kept `"meshes"` after the one robot with meshes went, and choosing it changed nothing.
  When the option shown goes, the dropdown shows the first one left and runs no callback: removing or replacing a
  layer never restyles the others, and a robot that replaces another keeps its own `style=`.
- `Theme.shade` and `RobotVisual(hue=)` raise `ShapeError` for a hue that is not three numbers, so
  `RobotVisual(hue=(1.0, 0.0))` fails where it is made, with the library's error (was the builtin error of an
  unpack, in `Viewer.add`).
- `Viewer.save` pauses playback while it saves and shows the step it showed before, as `record` does: the playback
  thread's steps landed between the saved ones. It refuses an `fps` that is not positive (`0` meant the timeline's).
- `SkinnedMeshLayer` names `weights` and `bone_poses` apart, each with the shape it needs: a `RobotVisual`'s `Skin`
  with one weight column per joint of the body instead of per model joint raised an error about bone poses the
  caller never passed.
- [Use with BetterHuman](../guides/use-with-betterhuman.md) names BetterHuman's actual API (`HumanModel`,
  `Identity.forward(q, streams=)`, the `vertices` property, `joint_translations`) and shows a body in BetterRobot's
  viewer: `RobotVisual(identity.model, q, skin=Skin(...))` with the weights scattered by `joint_ids`, beside a G1
  through a tag pose, recorded to mp4; its viewer fences run in the docs tests on a stand-in body.

## v0.3.0 (2026-11-10)

### Added
- Constraint terms: `Problem.equality(name, fn)` holds `fn = 0` and `Problem.inequality(name, fn, lower=, upper=)`
  holds `lower <= fn <= upper`, with the function, `reads` and `jacobian` of a residual term; `Term.kind` gains
  `"equality"` and `"inequality"`, and `Term.lower` and `Term.upper` carry the bounds. A constraint term is no part
  of the loss. `optim.SequentialQP(damping=, gain=0.5, tolerance=1e-4, gradient_tolerance=, step_tolerance=)` holds
  them: each step is one `solve_qp` program (the damped Gauss-Newton model under the linearized constraint terms
  and the box of the bounds), scaled back to a step radius and taken where it lowers an L1 merit (record 0055).
  `State.x` is the last iterate within `tolerance` of every bound, the iterate is `State.extra["iterate"]`, and the
  `violation` diagnostic reports the distance of `State.x` per step (record 0056). `tolerance` must be above 0;
  `State.extra["radius"]` is new and a warm key with `damping`, `growth` and `qp`. A constraint term's weight is
  a number at least 0 (0 switches the term off, a negative number raises `ArgumentError`) or a tensor whose entry 0
  switches its row off. The other solvers, `solve_constrained` and `solve_layer` raise `ArgumentError` on a problem
  with an active constraint term ([Constrained inverse kinematics](../guides/constrained-ik.md)).
- `optim.track(problem, x, *, step, warm=None, gain=0.5, damping=1e-3, iterations=20) -> (x, QPSolution)`: one
  step of differential inverse kinematics, one warm `solve_qp` program of `iterations` per call: the Gauss-Newton
  model of the cost terms with `damping` on the identity, under the rows of `SequentialQP` at `gain` and the box of
  the bounds at the same gain (a discrete barrier), each step entry within the required `step`. The step is taken:
  the program holds each row with a reserve, and its step is clamped to the box, moved back onto the rows it
  leaves and scaled back where that is not enough. The targets move between calls through `Problem.with_(data=)`;
  the solution is the warm start of the next call (record 0057; [Track moving targets](../guides/track-targets.md)).
- `optim.solve_qp(hessian, gradient, *, rows, lower, upper, iterations=20, weight=100.0, warm=None) -> QPSolution`:
  one warm-startable step towards the minimisers of a batch of quadratic programs with two-sided linear rows, by a
  fixed number of ADMM steps and one active-set step, in float64, with no host read (record 0054). Accuracy comes
  from repeated warm calls, not from `iterations`. The rows are L1-elastic: a row gives way once its multiplier
  would pass a cap, `weight` (a positive Python number) times the larger of `|g|∞` and the largest diagonal entry
  of `H`. `QPSolution` has `x`, `dual`, `primal_residual`, `dual_residual`, `penalty` and `exact` (the mask of the
  elements whose step is the minimiser: the active-set point was kept). No solver calls it yet.
- `better_robot` exports its twelve packages (`exceptions`, `lie`, `spatial`, `model`, `kinematics`, `dynamics`,
  `collision`, `io`, `optim`, `terms`, `viewer`, `testing`) and `JointFreeFlyer`, `solve_ik` and `Viewer` beside the
  core functions.
- `model.Geometry`: the shapes of a robot as a value beside `Model`, made of final records `Sphere`, `Box`,
  `Cylinder`, `Capsule` and `Mesh` attached to bodies by `AttachedShape(body, shape, collision)`; built by
  `io.build_geometry(ir, model)` or `io.load_geometry(path, model)`, extended by `ModelBuilder.add_shape`.
- `model.JointRevolute(axis)` and `model.JointPrismatic(axis)` for any unit axis; each joint class declares its
  `model.JointManifold`, and `kinematics.integrate(model, q, v)` and `kinematics.difference(model, q0, q1)` step
  and compare whole configurations through it.
- `Model.structure_key` (a hashable value of the static fields), `Model.device` and `Model.dtype`.
- `kinematics.subtree(model, joint)`, `kinematics.support(model, joint)`, `kinematics.full_configuration(model, q)`
  (mimic joints filled in) and `kinematics.euclidean_columns(model)`.
- `kinematics.point_jacobian(model, state, *, frames, points)`: the `(B..., K, 3, nv)` world-velocity Jacobian of
  points fixed to frames.
- `kinematics.frame_jacobian_time_variation` and `kinematics.joint_jacobian_time_variation`: the `J̇` of
  `a = J v̇ + J̇ v` in the world, local and local-world-aligned references.
- `KinematicState.joint_pose_of(joint)`, the world pose of one joint; `KinematicState.jacobian_columns`, the world
  Jacobian column table every joint and frame Jacobian shares; `KinematicState.world_geometry` for `dynamics`.
- `spatial` functions on tensors: `cross_motion`, `cross_force`, `transform_motion`, `transform_force`,
  `transform_inertia`, `inertia_matrix`, `rotational_inertia`, `inertia_from_mass_com` and `inertia_of_sphere`,
  `inertia_of_box`, `inertia_of_capsule`, `inertia_of_ellipsoid`, `inertia_of_mesh`.
- `io.parse_urdf` and `io.parse_mjcf` (through MuJoCo's compiler, the `mjcf` extra); `io.IRShape` and
  `IRModel.shapes`; `IRJoint.damping`, `IRJoint.armature` and `IRJoint.neutral`.
- `exceptions.ArgumentError` (also a `ValueError`) for an argument outside its documented values.
- `collision`: `RobotSpheres` and `RobotCapsules` (frame-attached spheres and segments), `self_collision_pairs`
  (with `ignore=`), `self_distances`, `self_capsule_distances` and the pair distances `sphere_sphere`,
  `sphere_capsule`, `capsule_capsule`, `sphere_halfspace`, `sphere_box`, `sphere_sdf`.
- `collision.load_spheres(path, model)`: a sphere file (JSON, spheres per frame and the frame pairs self-collision
  ignores, each with its reason) as `RobotSpheres` and its ignore list; a malformed file raises `ModelError`.
  `collision.SPHERE_SETS`, the folder of the files for the Franka Panda and the Unitree G1, fitted to their collision
  shapes by `tools/fit_spheres.py`, which ignores pairs by MoveIt's rules checked on the shapes.
- `optim`: named-tensor problems of plain-function terms. `Problem` with `residual`, `scalar` and `cached` terms
  (`TermFn`, `Term`, `RobustKernel` and the kernels `Huber`, `Cauchy`, `Tukey`, `GemanMcClure`), the solvers
  `LevenbergMarquardt`, `GaussNewton`, `LBFGS` and `TorchOptim` (an abstract `Solver` with `warm_keys`), `solve`,
  `run` with `Phase`s, `solve_layer` (differentiable), `solve_constrained`, `check_jacobian`, the `Normal`,
  `BandedNormal` and `ArrowNormal` linear systems, the `Manifold` base and `Euclidean`.
- `Problem.evaluate(x)` returns the loss, the per-term losses and the `info` a term returns beside its value;
  `Problem.gradient(x)` adds the per-term losses; `Problem.cached_values` lists the cached values.
- `Problem.linearize` assembles the normal equations through a dense, banded or arrowhead sink, the seam a new
  sparsity structure plugs into ([extension points](../contracts/08-extension-points.md), C-EXT-7): a sink's
  `add(columns_a, columns_b, values, *, shared)` is told by the variables which columns have no time axis, so no
  sink reads an index value back to the host.
- `State.solver`, `State.extra["terms"]` and `Result.diagnostics` (`grad_norm`, `step_norm`, `accepted`, LM
  `damping`); `Result.state` warm-starts `solve` and `run` through `state=`.
- `terms`: `frame_pose`, `frame_position`, `joint_limits`, `rest`, `finite_difference` (any `order`,
  `world_translation=`), `velocity_limit(model, *, dt, fraction=1.0)`, `sphere_clearance`, `self_clearance`,
  `kinematic_state`, `RobotManifold`, `ik_problem`, `random_start` and `solve_ik`.
- `viewer`: `Viewer` (viser) with `add`, `replace`, `remove_group`, `set_offset`, `set_visible`, `set_fps`,
  `aim`, `play`, `record`, `save`, `on_pointer` and `clear_pointer`; a `Timeline`; the layers `FramesLayer`,
  `PointsLayer`, `SpheresLayer`, `ArrowsLayer`, `MeshLayer`, `RigidMeshLayer`, `SkinnedMeshLayer`,
  `SkeletonLayer`, `GroundLayer`, `CameraLayer`, the panels `ImagePanel` and `PlotPanel`; `Theme`, `LIGHT`,
  `DARK`; `robot_meshes` and `robot_skeleton`.
- `Viewer.add(..., times=, valid=)`: per-layer sample times, and a per-step (or per-element) mask that hides
  samples; `Layer.set_opacity` and `Layer.opacity`; `Layer.elements`.
- `robot_skeleton(..., tau=)` shows each joint's `|tau| / effort_limit` in colour (`Theme.torque_ramp`,
  `Theme.over_limit`) and thickness; `end_points=` adds bones out to points in a frame; co-located joints are drawn
  apart along their axes.
- `ArrowsLayer(shaft_radius=, head_radius=, min_length=)` draws solid arrows and skips vectors shorter than
  `min_length`; `GroundLayer(origin=, normal=, fill=, fill_color=)` draws a tilted, filled floor.
- `better_robot.testing`: `FakeServer` (viser without sockets), `check_joint_model`, `check_robust_kernel`,
  `public_api`, `public_objects`, `imported_packages`, `dead_citations`, the docs snippet runner `snippets` and the
  `no_skips` pytest plugin; the `testing` extra installs pytest for it.
- `FakeServer.gui` declares the GUI calls of an app built on the viewer, with viser's keywords: `add_dropdown`,
  `add_text`, `add_number`, `add_markdown`, `add_tab_group` (tabs by `add_tab`) and `add_slider(..., disabled=)`;
  each control reads back the value it was given.
- `testing.glossary`: `read()` returns the terms and everyday words of BetterRobot's glossary page, which ships in
  the package, and `read(page)` those of another page; `words`, `entry` and `spelling` check a name against them.
- `tests/contract/public_api.json`, the tracked public API, kept by `tools/api_snapshot.py`.
- [Release a version](../guides/release.md): the release checklist, from the public-API diff to the downstream pins.
- `examples/viewer/05_avoid_joint_limits.py` to `09_avoid_self_collision.py`: the five behaviours of `optim.track`
  (joint limits, singular poses, floor, obstacle, self-collision) live in the viewer, a dragged target tracked one
  call per tick with a switch between the rows, a cost term and off; the floor, obstacle and self-collision demos
  also run the G1 (`--robot g1`). Each records its scripted path (`--record`); the clips are in
  [Try the avoidance demos](../guides/avoidance-demos.md).
- `terms.manipulability(model, *, frame, threshold, hinge=True)`: a hinge on Yoshikawa's manipulability of a frame's
  position Jacobian that pushes away from singular configurations below `threshold`, with an analytic Jacobian;
  `hinge=False` gives the plain `threshold - w`, the row of an inequality term.
- `terms.joint_limits(..., margin=)` starts its penalty `margin` inside the limits (a number or `(nq,)`; Euclidean
  coordinates only, at most half a joint's range);
  `terms.self_clearance(..., closest=k)` keeps the `k` sphere pairs with the least room in each configuration and
  `terms.sphere_clearance(..., closest=k)` the `k` smallest sphere-obstacle distances; both clearance terms take
  `q=`, the variable their Jacobian is for. `terms.self_clearance(margin=)` is a number or `(P,)`, one margin per
  pair, for pairs that sit closer than the common margin in the rest pose.
- `kinematics.compiled(fn, *, mode="default" | "reduce-overhead")`: an opt-in `torch.compile` of a function that
  calls the library, one graph family per grad mode and argument kind, each compiled on its second call; under
  `"reduce-overhead"` (CUDA graphs) each result is the caller's own (record
  [0053](../contracts/decisions/0053-compiled-entry-point.md)). A compiled call reads the model's memoised values
  instead of deriving them again, and small batched products compile to fused kernels: at batch 4096 on one CPU
  core (float32) the compiled FK runs 5.2 times faster than this release's eager on the Panda arm and 3.7 times on
  the G1 humanoid, inverse dynamics 4.0 and 3.3 times.
- `solve_ik(..., seeds=, generator=)` replaces `restarts=`: `seeds` starts per target run as one batch (the first is
  `q0`, the others `terms.random_start` draws with the required `generator`) and the lowest per target is kept.
  `solve_ik(..., step=)`, `Phase(step=)` and `optim.solve(..., step=)` take a callable `(problem, state) -> State`
  in place of `solver.step`, such as a step compiled once with `kinematics.compiled` and reused across solves; the
  driver hands it `state.iteration == 0` and counts itself, so it compiles once (a solver's own step, without the
  hook, sees the steps taken so far; inside `solve_ik` it is handed 0 too). `LBFGS(stop_search=False)` runs every
  line-search trial and reads nothing back to the host.

### Changed
- `Model` is one flat frozen record: the fields of `ModelStructure` and `ModelValues` live on it, it compares by
  identity, and `Model.with_values(**fields)` replaces any tensor field by keyword.
- Field names say what they hold: `Model.parent_joint_ids`, `Model.mimic_source_ids`, `Model.mimic_multipliers`,
  `Model.mimic_offsets`, `Model.lower_position_limit`, `Model.upper_position_limit`, `Model.joints`; the
  `KinematicState` accessors are plural (`joint_poses`, `frame_poses`, `joint_rotations`, ...).
- `KinematicState.frame_transforms` is a property computed from the joint transforms on first use and memoised on
  the state, so `forward_kinematics` and the dynamics never compute frames they do not read; the constructor takes
  `model`, `q` and `joint_transforms` only. It and the accessors built on it (`frame_rotations`,
  `frame_translations`, `frame_poses`, `frame_pose_of`) follow the grad mode of the read, not that of
  `forward_kinematics`: read under `torch.no_grad` or `torch.inference_mode` they carry no graph, and read with grad
  enabled from a state made under `torch.no_grad` they carry one to the model tensors that require grad.
- `build_model` orders joints parents first and keeps the file's order wherever it already is: joint ids change
  for some robots (the G1). Joint 0 is a `JointFixed` named `"universe"`.
- Every body's frame carries the body's name (`"panda_hand"`, not `"body_panda_hand"`); a frame named like a body
  or another frame is a `ModelError`.
- `Model.q_neutral` is clamped into the joint limits at build.
- `load(path, *, format=None, root_joint=None, device=None, dtype=torch.float32)` and
  `parse(path, *, format=None)` take a path only; `root_joint=JointFreeFlyer()` makes a floating base.
- `ModelBuilder.add_revolute` and `add_prismatic` take `axis=` as a unit 3-vector; `add_frame` takes `parent=`;
  `add_joint` takes `kind=` as a joint object.
- `IRJoint.joint` holds the joint object (with its axis and pitch).
- Keyword renames: `iters=` is `iterations=` (`solve`, `solve_layer`, `solve_constrained`, `Phase`, `Result`,
  `solve_ik`), `fext=` is `external_forces=` (`inverse_dynamics`, `forward_dynamics`), `jac=` is `jacobian=`
  (`TermFn`, `Problem.residual`), `gtol`, `ftol`, `xtol` are `gradient_tolerance`, `loss_tolerance`,
  `step_tolerance`, `State.it` is `State.iteration`.
- `TorchOptim(factory, *, scheduler=None)` takes a function from the parameters to a `torch.optim.Optimizer`.
- `Phase.on_start(problem, x)` returns `(x, data)`; `Problem.with_(*, data, weights, free)` replaces data entries.
- `Solver` and `optim.Manifold` are abstract base classes; `RobustKernel` is the kernels' base.
- `Model.random_configuration(*, generator)` and `terms.random_start(model, q0, *, generator)` require the
  generator.
- `forward_kinematics(..., check_quaternion_norm=True)` checks every quaternion joint.
- `kinematics.integrate` raises `DtypeMismatchError` on mixed dtypes.
- Public dataclasses and constructors are keyword-only after their leading arguments: `KinematicState`,
  `RobotSpheres`, `RobotCapsules`, `Normal`, `BandedNormal`, `ArrowNormal`, `TermFn`, `Phase`, `State`, `Result`,
  the IR records, `MeshSpec`, `Theme`, `Timeline`, and the layers (`SkeletonLayer(..., parent_ids=)`).
- `Viewer.add(layers, *, group)` takes an iterable of layers; `Viewer.play(*, block=True)` pauses the timeline when
  the serve ends.
- A second `Viewer.on_pointer` registration raises `ArgumentError` (`clear_pointer` first); the pointer callbacks
  receive viser's click and rect-select events.
- `Viewer.add` and `Viewer.replace` reject a layer added twice with `ArgumentError`, and a failing layer leaves
  the viewer as it was; `set_offset` and `remove_group` raise `UnknownNameError` for an unknown group.
- A raising `PoseGizmo` callback stops the gizmo; `wait()` raises the callback's error.
- `forward_kinematics` returns canonical-sign, normalized quaternions (`q` and `-q` are one rotation): compare
  rotations, or align signs, before comparing raw components.
- `inverse_dynamics` and `mass_matrix` include `Model.armature` (motor inertia; MJCF `armature`, zero from URDF),
  so `forward_dynamics` does too, as in Pinocchio.
- `forward_dynamics` gives NaN accelerations for a batch element whose mass matrix is not positive definite (a
  non-physical model) instead of raising torch's `_LinAlgError` for the whole batch; the other elements keep theirs.
  The factorization is not checked on the host, so a CUDA graph can capture the call. An input shared across the
  batch then gets non-finite gradients; one batched per element keeps finite gradients for the other elements.
- URDF `<dynamics>` and MJCF `damping` and `armature` reach `Model.damping` and `Model.armature`.
- `inverse_dynamics`, `forward_dynamics`, `RobotSpheres.centers` and `RobotCapsules.segments` have bitwise
  reproducible CUDA gradients (`tests/collision/test_collision.py::test_cuda_backward_is_bitwise_reproducible`).
- `LBFGS` evaluates the first line-search trial with its gradient and stops the search once every problem has a
  step; with bounds it holds the coordinates an active bound pins and tests the Armijo condition on the clipped
  step; it tries up to 20 step sizes.
- `Model.name` is the robot name of the file (`<robot name=...>`, `<mujoco model=...>`).
- The URDF and MJCF parsers raise `ModelError` naming the file and the element at fault.
- Errors are the `exceptions` classes throughout, `optim` included.
- Tensor arguments of the `lie` and `spatial` functions whose batch axes do not broadcast raise `ShapeError` naming
  both, not torch's `RuntimeError`; so do `so3.slerp`'s tensor `t` and `sphere_halfspace`'s tensor `offset`.
- `ik_problem` and `solve_ik` check `q0` as `(B, nq)` and each `targets[<frame>]` as `(..., 7)` in the model's dtype
  and device, with a batch that broadcasts with `q0`'s, and name the argument at fault.
- `solve_ik` runs a fixed `iterations=50` (it ran at most 100 and stopped once every target converged) and reads
  nothing back to the host inside the loop. One start, Panda7, one CPU core, eager, against 0.3.0 before this
  change: batch 1 takes 87 ms (66 ms before, which stopped after 23 iterations), batch 64 114 ms (237 ms), batch 1024
  395 ms (1110 ms); success 0.824 (0.829). On one GPU (RTX PRO 6000 Blackwell) batch 1 takes 229 ms eager (102 ms
  before, which stopped after 13 iterations) and 26.0 ms with its step compiled (`solve_ik(step=)`; 9.2 ms for the
  whole call under `torch.compile` before); batch 1024 takes 239 ms eager (828 ms) and 26.2 ms compiled (51.4 ms).
  G1 from one start reaches 0.580 of its targets (0.657 with 100 iterations): use `seeds=`.
- LM and Gauss-Newton keep the normal equations at `x` in `State.extra["normal"]`: a step linearizes once, at its
  candidate, instead of linearizing `x` and evaluating the candidate, with the same iterates. `frame_pose` builds its
  error from the frame's `[R | p]` and has a closed-form Jacobian (the same residual within rounding);
  `so3.from_matrix` gathers one row of `4 q qᵀ`. An LM step of Panda7 IK has 764 operators instead of 1391 and runs
  1.6 times faster on one CPU core at batch 1 (G1: 1.8 times).
- `Problem` raises `DtypeMismatchError` for a variable that is not float32 or float64 or for variables in two
  dtypes, and `DeviceMismatchError` for two devices; a term output of the wrong shape raises `ShapeError` naming the
  term; `iterations=` and `damping=` below 0 raise `ArgumentError`.
- `RobotSpheres` checks its tables at construction (`frame_ids` `(S,)` long, `offsets` `(..., S, 3)`, `radii`
  `(S,)`); `centers` and `self_distances` raise `ArgumentError` for a frame or sphere index outside its table.
- `KinematicState` checks the shapes of `q` and its transform tables at construction (`ShapeError`).
- `io.ModelBuilder`, `optim.Problem`, the solvers and `Euclidean` print a one-line repr with their settings.
- The `no_skips` plugin makes a run with a skipped, xfailed or xpassed test exit nonzero.
- Python 3.12 or newer; the `io-mjcf` extra is `mjcf`; the `viewer` extra needs `viser>=1.0.30` and `pillow`;
  the `dev` extra is a dependency group, beside `test-oracles` for the parity oracles.
- `optim.State` is a frozen dataclass: assigning to a field raises `FrozenInstanceError` (build a new state with
  `dataclasses.replace`).
- `testing.snippets` is stricter: a printed float with no fraction digits must match exactly and any other within
  half a unit of its last digit; `py`, `pycon`, `~~~python`, four-backtick, braced (`{ .python }`) and capitalised
  (`Python`) fences, a blank `data-reason`, a printing fence without an output fence and a fence that exits early
  are reported.
- The contributing pages are gone: [extending](../guides/extend.md) and [running the tests](../guides/run-the-tests.md)
  are guides, and [the test suite](tests.md) is reference.
- `benchmarks/run.py` is the counter gate: the full tier fails when a graph break, recompile, operator or host sync
  of its counter rows (forward kinematics, Jacobian, RNEA, CRBA, ABA, `solve_ik`) exceeds
  `benchmarks/thresholds/torch-<version>.json` (`tests/contract/test_compile_counters.py`).
  `benchmarks/speed.py` records wall time, never gated: kinematics, dynamics and IK functions on Panda, a 7-dof
  Panda and G1, CPU and CUDA, eager and cold-compiled, forward and backward, batch 1 to 65536, with IK iterations and
  success rates over 1024 targets, Pinocchio rows and a table, with spreads, against an earlier run or another
  library's timings ([benchmark and profile](../guides/benchmark-and-profile.md)).
- The hooks in `.pre-commit-config.yaml` run through `uv run --locked`: ruff check, ruff format --check and the fast
  tier before a commit, the full tier before a push. The release tier is one command,
  `bash tools/release_tier.sh --gpu <index>` ([running the tests](../guides/run-the-tests.md)).
- `terms.sphere_clearance` and `terms.self_clearance` have analytic Jacobians (the distance gradient at each sphere
  centre times `kinematics.point_jacobian`), 5-47x cheaper to linearize than autodiff at batch 64 to 1024 on one
  CPU core (Panda, float64), and their hinge is C1: `gap² / (2 margin)` inside the margin band, `gap - margin / 2`
  beyond it, so a soft clearance settles deeper inside the margin than with the 0.2 hinge at the same weight.
  A `distance_fn` keeps entry `s` on centre `s` and runs under `torch.func` (record
  [0051](../contracts/decisions/0051-distance-functions-for-the-analytic-jacobian.md)); any other function takes
  the autodiff route through `optim.TermFn(fn=term.fn, reads=term.reads)`.
- `Problem.linearize` raises `ShapeError` naming `window=1` when an analytic Jacobian block does not have the term's
  shape, as when a per-knot term (`frame_pose`, the clearance terms) reads a time variable without `window=`.

### Removed
- `ModelStructure`, `ModelValues`, `Model.structure`, `Model.values`: use the fields of `Model`.
- `Frame`, `Model.frames`: use `Model.frame_parent_joint_ids` and `Model.frame_placements`.
- `Body`: use `Model.body_names` and `Model.body_inertias`.
- `Joint`, `FrameType`, `JOINT_DIMENSIONS`, `Trajectory`, `StateMultibody`: no replacement; no caller used
  them (joint objects carry `nq` and `nv`; keep clips in your own type).
- `SE3`, `lie.SE3`, `lie.SO3`, `lie.Pose` and their `spatial` re-exports: use the `lie.se3` and `lie.so3` functions
  on tensors.
- `spatial.Motion`, `spatial.Force`, `spatial.Inertia`, `spatial.Symmetric3`: use `(..., 6)`, `(..., 10)` and
  `(..., 3, 3)` tensors with the `spatial` functions (`transform_motion`, `cross_force`, `inertia_of_box`, ...).
- `forward_kinematics_state`, `Model.fk`: use `forward_kinematics`.
- `frame_jacobian_time_variation`, `joint_jacobian_time_variation`, `centroidal_map`, `centroidal_momentum` at the
  root: use `kinematics.frame_jacobian_time_variation`, `kinematics.joint_jacobian_time_variation`,
  `dynamics.centroidal_map` and `dynamics.centroidal_momentum`.
- `data_model`: use `model`.
- `Model.get_subtree`, `Model.get_support`, `Model.children`, `Model.subtrees`, `Model.supports`,
  `Model.topo_order`: use `kinematics.subtree` and `kinematics.support`; joints are stored parents first.
- `Model.expand_configuration`, `Model.has_mimic`, `Model.q_expansion`, `Model.q_offset`, `Model.v_expansion`: use
  `kinematics.full_configuration`.
- `Model.integrate`, `Model.difference`, `JointModel.integrate`, `JointModel.difference`: use
  `kinematics.integrate` and `kinematics.difference`.
- `Model.joint_name_to_id`, `Model.body_name_to_id`, `Model.frame_name_to_id`: use `Model.joint_id`,
  `Model.body_id` and `Model.frame_id`.
- `Model.body_inertia`, `Model.spatial_inertias`: use `Model.body_inertias` and `spatial.inertia_matrix`.
- `Model.meta`, `IRModel.meta`: use `io.load_geometry` and `io.build_geometry`.
- `Model.rotor_inertia`, `Model.friction`, `Model.reference_configurations`, `IRJoint.friction`: no replacement;
  they were always zero or never read.
- `JointRX`, `JointRY`, `JointRZ`, `JointRevoluteUnaligned`, `JointPX`, `JointPY`, `JointPZ`,
  `JointPrismaticUnaligned`: use `JointRevolute(axis)` and `JointPrismatic(axis)`.
- `JointUniverse`: use `JointFixed`.
- `JointComposite`, `JointKind`, `JointModel.kind`, `joint_bias_acceleration`, `joint_motion_subspace_derivative`:
  no replacement; nothing called them, and a joint's kind is its class.
- `joint_placement`, `joint_motion_subspace`, `joint_velocity`, `JointModel.random_configuration`: use
  `JointModel.placement`, `JointModel.motion_subspace` and `JointModel.random`.
- Kind strings (`"revolute_rx"`, `"prismatic_px"`, `"revolute_unbounded"`, `"composite"`, `"ball"`, `"free"`, ...):
  use the joint classes (`JointRevolute`, `JointPrismatic`, ...).
- `se3.adjoint_inv` and the `tangents.*_jacobian_inv_*` functions: use `se3.adjoint_inverse` and
  `tangents.*_jacobian_inverse_*`.
- `se3.apply_base`: use `se3.compose(base[..., None, :], poses)`.
- `so3.adjoint`: use `so3.to_matrix`.
- `exceptions.BackendNotAvailableError`: use `exceptions.OptionalDependencyError`.
- `exceptions.ModelInconsistencyError`, `io.IRError`: use `exceptions.ModelError`.
- `load(free_flyer=)`, `load(preserve_joint_order=)`, `build_model(preserve_joint_order=)`, `format="builder"`, a
  callable or `yourdfpy` source: use `root_joint=JointFreeFlyer()`, one joint order, and `io.build_model` on an IR.
- `io.register_parser` and the parser registry: use `build_model(my_parser(path))`.
- `io.AssetResolver`, `FilesystemResolver`, `PackageResolver`, `CompositeResolver`, `CachedDownloadResolver`: use
  `viewer.robot_meshes(geometry, root=...)`, which reads the mesh URIs under `root`.
- `io.IRGeom`, `IRBody.visual_geoms`, `IRBody.collision_geoms`: use `io.IRShape` in `IRModel.shapes`.
- `IRJoint.kind`, `IRJoint.axis`, `IRJoint.pitch`, `IRJoint.joint_model`: use `IRJoint.joint`.
- `IRFrame.frame_type`, `ModelBuilder.add_frame(frame_type=)`: no replacement; every frame has one type.
- `ModelBuilder.add_revolute_x/y/z`, `add_prismatic_x/y/z`: use `ModelBuilder.add_revolute` and
  `ModelBuilder.add_prismatic` with `axis=`.
- `ModelBuilder.add_collision_geom`: use `ModelBuilder.add_shape`.
- `io.builders` (`build_kinematic_tree_model`, `build_kinematic_tree_body`): use a loop over `ModelBuilder`.
- `optim.CacheRef`: use the name string `Problem.cached` returns.
- `Result.__getitem__`: use `Result.x`.
- `viewer.RobotView`, `viewer.robot_layers`: use `RigidMeshLayer`, `robot_skeleton`, `FramesLayer` and
  `SpheresLayer`.
- `viewer.skeleton_spec`, `viewer.spread_offsets`: use `viewer.robot_skeleton`.
- `Layer.times`, `Layer.root`: use `Viewer.add(..., times=)`.
- The module paths `kinematics.state`, `kinematics.jacobian`, `dynamics.state`, `io.parsers`, `lie.types`,
  `lie.alignment`, `spatial.motion`, `spatial.force`, `spatial.inertia`, `spatial.mesh` and their neighbours: use
  the package (`better_robot.kinematics`, ...).
- `Data` (`oMi`, `oMf`, `liMi`, `M`, `nle`, `J`, `dJ`, `Ag`, `hg`, `com`, ...) and `update_frame_placements`: use
  `forward_kinematics` (`KinematicState.joint_poses`, `frame_poses`) and the `kinematics` and `dynamics` functions
  (`mass_matrix`, `inverse_dynamics`, `joint_jacobian`, `dynamics.centroidal_map`, `center_of_mass`).
- `compute_joint_jacobians`, `get_joint_jacobian`, `get_frame_jacobian`, `compute_joint_jacobians_time_variation`,
  `get_*_jacobian_time_variation`, `kinematics.ReferenceFrame`: use `joint_jacobian`, `frame_jacobian` and the
  `kinematics.*_jacobian_time_variation` functions with `reference="world"`, `"local"` or `"local_world_aligned"`.
- `rnea`, `aba`, `crba`, `ccrba`, `CCRBAResult`, `compute_centroidal_map`, `compute_centroidal_momentum`, `nle`,
  `bias_forces`: use `inverse_dynamics`, `forward_dynamics`, `mass_matrix`, `dynamics.centroidal_map` and
  `dynamics.centroidal_momentum`; the bias forces are `inverse_dynamics` with zero acceleration.
- `compute_minverse`, `compute_coriolis_matrix`, the dynamics derivative helpers (`compute_rnea_derivatives`,
  `compute_aba_derivatives`, `compute_crba_derivatives`, centroidal derivatives): use `torch.linalg.inv` on
  `mass_matrix` and differentiate the dynamics functions with autograd.
- `integrate_q`, `semi_implicit_euler`, `symplectic_euler`, `rk4` and `dynamics.integrators`: use
  `kinematics.integrate`; physics integration belongs to the caller's simulation.
- `better_robot.dynamics.action`, `retarget`, `RetargetCostConfig`, `better_robot.tasks`: no replacement; build
  the task as an `optim.Problem` from `terms`.
- `solve_trajopt` and its parameterizations (`KnotTrajectory`, `BSplineTrajectory`,
  `TrajectoryParameterization`): use an `optim.Problem` with a time variable (`Problem(..., time=("q",))`) and
  `optim.solve`.
- `CostStack`, `CostItem`, `CostKind`, `LeastSquaresProblem`, `ResidualItem`, `ObjectiveItem`, `ObjectiveTerm`,
  `register_residual`, `get_residual`, `registered_residuals`, `better_robot.costs`: use `optim.Problem` with
  `Problem.residual` and `Problem.scalar` terms.
- The residual classes (`Residual`, `SmoothnessResidual`, `AccelerationResidual`, `JerkResidual`,
  `RestResidual`, `NullspaceResidual`, `YoshikawaResidual`, `SelfCollisionResidual`, `WorldCollisionResidual`,
  `JointAccelLimit`, `ContactConsistencyResidual`, `PointProjectionResidual`, the scene-SDF and masked-Chamfer
  residuals, `ScalarCost`): use plain functions as `optim` terms and the `terms` factories (`rest`,
  `finite_difference`, `self_clearance`, `velocity_limit`, ...).
- Residual weights and gates (`weight` objects `ScaleWeight`, `DiagonalWeight`, `Weight`, `row_weight`,
  `residual.enabled`, reductions, activity masks): use the term `weight=` and `Problem.with_(weights=...)`.
- `ResidualState`, `residual_jacobian`, `ResidualSpec`, `jacobian_spec`, `JacobianStrategy`, provider `inputs` and
  `reads` declarations: use `TermFn(fn, reads, *, jacobian=None)`; without `jacobian` the term is differentiated
  by autograd.
- `Variable`, `SO3Variable`, `SE3Variable`, `RobotVariable`, `VarSpec`, `Values`, `detach_values`, `Bounds`,
  `SO3Manifold`, `SE3Manifold`, `RobotConfig`, `Node`, `RobotState`, `EvaluationContext`, `Provider`,
  `RobotStateProvider`: use named tensors in `optim.Problem` with `manifolds=` (`terms.RobotManifold`) and
  `bounds=`, and `Problem.cached` for shared work.
- `Optimizer`, `OptimizationResult`, `OptimizerInfo`, `OptimizerConfig`, `SolverState`, `SolverStatus`, `LMState`,
  `LMStatus`, `FirstOrderResult`, `run_first_order`, `TorchOptimizer`, `Adam`, `AdamState`, `AdamStatus`,
  `LMThenLBFGS`, `MultiStageOptimizer`, `OptimizerStage`, `PhaseResult`, `run_phases`, `resume`: use
  `optim.solve` and `optim.run` with `Phase`s, which return a `Result`.
- `DampingStrategy`, `Constant`, `Adaptive`, `LevenbergMarquardt(block_step_limits=)`: use
  `LevenbergMarquardt(damping=...)`.
- `Cholesky`, `BandedCholesky`, `LSTSQ`, `LU`, `NormalCG`, `NormalOperator`, `BlockBandedMatrix`,
  `TemporalAnalysis`, `LinearizationReason`, `LinearSystemKind`, `ImplicitDiffConfig`, `linearization=`: use
  `Normal`, `BandedNormal` and `ArrowNormal`, chosen by `Problem(time=..., dense=...)`, and `optim.solve_layer`
  for implicit gradients.
- The import paths `optim.blocks`, `optim.kernels.*`, `optim.solvers.*`, `optim.structure`, `optim.providers`,
  `optim.autograd`, `optim.manifolds`, `optim.optimizers`, `optim.strategies`: use `better_robot.optim`.
- `IR_SCHEMA_VERSION`, `IRModel.schema_version`, `IRSchemaVersionError`: no replacement; the IR is an in-process
  value, so parse the source file again after an upgrade.
- `kinematics.chain`, `data_model.indexing`, `better_robot.utils`, `ExecutionBatch`, `flatten_execution_batch`,
  `ModelValues.execution_batch_shape`: no replacement; they had no caller.
- The Warp lanes (`use_warp`, the `warp` extra, the kernel modules, `kinematics._warp_bridge`) and
  `forward_kinematics(use_compile=)`: use `torch.compile` on the library functions; torch is the one
  implementation.
- `Visualizer`, `SkeletonMode`, `URDFMeshMode`, `TrajectoryPlayer`: use `Viewer` with `robot_skeleton`,
  `RigidMeshLayer` and its `Timeline`.
- The development dependencies `hypothesis`, `pyperf`, `mypy`, `jaxtyping`, `pytest-cov`, `pytest-xdist`,
  `matplotlib` and the GitHub Actions workflow: no replacement; nothing used them, and the test tiers run by local
  command and hooks.
- The `bench` pytest marker, the `tests/bench/` micro-benchmarks and `pytest-benchmark`, which ran them: use
  `benchmarks/run.py`, whose counter rows the full tier gates.

### Fixed
- The MJCF parser places capsule and cylinder `fromto` geometry as MuJoCo's compiler does.
- URDF inertias honour `<inertial><origin rpy>`.
- Zero-size batches run through forward kinematics, the dynamics functions and every solver.
- `Viewer.url` reports the bound port, not the requested one.
- `so3.to_euler` is exact at pitch ±π/2.
- `robot_skeleton` draws the markers of axis-aligned revolute and prismatic joints along their axes.
- `LBFGS` with bounds reaches the optimum of bound-heavy problems (a nonnegative force fit stopped 0.4-4 % above
  it, now within 0.01 %), and a problem whose plain gradient step finds no decrease is done.
- `robot_skeleton` draws a joint kind defined outside the library as a sphere instead of raising `KeyError`.
- `build_model` raises `ModelError` naming the joint or frame whose `origin` or `placement` is not a `(7,)` pose
  with a unit quaternion; it built such a model silently, or failed an internal check as a library bug.
- The first `forward_kinematics`, `integrate` or `spatial` call of a process no longer imports sympy
  (through `torch.broadcast_shapes`): 0.2 s with a warm disk cache, over a second cold.
- `RobustKernel.weight` documents the factor the solvers assume, twice the derivative of `rho` (it said the
  derivative: a kernel written to it stopped at the wrong minimum beside a plain term).
- `Model.random_configuration` and every joint kind's `random` draw on the limits' device with a generator that lives
  there, so a CUDA or meta model samples instead of raising torch's device-mismatch error.

## v0.2.0 (2026-04-11)

The first stable release of the PyTorch-native BetterRobot stack. Its named stubs (`compute_minverse`,
`compute_coriolis_matrix`, the higher-order integrators, analytic derivatives, `JerkResidual`, `YoshikawaResidual`,
`NullspaceResidual`, `SelfCollisionResidual`, `WorldCollisionResidual`, `JointAccelLimit`, `retarget`, the Warp
kernels) raised on call.

### Added
- **Pinocchio-style `Model` / `Data` architecture.** Frozen `Model`,
  mutable `Data`, polymorphic `.to()`.
- **Universal joint system.** Revolute (R{X,Y,Z}, unaligned, unbounded),
  prismatic, spherical, free-flyer, fixed, helical, planar, mimic,
  composite. Single code path for fixed and floating base — a
  floating-base robot is one whose root joint is `JointFreeFlyer`.
- **Batched-by-default FK.** `forward_kinematics(model, q: (B..., nq))`
  runs over any batch shape with no Python loops.
- **Analytic Jacobians.** All built-in residuals have a `.jacobian()`
  method; `JacobianStrategy.AUTO` prefers analytic, falls back to
  central finite differences.
- **Unified solver stack.** `LeastSquaresProblem` + `CostStack` + `Optimizer`
  serves IK and trajectory optimisation through the same substrate.
  Pluggable optimisers (LM / GN / Adam / L-BFGS / multi-stage), linear
  selectable solvers (Cholesky / LSTSQ), robust kernels
  (L2 / Huber / Cauchy / Tukey), and damping strategies (Constant /
  Adaptive).
- **Featherstone dynamics.** RNEA, ABA, CRBA, CCRBA, centroidal momentum,
  centre of mass, and autograd-derived RNEA, ABA, and CRBA derivative helpers. Three-layer
  Crocoddyl-style action models for future optimal-control work.
- **Trajectory optimisation.** `solve_trajopt` with knot parameterisation;
  manifold-aware `Trajectory.resample`.
- **URDF + MJCF parsers.** `br.load(path)` dispatches by suffix;
  `free_flyer=True` adds a free-flyer root. Programmatic `ModelBuilder`
  for robots not described by a file.
- **Asset resolution.** `AssetResolver` Protocol with `Filesystem`,
  `Package`, `Composite`, and `CachedDownload` resolvers; mesh path
  logic lives in one place.
- **Viewer.** `Visualizer` with viser backend, `SkeletonMode`,
  `URDFMeshMode`, draggable IK target gizmos, frame-axes / grid /
  force-vector overlays, minimal `TrajectoryPlayer`.
- **Public API.** 26 symbols at `better_robot.__init__`: `Model`, `Data`, `Frame`, `Joint`, `Body`, `load`,
  `ModelBuilder`, `SE3`, `forward_kinematics`, `update_frame_placements`, `compute_joint_jacobians`,
  `get_joint_jacobian`, `get_frame_jacobian`, `JacobianStrategy`, `rnea`, `aba`, `crba`, `center_of_mass`,
  `compute_centroidal_map`, `register_residual`, `CostStack`, `LeastSquaresProblem`, `solve_ik`, `solve_trajopt`,
  `retarget`, `Trajectory`.
