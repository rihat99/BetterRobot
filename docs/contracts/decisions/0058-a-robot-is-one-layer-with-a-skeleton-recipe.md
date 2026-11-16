# 0058 A robot is one layer, `RobotVisual`, and its skeleton is laid out by a `SkeletonRecipe`

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-17

## Context
Showing a robot took three calls that each ran or needed forward kinematics (`robot_skeleton`, `robot_meshes`, a
`FramesLayer` or `SpheresLayer` built by hand), and switching between a mesh view and a skeleton meant removing and
adding layers. The skeleton was a ball per joint and a line per parent, the same for every joint kind, so joints
that share a point (a hip's three axes) were one blob. A Unitree G1 as meshes, skeleton, spheres and frames was 82
scene nodes and 145 messages per step. Records [0004](0004-viser-viewer.md) and [0022](0022-viewer-boundary.md) fix
the viewer's backend and its boundary; neither says how a robot is drawn.

## Decision
1. **One layer per robot.** `viewer.RobotVisual(model, q)` runs forward kinematics once over `q`, `(nq,)` or a clip
   `(T, nq)`, and owns every way the robot is drawn: a style (`"meshes"`, `"minimal"`, `"maximal"`), a collision
   overlay (the sphere set, else the collision shapes of the geometry) and frame triads (every moving joint, or
   frames by id or name). `update(q)` shows new configurations of the same number of steps.
2. **Parts are built once and then hidden.** A style, the overlay and the triads are each built the first time
   they are shown; a switch only shows and hides them, and a step is written only to the parts on screen.
   `set_style`, `set_collision` and `set_frames` switch from code, and the sidebar's Robot section holds one Style
   dropdown, one Collision box and one Frames box for every robot in the viewer.
3. **The skeleton is a recipe.** `viewer.SkeletonRecipe` (frozen, keyword-only) holds the choices: `glyphs`,
   `bones`, `hulls`, `limits`, `cluster`, `twists`, `sides`, `size`, and the per-name overrides `joint_size`,
   `body_hull`, `color`. `MAXIMAL` draws a glyph per joint kind (a drum for a revolute joint, a sleeve on a rail
   for a prismatic one, a ball for a spherical one, a cube for a free base), bones, body hulls and end blocks;
   `MINIMAL` draws one ball per cluster of close joints and sticks. `RobotVisual(minimal=, maximal=)` takes others.
4. **Sizes come from the robot, positions stay true.** Every radius is a fraction of the robot's size. Glyphs that
   would overlap slide along their own axis, so each stays on its true axis line, and shrink only when sliding is
   not enough. Hull sizes come from the sphere set, else the collision shapes, else the inertias.
5. **The layout is rigid.** It is computed once at `model.q_neutral`; every block is rigid in one joint frame, so
   a skeleton is at most three batched nodes (cylinders, balls, boxes) and one line node for limit arcs, whatever
   the robot.
6. **Colour is the theme's.** A robot takes the theme's next hue (or `hue=`), and the skeleton's parts take
   `theme.shade(hue, role)`; the overlay takes `theme.collision` and `theme.collision_opacity`.
7. **A skin is a mesh style.** `viewer.Skin(vertices, faces, *, weights, bind=None)` is shown as `"meshes"`, skinned
   in the browser.
8. **The old adapters go.** `viewer.robot_skeleton`, `viewer.robot_meshes` and `viewer.SkeletonLayer` are removed,
   with the theme fields `joint`, `link`, `torque_ramp` and `over_limit`.

## Example
```python { .no-run }
from better_robot.viewer import RobotVisual, SkeletonRecipe, Viewer

viewer = Viewer()
robot = viewer.add(RobotVisual(model, q, geometry=geometry, root=URDF, spheres=spheres, frames=["panda_hand"]))
robot.set_style("maximal")       # built now; "meshes" stays in the scene, hidden
robot.set_collision(True)        # the sphere set, faint, over any style
robot.update(q_next)             # one kinematics pass, then a redraw
viewer.add(RobotVisual(model, q, name="ghost", maximal=SkeletonRecipe(limits=True)), offset=(0.0, 1.0, 0.0))
```

## Consequences
Measured on the Unitree G1 of `robot_descriptions` (29 revolute joints on a free base), 300 steps on a real viser
server with no browser, at host load average 40 to 66 (the times are upper bounds):

| G1 | drawing nodes | messages per step | kB per step | ms per step |
|---|---|---|---|---|
| before: meshes, skeleton, spheres, frames | 82 | 145 | | 4.89 |
| meshes | 36 | 66 | 7.0 | 1.86 |
| meshes, overlay, triads | 38 | 69 | 9.4 | 1.95 |
| minimal | 3 | 5 | 1.3 | 0.82 |
| minimal, overlay, triads | 5 | 8 | 3.7 | 0.88 |
| maximal | 3 | 7 | 3.7 | 0.60 |
| maximal, overlay, triads | 5 | 10 | 6.1 | 0.67 |

- Each part adds one grouping frame node; the message counts include one timeline message. A second switch to a
  style creates no node. `update` of a `(300, nq)` clip takes 14.5 ms, a live `update` of one configuration with
  kinematics and redraw 1.3 ms.
- A step writes positions and rotations to each node; the stick node of a minimal skeleton also writes its
  lengths, because a stick between two moving balls stretches.
- The G1's maximal layout: 16 glyphs slide, 661.5 mm in total, and two shrink (the hip pitch drums, to 0.85); it is
  94 cylinders (29 drums, 29 faces, 36 bones), one ball (the head) and 7 boxes. Its minimal layout is 13 clusters:
  12 balls, 14 sticks and the base cube. The Panda slides 5 glyphs and shrinks none. Building the G1's layout takes
  about 40 ms.
- `bounds` is the box of the joints and of the maximal skeleton's blocks in every style, so the camera does not
  jump on a style switch.
- A switch from code does not move the sidebar's control: the control is shared by every robot, and moving it
  would switch them all.
- Torque colouring (`robot_skeleton(tau=)`) and `end_points=` have no replacement.

## Alternatives
- Keep three adapter functions and add a switcher on top: three kinematics passes and three owners of one robot's
  nodes; a switch would still remove and add layers.
- Rebuild the nodes on every style switch: fewer hidden nodes, but a mesh style reloads its files and the browser
  flashes.
- One node per joint glyph: the G1 would be over 100 nodes again; batched nodes keep a step at 5 to 10 messages.
- Lay the skeleton out per step: joints that share a point would be solved every step, and the blocks would swim
  as the robot moves.
- Scale glyphs down until they do not overlap, without sliding: the hip's three drums become too small to read.
- Options as keyword arguments of `RobotVisual` instead of a recipe value: eleven more parameters, twice (one set
  per skeleton style).
- A control per robot in the sidebar: the Robot section grows with the number of robots; comparison scenes have
  two to four.

## Compatibility
Breaks callers of `robot_skeleton`, `robot_meshes`, `SkeletonLayer` and of four `Theme` fields, inside the
unreleased 0.3.0. The changelog and `docs/reference/migration-0.3.0.md` name the replacement of each.

## Enforcement
`tests/viewer/test_robot.py` (styles, the overlay, triads by id or name, a skin, `bounds`, remove and add again, a
theme switch, a name clash, a joint kind from outside the library, one Style control for two robots, a second
switch creates no node), `tests/viewer/test_skeleton.py` (the layout: no overlap inside a rigid group, mirrored slides, bone
thickness, the G1's 13 clusters, end block sizes, a clip equals its steps, the shades), `tests/viewer/test_fake_server.py` (the sidebar handles against viser), `tests/contract/test_api_snapshot.py`
(the five names), `tests/contract/test_keyword_only.py` and `tests/contract/test_identifiers.py` (the signatures),
`tests/contract/test_name_words.py` (rule C-NAME-4 lets `RobotVisual` say frames).

## Open questions
- Should a switch from code move the shared sidebar control? It would need a way to set a control without running
  its callbacks, or one control per robot.
- A triad of a named frame is sized like a joint's (two glyph radii long), so one inside a hull or a mesh shows
  only the axes that leave it. A size of its own for named frames would be one more constant.
- `theme.colormap` has no caller in `src` now that torque colouring is gone: remove it, or keep it for
  applications.
