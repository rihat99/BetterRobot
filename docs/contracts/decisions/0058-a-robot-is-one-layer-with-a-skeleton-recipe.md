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
   `(T, nq)`, and owns every way the robot is drawn: a style (`"meshes"`, `"minimal"`, `"medium"`, `"maximal"`), a
   collision overlay (the sphere set, else the collision shapes of the geometry) and frame triads (every moving
   joint, or frames by id or name). `update(q)` shows new configurations of the same number of steps.
2. **Parts are built once and then hidden.** A style, the overlay and the triads are each built the first time
   they are shown; a switch only shows and hides them (another set of named frames is another triad node), and a
   step is written only to the parts on screen. `set_style`, `set_collision` and `set_frames` switch from code, and
   the sidebar's Robot section holds one Style dropdown, one Collision box and one Frames box for every robot in the
   viewer: a choice there switches every robot, and a switch from code moves the control through
   `Sidebar.set_value`, which runs no callback, so the other robots stay as they are.
3. **The skeleton is a recipe.** `viewer.SkeletonRecipe` (frozen, keyword-only) holds the choices: `glyphs`,
   `bones`, `hulls`, `limits`, `cluster`, `twists`, `sides`, `size`, and the per-name overrides `joint_size`,
   `body_hull`, `color`. `MAXIMAL` draws a glyph per joint kind (a drum for a revolute joint, a sleeve riding on
   a rail that spans the travel for a prismatic one, a ball for a spherical one, a cube for a free base), slim
   tapered limbs, trunks for the root body and the bodies that branch, and the ends of the chains (a sole, a finger
   block, a limb that tapers to the end of its body, with no ball at its tip), in two tones: joints in the hue,
   bodies pale (`Theme.shade` role `bone`). No ball sits among the drums but a spherical joint's, where it would
   read as a ball joint, and a head is a ball only where it stands above a trunk and every joint (the end of a
   neck, a round second frame of the trunk). `MINIMAL` draws one ball per cluster of close joints (a cube for a
   free base) and sticks between them, a stick and a small ball on to the end of each body that ends a chain, and
   a ball for a trunk's head. `MEDIUM` puts the maximal glyphs on the minimal frame: `hulls="none"` with
   `glyphs="kind"` draws no body, and thin rods in the pale body shade join the glyphs, with a rod on to the end of
   each chain closed by a round cap of its width and the maximal heads as small pale balls (`bones=0.35`,
   `size=0.7`), so the collision overlay can show the volume around the motors.
   `RobotVisual(minimal=, medium=, maximal=)` takes others.
4. **Sizes come from the robot, positions stay true.** A glyph is as large as the motor that would carry its
   subtree: 26 mm times the cube root of the subtree's mass in kg, kept between 0.5 and 1.2 reference radii (a
   third of the mean bone, kept between 2 % and 3 % of the robot's size), so a wrist is smaller than its elbow and
   a chain narrows towards its tip; a robot without mass takes the reference. The drums of joints that nearly share
   a point (a compound: a hip, a waist, a knuckle) cross there, as a universal joint's do: each stays on its own
   axis at 0.85 of the drum before it, never under 0.7 of its own size, and one whose joint sits inside another
   slides along its axis to centre on it. Any other glyph that would overlap slides along its own axis, at most
   half its length, so each stays on its true axis line, and shrinks only when sliding is not enough; no glyph is
   wider than the one before it (unless `joint_size` makes it so), and a glyph that ends a chain shrinks at its
   place before it slides. On rods a drum outside a compound is at most 0.35 of its shortest rod, so a wrist reads
   motor, rod, motor. A maximal limb is `bones` glyph radii wide, or 0.7 of its link's thinner half width when that
   is more, up to 0.8 of the radius its mass gives (a thigh is wider than its shin), and starts or ends on the side
   of a narrower drum it joins. Body sizes come from the collision shapes, else the sphere set, else the inertias;
   a collision mesh file that cannot be read is left out with a warning, since only the collision overlay draws it
   (and raises for it).
5. **The layout is rigid.** It is computed once at `model.q_neutral`; every block is rigid in one joint frame,
   except the minimal skeleton's cluster balls, each at the mean of its joints, and the sticks of the minimal and
   medium skeletons, whose two ends move apart and together with the joints; a skeleton is at most three batched
   nodes (cylinders, balls, boxes) and one line node for limit arcs, whatever the robot.
6. **Colour is the theme's.** A robot takes the first of the theme's hues that no other robot of its viewer is
   drawn in (or `hue=`; `RobotVisual.hue`), and the skeleton's parts take `theme.shade(hue, role, side=)`; the
   overlay takes `theme.collision` and `theme.collision_opacity`, a pale neutral glass in both themes. Like every
   mesh layer below opacity 1 it casts and catches no shadow (`Layer._surface`, a subclass hook beside `_track`,
   `_store_opacity` and `_fade`): the shadow map ignores opacity, so a glass would throw a solid shadow and wear
   the shadows of what it holds. `opacity=` (`set_opacity`) draws every style of a robot see-through by the same
   rule, a ghost, while the overlay and the triads keep their look. `joint_values=` (`set_joint_values`), one value
   per model joint and step, `(T, njoints)` or `(njoints,)`, colours the joint glyphs of the skeleton styles
   through the two colours of `theme.ramp` over `value_range`, clamped, with one more write of colours per node
   that holds a glyph and no new node.
7. **A skin is a mesh style.** `viewer.Skin(vertices, faces, *, weights, bind=None)` is shown as `"meshes"`, skinned
   in the browser.
8. **The old adapters go.** `viewer.robot_skeleton`, `viewer.robot_meshes` and `viewer.SkeletonLayer` are removed,
   with the theme fields `joint`, `link`, `torque_ramp` (now `ramp`, item 6) and `over_limit`, and
   `theme.colormap`: joint values blend the two colours of the ramp inside the robot visual.

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
Measured on the Unitree G1 of `robot_descriptions` (29 revolute joints on a free base) with its sphere set, 300
steps on a real viser server with no browser; the rows before and with meshes at host load average 40 to 66, the
skeleton rows after the layout rules of item 4 at about 40 (the times are upper bounds):

| G1 | drawing nodes | messages per step | kB per step | ms per step |
|---|---|---|---|---|
| before: meshes, skeleton, spheres, frames | 82 | 145 | | 4.89 |
| meshes | 36 | 66 | 7.0 | 1.86 |
| meshes, overlay, triads | 38 | 69 | 9.4 | 1.95 |
| minimal | 3 | 6 | 2.1 | 0.69 |
| minimal, overlay, triads | 5 | 9 | 4.6 | 0.81 |
| medium | 3 | 6 | 4.7 | 0.73 |
| medium, overlay, triads | 5 | 9 | 7.1 | 0.79 |
| maximal | 3 | 7 | 3.6 | 0.41 |
| maximal, overlay, triads | 5 | 10 | 6.1 | 0.46 |

- Each part adds one grouping frame node; the message counts include one timeline message. A second switch to a
  style creates no node. `update` of a `(300, nq)` clip takes 8.3 ms, a live `update` of one configuration with
  kinematics and redraw 0.77 ms, and a switch to a style already built 2.6 ms.
- A step writes positions and rotations to each skeleton and triad node (positions only to the overlay's spheres,
  points to the limit arcs); the stick node of a minimal skeleton also writes its lengths, because a stick between
  two moving balls stretches, and so does the rod node of a medium one.
- The G1's maximal layout (with its sphere set): 6 glyphs slide, 160.5 mm in total, and none shrinks; it is 58
  cylinders (29 drums, 29 faces), 1 ball (the head) and 37 boxes (30 limbs, 4 ends, 2 trunks and the base). Its
  minimal layout is 18 balls and 22 sticks beside the base cube. The Panda's maximal layout slides 4 glyphs,
  94.9 mm, and shrinks one, the first finger's sleeve to 0.5 at its place. Building the G1's layout takes 8 ms
  (minimal), 22 ms (medium) and 24 ms (maximal).
- The G1's medium layout (with its sphere set): 6 glyphs slide, 106.3 mm in total, and none shrinks; it is 95
  cylinders (29 drums, 29 faces, 37 rods), 5 balls (the caps of the four ends and the head) and the base cube,
  three nodes like the others, and a step writes what the maximal one writes plus the rods' lengths.
- `bounds` is the same in every style, so the camera does not jump on a style switch: a box per joint, per
  collision sphere and per collision shape that is not a mesh, per block of the maximal skeleton on a body
  without one of those, and per bone of a skin that carries vertices.
- A switch from code moves the shared sidebar control and switches no other robot, so the control shows the robot
  switched last, and the other robots may show another style, overlay or triads than it does.
- Torque colouring (`robot_skeleton(tau=)`) is `joint_values=`, each joint's `|tau|` over its effort limit: on the
  G1 one more message, 0.2 to 0.4 kB and 0.12 to 0.15 ms per step and no node (minimal 0.83, medium 0.88, maximal
  0.53 ms per step with values, at load average 16); a ghost at opacity 0.35 costs nothing per step. `end_points=`
  has no replacement.

## Alternatives
- Keep three adapter functions and add a switcher on top: three kinematics passes and three owners of one robot's
  nodes; a switch would still remove and add layers.
- Rebuild the nodes on every style switch: fewer hidden nodes, but a mesh style reloads its files and the browser
  flashes.
- One node per joint glyph: the G1 would be over 100 nodes again; batched nodes keep a step at 5 to 10 messages.
- Lay the skeleton out per step: joints that share a point would be solved every step, and the blocks would swim
  as the robot moves.
- Scale glyphs down until they do not overlap, without sliding: the hip's three drums become too small to read.
- A compound as one ball: it reads as a ball joint, which a G1's hips and wrists are not.
- A glyph sized by its joint's effort limit (0.8 to 1.2 of the reference): the G1's wrist roll was as wide as its
  elbow, and every drum of a Talos or a UR5e alike.
- Options as keyword arguments of `RobotVisual` instead of a recipe value: eleven more parameters, twice (one set
  per skeleton style).
- A control per robot in the sidebar: the Robot section grows with the number of robots; comparison scenes have
  two to four. For the same reason the section has no opacity slider: one shared slider would fade a ghost and its
  solid twin alike.

## Compatibility
Breaks callers of `robot_skeleton`, `robot_meshes`, `SkeletonLayer` and of four `Theme` fields, inside the
unreleased 0.3.0. The changelog and `docs/reference/migration-0.3.0.md` name the replacement of each.

## Enforcement
`tests/viewer/test_robot.py` (styles, the overlay, triads by id or name, a skin, `bounds`, remove and add again, a
theme switch, a name clash, a joint kind from outside the library, one Style control for two robots, a second switch
creates no node, the rotations of a step against forward kinematics and its exact writes, a collision mesh that does
not load), `tests/viewer/test_skeleton.py` (the layout: no overlap inside a rigid group but the crossing drums of a
compound, mirrored slides, glyphs that narrow towards the tips, a drum on its axis within its length of its joint,
a hinge a drum and a slider a sleeve, never a ball, a neck's head above its trunk, the medium glyphs of every kind
on rods without bodies, none sized wider than its maximal one before the slide rule shrinks either, rods between
the G1's wrist drums, its chest above its waist drums, maximal limbs from their links, the Panda's rails over the
fingers' travel, bone thickness, the G1's clusters alike on both sides, end block sizes, the size rule, a clip
equals its steps, the shades, `hulls` and `body_hull`, limit arcs over the joint's range, a recipe value that is
not finite),
`tests/viewer/test_core.py` (a see-through mesh casts and catches no shadow), `tests/viewer/test_fake_server.py`
(the sidebar handles against viser), `tests/viewer/test_server_smoke.py` (on a
real server, a switch from code moves the shared control and no other robot), `tests/contract/test_api_snapshot.py`
(the six names), `tests/contract/test_keyword_only.py` and `tests/contract/test_identifiers.py` (the signatures),
`tests/contract/test_name_words.py` (rule C-NAME-4 lets `RobotVisual` say frames).

## Open questions
- A triad of a named frame is sized like a joint's (two glyph radii long), so one inside a hull or a mesh shows
  only the axes that leave it. A size of its own for named frames would be one more constant.
