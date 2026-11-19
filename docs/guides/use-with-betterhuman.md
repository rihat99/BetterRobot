# Use with BetterHuman

BetterHuman is a separate package that builds SMPL-family human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR and
SOMA) as ordinary BetterRobot models. This page lists what it adds on top of BetterRobot, so you can tell which
package owns which step, and shows a body beside a robot in BetterRobot's viewer, drawn from plain arrays.

## BetterHuman adds bodies, not a second library

| Piece | What it is |
|-------|------------|
| `bh.load(family, path=None)` | a `HumanModel`: the family's BetterRobot `Model` at zero identity, the skin weights `skin_matrix (V, J)`, the BetterRobot joint of each of the family's `J` joints `joint_ids (J,)`, and `spec.faces (F, 3)` |
| `HumanModel.bake(betas=...)` | an `Identity`: the shape baked once into `Identity.model` (joint placements, part masses and inertias, contact frames) and the surface at rest, `rest_vertices (V, 3)` |
| `Identity.forward(q, streams=None)` | a `HumanState`: `kinematics` (the `KinematicState`), the posed surface with its pose correctives `vertices (..., V, 3)`, `joint_translations (..., J, 3)`; per-frame groups such as the SMPL-X `expression` go in `streams` |
| `better_human.terms` | cost terms for [`better_robot.optim`](../concepts/optimization.md): `joints`, `points`, `landmarks`, `reprojection`, `ground`, `pose_prior`, `shape_prior` and `root_wrench`, which read a `human_state(identity)` cached once per problem |

Everything below the human layer is BetterRobot: the configuration `q (B, nq)` is a free-flyer root followed by ball
joints, so `kinematics.integrate`, `forward_kinematics`, the dynamics in `better_robot.dynamics` and the solvers in
`better_robot.optim` apply unchanged. Because the shape lives in the model's tensor fields, gradients reach the shape
parameters the same way they reach any model tensor (see [Differentiate](differentiate.md)).

## A body is a BetterRobot model with a baked identity

This is the quick start of BetterHuman's `README.md`. It does not run in this repository's documentation tests:
BetterRobot does not install BetterHuman, and the SMPL-family weights are licensed, shipped with neither package.
BetterHuman's README says where to get them and where to put them (`$BETTERHUMAN_MODELS_DIR`).

```python { .no-run data-reason="needs BetterHuman and licensed SMPL weights, which this repository does not install" }
import torch
import better_robot as br
import better_human as bh

human = bh.load("smpl")                            # reads $BETTERHUMAN_MODELS_DIR/smpl/SMPL_NEUTRAL.npz
identity = human.bake(betas=torch.zeros(10))       # the shape, baked once into a BetterRobot Model
q = identity.model.q_neutral.expand(4, -1).clone()  # (4, nq): a free-flyer root and 23 ball joints
q[:, 1] = torch.linspace(0.0, 1.5, 4)              # lift the root by 0 to 1.5 m (y up)
state = identity.forward(q)                        # one batched forward-kinematics query
print(tuple(state.vertices.shape), tuple(state.joint_translations.shape))  # (4, 6890, 3) (4, 24, 3)
print(br.center_of_mass(identity.model, state.kinematics)[:, 1])          # dynamics on the baked masses
```

`bake` fixes the shape once and `forward` poses the baked body for a batch of configurations; a clip is a batch
`q (T, nq)`. Per-frame groups such as the SMPL-X expression go to `forward`:
`identity.forward(q, streams={"expression": torch.zeros(T, 10)})`.

## Show a body beside a robot

The viewer needs nothing from BetterHuman: a body is a `RobotVisual` of its model with a `Skin`, the surface at
rest with one weight per model joint. `RobotVisual(model, q, skin=skin)` draws the surface skinned in the browser as
its `"meshes"` style, and the same clip as the `"minimal"`, `"medium"` or `"maximal"` skeleton. A family's weights
name its own joints, so they are scattered into the model's joint columns by `joint_ids`; `rest_vertices` are the
surface at the model's neutral configuration, the `Skin`'s default `bind`.

```python { .no-run data-reason="needs BetterHuman and licensed SMPL weights, which this repository does not install" }
import numpy as np
from better_robot.viewer import Skin

model = identity.model
q = model.q_neutral.repeat(60, 1)                  # (T, nq): your clip, in the body's y-up frame
weights = np.zeros((len(identity.rest_vertices), model.njoints), np.float32)
weights[:, human.joint_ids.numpy()] = human.skin_matrix.numpy()  # the family's joints -> the model's joints
skin = Skin(identity.rest_vertices.numpy(), human.spec.faces.numpy(), weights=weights)
```

The rest of this page runs without BetterHuman, on a stand-in with the same layout: a free root and ball joints, y
up, and a surface whose every vertex follows one model joint.

```python
import numpy as np
import torch
import trimesh
import better_robot as br
from better_robot.lie import so3
from better_robot.viewer import Skin

builder = br.ModelBuilder("body")                  # a pelvis, a chest and a head on ball joints
for name in ("pelvis", "chest", "head"):
    builder.add_body(name, mass=10.0)
builder.add_free_flyer_root(child="pelvis")
builder.add_spherical("spine", parent="pelvis", child="chest", origin=torch.tensor([0, 0.1, 0, 0, 0, 0, 1.0]))
builder.add_spherical("neck", parent="chest", child="head", origin=torch.tensor([0, 0.45, 0, 0, 0, 0, 1.0]))
model = br.io.build_model(builder.finalize())

T = 60
q = model.q_neutral.repeat(T, 1)                   # (T, nq): the root, then the spine and the neck
q[:, 1] = 0.95                                     # the pelvis 0.95 m up (y up)
q[:, 7:11] = so3.from_axis_angle(torch.tensor([1.0, 0.0, 0.0]).expand(T, 3), 0.4 * torch.sin(torch.arange(T) / 10))
capsule = trimesh.creation.capsule(height=1.5, radius=0.14)
rest = capsule.vertices[:, [1, 2, 0]] - [0.0, 0.08, 0.0]  # stood along y: from the feet at -0.95 to the crown
weights = np.zeros((len(rest), model.njoints), np.float32)  # one column per model joint, the universe included
weights[np.arange(len(rest)), np.digitize(rest[:, 1], [0.1, 0.55]) + 1] = 1.0  # the pelvis, chest or head joint
skin = Skin(rest, capsule.faces, weights=weights)
print(skin)
```
```output
Skin(V=1922, F=3840, njoints=4)
```

The body is y up and faces +z, as SMPL is; the G1 is z up and faces +x. Leave both arrays as they are and place
the body's tag with a pose (`Viewer.add(pose=)`): a third of a turn about `(1, 1, 1)`, the quaternion
`(0.5, 0.5, 0.5, 0.5)`, takes +y onto +z and +z onto +x, so the body stands upright, faces the way the robot does,
and is framed by `aim` and `record` where it stands.

```python
from robot_descriptions import g1_description
from better_robot import Viewer
from better_robot.viewer import RobotVisual

g1 = br.load(g1_description.URDF_PATH, root_joint=br.JointFreeFlyer())
g1_q = g1.q_neutral.repeat(T, 1)                   # yours comes from retargeting the body's clip
g1_q[:, 2] = 0.79                                  # the G1 stands on the ground
geometry = br.io.load_geometry(g1_description.URDF_PATH, g1)

viewer = Viewer()
robot = viewer.add(RobotVisual(g1, g1_q, geometry=geometry, root=g1_description.URDF_PATH))
Z_UP = (0.5, 0.5, 0.5, 0.5)                        # y up, z forward -> z up, x forward
body = viewer.add(RobotVisual(model, q, skin=skin, name="body"), pose=(0.0, 1.0, 0.0, *Z_UP))
body.set_style("minimal")                          # the body's skeleton; the sidebar's Style switches both
print(body.styles, viewer.timeline.nsteps)
```
```output
('meshes', 'minimal', 'medium', 'maximal') 60
```

The body takes the first of the theme's hues no other robot has; `RobotVisual(..., hue=(r, g, b))` gives it a colour
of its own. The skinned surface is a linear blend of the joints, without the pose correctives of `state.vertices`; it costs
one bone pose per joint and step. For the exact surface, show the vertices themselves under the same pose:
`viewer.add(MeshLayer("surface", state.vertices.numpy(), faces=human.spec.faces.numpy()), tags="surface",
pose=(0.0, 1.0, 0.0, *Z_UP))`, which sends `T x V x 3` floats. A body without a model, given only its bone poses
`(T, J, 7)`, is a `SkinnedMeshLayer(name, vertices, faces=, weights=, bind_poses=, bone_poses=)`.
On viser 1.1.1 a connected browser goes blank when `viewer.remove` or `viewer.replace` takes a body away, as a body
swap does (a bug of viser's client, fixed upstream in viser pull request 791); reloading the page shows the scene again.

## Record the clip

```python { .no-run data-reason="records through a headless Chromium; tests/viewer/test_record.py runs record" }
out = viewer.record("body_and_g1.mp4")             # 1920 x 1080, every step, the view aim() would choose
```

[Record a video](record-a-video.md) gives the options (size, steps, camera, GPU) and what the browser needs.

## Human terms mix with robot terms in one `Problem`

The terms follow the `TermFn` contract of `better_robot.optim`, so they mix freely with robot terms from
`better_robot.terms` in one `Problem`. Cache the human state once, then add terms that read it:

```python { .no-run data-reason="needs BetterHuman and licensed SMPL weights, which this repository does not install" }
from better_human import terms as human_terms

problem.cached(human_terms.human_state(identity), name="state")
# then add human_terms.joints(target=...), human_terms.points(vertex_ids, target=...),
# or human_terms.reprojection(...) as terms of the problem
```

To fit the shape as well as the pose, cache the identity from the shape variables with
`problem.cached(human_terms.baked_identity(human), name="identity")` and read it with
`human_terms.human_state(identity_name="identity")`.

## Where next

- [Fit a clip](fit-a-clip.md): a full trajectory fit in phases.
- [Write a term](write-a-term.md): the term contract the human terms follow.
- [Viewer](../concepts/viewer.md): layers, tags and their poses, and the skeleton styles.
