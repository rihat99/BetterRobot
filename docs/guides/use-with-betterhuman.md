# Use with BetterHuman

BetterHuman is a separate package that builds SMPL-family human bodies (SMPL, SMPL-H, MANO, SMPL-X) as
ordinary BetterRobot models. This page lists what it adds on top of BetterRobot and shows its quick
start, so you can tell which package owns which step.

## What BetterHuman adds

| Piece | What it is |
|-------|------------|
| `bh.load(family, gender=...)` | returns a `Body`: one BetterRobot `Model` per family plus the shape and skinning data |
| `Body.bake(betas=...)` | bakes a body shape once into an `Identity`: joint placements, masses, inertias and contact frames become `ModelValues` |
| `Identity.forward(q, stream=...)` | returns a `HumanState`: the `KinematicState` of the skeleton plus skinned `vertices()` and `joints` |
| `better_human.terms` | cost terms for [`better_robot.optim`](../concepts/optimization.md): `joints`, `points`, `landmarks`, `reprojection`, with `human_state` registered once as a cached value so one forward pass serves every term |
| `better_human.viz` | viewer layers for [`better_robot.viewer`](../concepts/viewer.md): `mesh_layer`, `skinned_layer`, `skeleton_layer`, `frames_layer`, `forces_layer`, and `HumanView`, which groups them and updates them in place |

Everything below the human layer is BetterRobot: the configuration `q (B, nq)` is a free-flyer root
followed by spherical joints, so `model.integrate`, `forward_kinematics_state`, the dynamics in
`better_robot.dynamics.state` and the solvers in `better_robot.optim` apply unchanged. Because shape
lives in `ModelValues`, gradients reach the shape parameters the same way they reach any model tensor
(see [Differentiate](differentiate.md)).

## Quick start

This is the quick start from `/home/rikhat.akizhanov/better/dev/BetterHuman/README.md`. It does not run
in the documentation tests: SMPL-family model files are licensed and are not shipped with either
package. Download them from their project pages and follow the BetterHuman README to convert them.

```python no-run
import torch
import better_human as bh
from better_human.viz import HumanView

body = bh.load("smpl", gender="neutral")         # one BetterRobot model per family
identity = body.bake(betas=torch.zeros(1, 10))   # bake shape once (mass, joints, contact frames)
q = identity.q_neutral.expand(4, -1).clone()     # (B, nq): free-flyer root + spherical joints
q[:, 1] = 1.0                                    # root 1 m up (y-up, metres)
state = identity.forward(q)                      # HumanState for 4 poses
print(state.vertices().shape, state.joints.shape)  # (4, 6890, 3) (4, 24, 3)

view = HumanView(identity.forward(identity.q_neutral))   # viewer layers, updated in place
# from better_robot.viewer import Viewer
# viewer = Viewer(); [viewer.add(layer, group="human") for layer in view]
# view.update(state_for_next_frame); viewer.show()
```

Per-frame groups such as the SMPL-X expression go to `forward`:
`identity.forward(q, stream={"expression": torch.zeros(B, 10)})`.

## Fit a pose with human terms

The terms follow the `TermFn` contract of `better_robot.optim`, so they mix freely with robot terms
from `better_robot.terms` in one `Problem`. Register the human state once, then add terms that read it:

```python no-run
from better_human import terms as human_terms

problem.cached(human_terms.human_state(identity), name="state")
# then add human_terms.joints(target=...), human_terms.points(vertex_ids, target=...),
# or human_terms.reprojection(...) as terms of the problem
```

To fit shape as well as pose, make the identity a variable with
`problem.cached(human_terms.shaped(body), name="identity")`. [Fit a clip](fit-a-clip.md) walks through
a full trajectory fit; [Write a term](write-a-term.md) explains the term contract.

<!-- TODO(api): the exact Problem construction for human terms (variables, cached values) was not
re-checked here; see better_human/terms.py and the BetterHuman README. -->
