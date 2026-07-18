# Model and Data

A robotics library answers two questions at very different rates. *What is
this robot?* is answered when a URDF, MJCF file, or builder becomes a
kinematic tree. *What does this robot look like at this configuration?* may be
answered thousands of times inside an optimizer.

BetterRobot separates those questions. `Model` holds robot identity:
topology, joint types, limits, inertias, frames, and numerical parameters.
`Data` is an optional workspace for the result of one calculation. A single
mutable robot object would mix long-lived identity with short-lived answers,
which makes batching, sharing, and autograd harder.

This split follows Pinocchio's successful `Model` / `Data` design. See the
{ref}`decision-model-data` decision for the alternative and its costs.

## `Model`: structure plus values

`Model` is a shallowly frozen dataclass. Its fields cannot be reassigned, but
Python cannot stop in-place mutation of a tensor stored inside it. Treat a
model as read-only after construction.

The model has two views of its contents:

- `ModelStructure` contains static topology and index tables. Python loops may
  walk this fixed structure, and `torch.compile` can specialize those loops.
- `ModelValues` contains tensors such as joint placements, body inertias,
  limits, and gravity. Replacing these values is how a differentiable model
  fit changes physical parameters without rebuilding the tree.

The public `Model` joins those two views for ordinary use. Power users can
call tensor-only `*_raw` passes with `model.structure` and `model.values`.
Those raw passes trust their caller; values are checked when attached to a
model, not during every robot calculation.

`model.to(device=..., dtype=...)` returns a new model with moved numerical
values. Keep separate CPU and GPU models instead of mutating one model in
place. `model.with_values(...)` returns a model with the same structure and
new checked values.

## Configuration and tangent coordinates

A configuration is named `q`. A velocity or small local change is named `v`.
They have shapes:

```text
q: (batch..., nq)
v: (batch..., nv)
```

`nq` counts stored configuration numbers. `nv` counts independent local
motion coordinates. They differ when a joint uses a redundant but convenient
representation. A spherical joint stores a four-number unit quaternion but
has three rotational degrees of freedom. A free-flyer stores seven pose
numbers and has a six-number twist, so it contributes `nq=7` and `nv=6`.

Use `model.integrate(q, v)` to apply a tangent change and
`model.difference(q0, q1)` to measure one. Do not use `q + v`: it is wrong for
quaternions, free-flyers, and other manifold-valued joints.

Public per-joint slices come from `model.idx_qs`, `model.nqs`,
`model.idx_vs`, and `model.nvs`. `model.q_permutation(names)` builds gather
indices when an external system stores the same joints in another order.
Zero-degree joints may be omitted from that external order.

The first entry in the topology is the universe, a zero-degree root. A
fixed-base robot attaches its first physical joint to it through a fixed
joint. A floating-base robot uses a free-flyer there. Nothing above
kinematics needs a separate floating-base code path.

## `Data`: one query's workspace

Kinematics returns a `Data` object because several related queries share its
placements and Jacobians. Common fields include:

```text
q                         (batch..., nq)
joint_pose_world          (batch..., njoints, 7)
frame_pose_world          (batch..., nframes, 7)
joint_jacobians           (batch..., njoints, 6, nv)
```

Unused fields stay `None`. `Data` therefore does not allocate every possible
robot quantity for every query.

Dynamics functions return their main tensor directly. Pass `data=workspace`
only when you also want the related cache fields populated. This keeps a
simple call such as `tau = rnea(model, q, v, a)` simple.

`Data` is mutable and belongs to one logical query. Do not share it between
concurrent calculations. Its batch shape follows the resolved leading axes of
the inputs.

## The cache rule

Some kinematics operations need earlier results. Frame placement needs joint
placement; a Jacobian needs a current kinematic state. `Data` records whether
placements, velocities, or accelerations are current. A consumer raises
`StaleCacheError` when its prerequisite has not been computed.

Assigning a new `q`, `v`, or `a` invalidates downstream caches. In-place
tensor mutation cannot be detected because it bypasses Python attribute
assignment:

```text
safe:    data.q = new_q
unsafe:  data.q.copy_(new_q)
```

Recompute the producer pass after assignment. If specialized code must mutate
in place, call `data.invalidate()` explicitly before reusing the workspace.

## Why names stay readable

Fields describe their contents: `frame_pose_world`, `mass_matrix`,
`bias_forces`, and `centroidal_momentum`. The
{doc}`/conventions/naming` page maps common Pinocchio abbreviations to these
names.

Continue with {doc}`joints_bodies_frames` for the objects that form the tree,
or {doc}`kinematics_and_jacobians` for the first calculation over it.
