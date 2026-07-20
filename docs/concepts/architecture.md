# Architecture

BetterRobot is split into layers that answer different questions. The lower
layers know mathematics and robot structure. The upper layers compose those
pieces into estimation tasks. Imports point downward, so a low-level tensor
operation never needs to know which solver or viewer called it.

The dependency shape is:

```text
viewer
  |
tasks
  |
optim
  |
residuals
  |
io   kinematics   dynamics   collision
 \       |           |          /
          data_model
              |
           spatial
              |
             lie
```

Packages on the same row have the same architectural rank, although most use
fewer dependencies than the rule permits. A contract test parses imports under
`src/better_robot` and reports any upward edge.

## Start with the mathematical floor

`lie` contains direct Torch functions for rotations and rigid transforms.
They take and return tensors, so autograd, batching, and compilation can see
the operations without an object-dispatch layer.

`spatial` builds the six-dimensional motion, force, and inertia operations
used by rigid-body dynamics. It depends on the pose conventions in `lie`, but
it has no reason to know about a robot model or an optimizer.

These packages are intentionally small in responsibility. Changing a solver
must not change how an SE(3) exponential is computed. The choice to keep Lie
operations functional is recorded in {ref}`decision-functional-lie`.

## Robot identity and query state

`data_model` owns the robot itself. A `Model` combines two sources of truth:
`ModelStructure` holds topology, joint kinds, names, and index tables, while
`ModelValues` holds placements, limits, inertias, gravity, and other tensors
that may participate in differentiation. The familiar flat `Model` attributes
are an explicit facade over those parts, not additional storage. A `Data`
object holds results for one execution batch: poses, velocities, Jacobians,
and other quantities produced by algorithms.

Separating structure from values lets algorithms reuse topology while values
are replaced, moved, batched, or differentiated. Keeping each field in one
part also prevents a flat model and its raw-pass inputs from drifting apart.
The further split from `Data` lets many queries share one model without
sharing mutable results. See {doc}`model_and_data` and
{ref}`decision-model-data`.

Model construction has a separate boundary. `io` parses URDF, MJCF, or a
programmatic builder into one intermediate representation, then
`build_model` validates it and constructs `ModelStructure` and `ModelValues`
directly. Format-specific choices stop there; kinematics does not contain
URDF cases. See {doc}`parsers_and_ir` and {ref}`decision-one-ir`.

## Algorithms consume the same model

`kinematics` computes joint and frame poses and their Jacobians. `dynamics`
computes inverse dynamics, forward dynamics, inertia matrices, centroidal
quantities, and related derivatives. Both use the same joint taxonomy and the
same configuration/tangent conventions.

The two packages sit at the same conceptual level. Dynamics may reuse a raw
kinematics pass, but kinematics does not import dynamics. A caller that only
needs poses therefore does not pull the dynamics implementation into its
execution path.

`collision` currently contains geometry containers and explicit placeholders
for capabilities that do not yet ship. Keeping it beside the robot algorithms
prevents an unfinished geometry surface from leaking into optimization.

## Residuals describe errors; optimization combines them

`residuals` turns a context into fixed-size error rows. A pose residual, for
example, reads frame placements and compares one frame with a target. The
package may use kinematics or dynamics concepts, but it does not import the
optimizer that will consume its output.

`optim` owns variables and their geometry, bounds, residual composition, robust
losses, Jacobian selection, and solvers. A `Problem` can optimize several
named variables, each with its own event shape and manifold. Providers place
shared work such as forward kinematics into one evaluation context so several
residuals can reuse it.

This direction is important: residual code describes the error, while the
problem decides how to differentiate, weight, and solve it. The resulting
least-squares model is explained in {doc}`residuals_costs_and_solvers`.

## Tasks are recipes, not a second solver stack

`tasks` contains convenient policies for common outcomes. `solve_ik` builds a
one-configuration robot problem. `solve_trajopt` builds a trajectory problem
with temporal structure. Both use the same `Problem` and solver classes that
direct callers use.

A task may choose default residuals, weights, bounds, and stopping settings.
It does not define a parallel optimization protocol. This keeps a simple call
simple without making advanced callers leave the supported path.

## The viewer stays at the edge

`viewer` translates model and trajectory data into interactive or offscreen
rendering. Nothing in the computational core imports it. Optional packages
such as `viser`, `pyrender`, and mesh loaders are imported only at their
integration boundaries, so `import better_robot` does not require a rendering
environment.

The same rule applies to optional acceleration. A fused implementation lives
beside the pass it accelerates and keeps tensors at the public boundary. It
does not become another architectural layer. See {doc}`the_compute_seam` and
{ref}`decision-whole-pass`.

## Two flows through the layers

A forward-kinematics call follows a short path:

```text
q + Model -> kinematics pass -> Data with joint/frame poses
```

An inverse-kinematics call composes more layers:

```text
targets -> task recipe -> Problem -> residual evaluation
                               -> kinematics provider -> poses
                               -> Jacobians and LM step -> new q
```

The second flow still bottoms out in the same kinematics functions as the
first. That is the practical benefit of the dependency rule: high-level
features compose the core instead of copying it.

## What enforces the boundary

Architecture is checked in code, not only described here:

- `tests/contract/test_layer_dependencies.py` rejects upward imports.
- Public-import tests pin the supported package surface.
- Optional-dependency tests import the core without viewer or kernel extras.
- Cache tests ensure algorithms do not read `Data` at an insufficient
  computation level.
- Whole-pass parity tests compare any accelerated lane with the Torch
  reference.

These checks leave room to extend a layer while keeping its direction clear.

## Where to continue

- {doc}`model_and_data` follows `Model` and `Data` through a query.
- {doc}`kinematics_and_jacobians` explains the most common algorithm flow.
- {doc}`residuals_costs_and_solvers` shows how residuals become an
  optimization problem.
- {doc}`the_compute_seam` explains reference and fused whole-pass execution.
