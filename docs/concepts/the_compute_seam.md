# The Compute Seam

BetterRobot has one public tensor API and one way to execute a complete
algorithm: the Torch implementation. It is the only implementation, and
`use_compile` is the only optional acceleration.

The seam sits around a **whole pass** such as forward kinematics or inverse
dynamics. It does not sit around each quaternion multiply or spatial cross
product. That granularity keeps the public API simple and gives
`torch.compile` enough work to matter. See {ref}`decision-whole-pass`.

## One shape convention

Public tensor operations use trailing event dimensions and leading batch
dimensions:

```text
configuration         (..., nq)
velocity or tangent   (..., nv)
pose                   (..., 7)
joint poses            (..., njoints, 7)
mass matrix            (..., nv, nv)
```

The leading `...` may be empty. A batch of one remains `(1, ...)`; it is not
silently squeezed. Compatible inputs follow right-aligned Torch broadcasting,
and outputs preserve the resolved execution batch.

The beginner-facing intuition and GPU example live in
{doc}`/getting_started/05_batched_gpu`. This chapter explains how the model
and the Torch passes honor the same contract.

## Structure and values cross the seam

A `Model` exposes two internal views:

- `ModelStructure` contains immutable topology: parent indices, joint kinds,
  coordinate offsets, traversal order, and flat metadata tables.
- `ModelValues` contains tensors that may need gradients: fixed joint and
  frame placements, inertias, limits, gravity, and related values.

The split keeps topology static and values differentiable. A Torch pass can
loop over Python tuples that `torch.compile` can unroll, and the compiled path
receives the same differentiable value tensors, so it does not invent another
robot representation.

Public wrappers validate call inputs against the attached model. Raw passes
consume the structure/value pair and trust that attachment boundary. This is
also useful for functional differentiation: a raw pass takes all of its tensor
inputs explicitly instead of closing over a mutable workspace.

Spatial inertia matrices are derived from the current packed
`ModelValues.body_inertias` tensor on every dynamics pass. `ModelValues` does
not carry an independently replaceable spatial-inertia cache, so functional
updates such as `dataclasses.replace` cannot retain stale physics or sever the
gradient route to replacement inertias.

## Raw passes return values

Tensor-only passes return frozen result records. This small CPU model exercises
both kinematics and dynamics through that seam:

```{testcode}
import torch
from better_robot.dynamics import rnea_raw
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics_raw

builder = ModelBuilder("raw_passes")
builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
builder.add_body("link", mass=1.0, inertia=torch.eye(3) * 0.05)
builder.add_revolute_z("joint", parent="base", child="link", lower=-3.14, upper=3.14)
model = build_model(builder.finalize(), dtype=torch.float64)
q = model.q_neutral
v = torch.zeros(model.nv, dtype=q.dtype)
a = torch.zeros_like(v)
fk = forward_kinematics_raw(model.structure, model.values, q)
rnea_result = rnea_raw(model.structure, model.values, q, v, a)
print("raw FK shape:", tuple(fk.joint_pose_world.shape))
print("raw RNEA shape:", tuple(rnea_result.tau.shape))
```

```{testoutput}
raw FK shape: (3, 7)
raw RNEA shape: (1,)
```

The public wrappers call the same mathematical pass and then populate `Data`
when that family uses a workspace. Kinematics fills and returns `Data`;
dynamics returns its main tensor and optionally fills a supplied `data=`
workspace. Raw result records make the functional return values explicit
without changing the public convention.

## Broadcast without copying every input

Query tensors and model values can have different but compatible leading
shapes. Internally, `ExecutionBatch` resolves their right-aligned broadcast
and flattens the common batch to an execution count `E`.
Index maps select which physical query and value row belongs to each execution
row.

This matters when one model-value batch is shared across many queries. The
implementation need not duplicate every inertia or placement tensor. During
backward, gathered gradients reduce to the original physical input shape.

The Torch path follows the same public broadcasting rules without requiring a
flat representation. `ExecutionBatch` is an implementation ABI, not a new
shape convention for callers.

## Torch is the implementation

The Torch pass has two jobs:

1. It is the CPU and GPU implementation.
2. It is the differentiable formula; ordinary `loss.backward()` differentiates
   it, so no separate derivative implementation exists.

Eager execution is useful for debugging. Callers may compile a stable raw pass
with `torch.compile` when their model and workload benefit. BetterRobot does
not change process-wide compiler or numerical settings.

Static topology and tensor control flow matter. Loops may depend on fixed
model metadata, but a hot path should not turn tensor values into Python
booleans or numbers. Such a conversion synchronizes an accelerator and can
freeze data-dependent behavior into a compiled graph.

## One implementation

There is no second lane to select and no process-wide compute selector. The
only optional acceleration is `use_compile` on `forward_kinematics`, which runs
the same Torch formulas through a `torch.compile` lane cached per structure.
Poses remain scalar-last and spatial vectors remain linear-first, and CUDA work
runs on Torch's current stream.

## Why workload size matters

Eager Torch recursions issue many small operations as they traverse a robot.
On a GPU, host dispatch and kernel launches can cost more than the arithmetic
for a tiny model or a single query. More GPU cores do not fix that launch-bound
case; a small CPU solve can be faster.

Compiling a whole pass can remove some of that dispatch cost. Conventional GPU
batching also pays off once many independent robots or a trajectory horizon
provide enough parallel tensor work. This is why large batches and
trajectory-optimization-sized workloads are better GPU candidates than the
one-joint teaching model in `examples/03_batched_ik.py`. The crossover still
depends on topology, dtype, hardware, and whether backward work is included,
so it is a measured property of a workload rather than a package-wide rule.

## What callers should rely on

- Public inputs and outputs are Torch tensors.
- Torch is the only implementation.
- A performance claim is scoped to its measured hardware, dtype, batch, and
  forward/backward workload.

## Where to continue

- {doc}`architecture` places the seam in the package dependency graph.
- {doc}`model_and_data` explains `ModelStructure`, `ModelValues`, and `Data`.
- {doc}`kinematics_and_jacobians` explains the FK pass itself.
- {doc}`/conventions/performance` states benchmark and hot-path rules.
- {doc}`/conventions/contracts` states public batch, dtype, and device
  behavior.
