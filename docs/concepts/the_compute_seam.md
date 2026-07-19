# The Compute Seam

BetterRobot has one public tensor API and more than one possible way to execute
a complete algorithm. The ordinary Torch implementation is the reference. A
pass may also have a fused implementation when measurements justify it and
value, gradient, stream, and fallback behavior are tested.

The seam sits around a **whole pass** such as forward kinematics. It does not
sit around each quaternion multiply or spatial cross product. That granularity
keeps the public API simple and gives a fused kernel enough work to matter.
See {ref}`decision-whole-pass` and {ref}`decision-warp-kernels`.

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
and optional fused passes honor the same contract.

## Structure and values cross the seam

A `Model` exposes two internal views:

- `ModelStructure` contains immutable topology: parent indices, joint kinds,
  coordinate offsets, traversal order, and flat metadata tables.
- `ModelValues` contains tensors that may need gradients: fixed joint and
  frame placements, inertias, limits, gravity, and related values.

The split serves both execution styles. A Torch pass can loop over Python
tuples that `torch.compile` can unroll. A device kernel can read flat topology
tables. Both receive the same differentiable value tensors, so the accelerated
path does not invent another robot representation.

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

Tensor-only passes return frozen result records. For example:

```text
fk = forward_kinematics_raw(model.structure, model.values, q)
joint_pose_world = fk.joint_pose_world

rnea_result = rnea_raw(model.structure, model.values, q, v, a)
tau = rnea_result.tau
```

The public wrappers call the same mathematical pass and then populate `Data`
when that family uses a workspace. Kinematics fills and returns `Data`;
dynamics returns its main tensor and optionally fills a supplied `data=`
workspace. Raw result records make the functional return values explicit
without changing the public convention.

## Broadcast without copying every input

Query tensors and model values can have different but compatible leading
shapes. Internally, `ExecutionBatch` resolves their right-aligned broadcast
and flattens the common batch to an execution count `E` for a fused pass.
Index maps select which physical query and value row belongs to each execution
row.

This matters when one model-value batch is shared across many queries. The
implementation need not duplicate every inertia or placement tensor. During
backward, gathered gradients reduce to the original physical input shape.

The Torch path follows the same public broadcasting rules without requiring a
flat representation. `ExecutionBatch` is an implementation ABI, not a new
shape convention for callers.

## Torch is the reference

The Torch pass has three jobs:

1. It is the ordinary CPU and GPU implementation.
2. It is the correctness oracle for any fused lane.
3. It is the differentiable formula used when an accelerated forward pass
   does not own a complete derivative implementation.

Eager execution is useful for debugging. Callers may compile a stable raw pass
with `torch.compile` when their model and workload benefit. BetterRobot does
not change process-wide compiler or numerical settings.

Static topology and tensor control flow matter. Loops may depend on fixed
model metadata, but a hot path should not turn tensor values into Python
booleans or numbers. Such a conversion synchronizes an accelerator and can
freeze data-dependent behavior into a compiled graph.

## The current fused lane: forward kinematics

The only shipped fused pass is an opt-in forward-kinematics implementation in
[NVIDIA Warp](https://nvidia.github.io/warp/stable/index.html). Callers select
it explicitly:

```text
data = forward_kinematics(model, q, compute_frames=True, use_warp=True)
```

For an eligible CUDA model, dtype, and layout, the wrapper uses the fused FK
operation. Otherwise it falls back to the Torch pass. Ordinary calls do not
select Warp automatically, and there is no process-wide compute selector.

Other dynamics and Jacobian passes do not currently have Warp
implementations. The presence of a seam is not evidence that a second
implementation exists.

## A functional custom operation

The fused boundary is a pass-specific `torch.library.custom_op`. Torch owns
fresh output tensors, Warp views those tensors and the inputs without a copy,
and the kernel launches on Torch's current CUDA stream.

Caller-owned output buffers are excluded from this boundary. A functional
operator gives Torch clear aliasing and autograd semantics and can provide a
FakeTensor shape function for compilation. Mutating an arbitrary output buffer
would make those responsibilities ambiguous.

## Gradient ownership

The current Warp FK forward kernel does not also claim ownership of the
gradient. Its registered autograd formula recomputes the Torch FK table and
uses that graph for the vector-Jacobian product. This costs more than a fused
hand-written backward, but it has two important properties:

- derivatives of `q` and differentiable model placements match the reference;
- higher-order derivatives follow the same direct Torch formulas.

Warp views are created without deferred Warp gradients, which avoids two
systems accumulating into the same input. A future fused backward would need
its own callers, parity tests, and higher-order policy before replacing this
clear ownership model.

## Layout and stream rules

Poses remain scalar-last and spatial vectors remain linear-first at the seam.
The implementation does not reinterpret BetterRobot spatial vectors as
Warp's differently ordered spatial type.

Supported contiguous pose rows can be viewed directly as Warp transforms. An
unsupported inner stride falls back to Torch; the public path does not insert
a hidden `contiguous()` copy. The launch uses Torch's current CUDA stream so
operations issued on a non-default stream remain correctly ordered.

## CUDA graph capture

The private functional FK operation and its direct selection path have
forward replay coverage under CUDA graphs. Unsupported input during capture
raises instead of silently recording a different lane into the graph.

That evidence has a deliberate boundary. The public FK wrapper is not claimed
as an end-to-end captured solver interface, and the Torch-recomputed backward
is not claimed as a captured hot backward loop. Capture support is described
only for the path that is actually tested.

## When a fused pass is justified

A second implementation is worthwhile only when all of these are true:

1. A real workload shows the whole pass is important.
2. Fusion can remove enough launches or intermediate traffic to help.
3. Value parity covers branches, batches, dtypes, and supported joint kinds.
4. Gradient ownership and fallback behavior are explicit.
5. Stream, compilation, and capture claims match tests.
6. Unsupported inputs retain the Torch result rather than changing semantics.

This threshold is intentionally higher than “a kernel can be written.” Raw
CUDA is rejected because maintaining another source language and derivative
surface is not justified for this project; the full trade is recorded in
{ref}`decision-warp-kernels`.

## What callers should rely on

- Public inputs and outputs are Torch tensors.
- Leading batch behavior does not depend on the chosen lane.
- Torch is always the reference and fallback.
- Lane selection is explicit and pass-specific.
- No fused implementation is implied for a pass unless its public function
  documents one.
- A performance claim is scoped to its measured hardware, dtype, batch, and
  forward/backward workload.

## Where to continue

- {doc}`architecture` places the seam in the package dependency graph.
- {doc}`model_and_data` explains `ModelStructure`, `ModelValues`, and `Data`.
- {doc}`kinematics_and_jacobians` explains the FK pass itself.
- {doc}`/conventions/performance` states benchmark and hot-path rules.
- {doc}`/conventions/contracts` states public batch, dtype, and device
  behavior.
