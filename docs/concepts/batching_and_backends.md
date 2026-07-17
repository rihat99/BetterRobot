# Batching and Compute Lanes

Tensor math uses the convention ``(B..., feature)``, where ``B...`` may be
an empty or multi-axis batch prefix and ``feature`` is the semantic last
axis. A single pose is ``(7,)``; a batch of one is ``(1, 7)``. FK,
residuals, and analytic Jacobians support leading batches. The current
optimizer stack and ``solve_ik`` are explicitly single-problem until M2b.

BetterRobot has one public tensor API and a whole-pass compute seam. The
canonical lane is eager or caller-compiled Torch. An opt-in Warp lane may
replace an entire pass such as FK or RNEA when that pass has a validated
kernel and adjoint. Lane choice is not exposed on individual Lie operations,
and there is no process-wide compute selector or generic compute object.
Where a prototype exists, a pass-specific flag such as
``forward_kinematics(..., use_warp=True)`` opts into that whole pass. Public
calls continue to take and return ``torch.Tensor`` objects.

## Shape convention

For every tensor field on ``Model`` or ``Data`` and every argument or return
of a public function:

```
Last dim:        the manifold / feature dim
                     3 for position
                     4 for quaternion
                     6 for twist / tangent / spatial wrench
                     7 for SE(3) pose
                     nq / nv for configurations and velocities
Second-to-last: (optional) per-joint, per-link, or per-frame dim
Second dim:      (optional) time axis T (trajectories only)
First dims:      arbitrary leading batch prefix B...
```

A tensor routine treats a missing batch prefix and arbitrary leading batch
prefixes through the same trailing-feature convention.

### Examples

| Object | Shape |
|--------|-------|
| Single SE(3) pose | ``(7,)`` |
| Joint configuration | ``(B..., nq)`` |
| ``forward_kinematics`` joint poses | ``(B..., njoints, 7)`` |
| Spatial Jacobian | ``(B..., 6, nv)`` |
| Self-collision residual | ``(B..., n_candidate_pairs)`` |
| Trajectory | ``(B, T, nq)`` |

## Batching in practice

```python
q = torch.rand(4096, model.nq, device="cuda")
data = forward_kinematics(model, q, compute_frames=True)
poses = data.frame_pose_world[..., model.frame_id("body_panda_hand"), :]
```

The Torch FK lane loops over ``model.structure.topo_order``, a fixed Python
tuple, rather than over the batch. Per-joint operations retain the leading
batch prefix. This is the same source for an unbatched configuration and a
multi-axis batch.

## The structure/value seam

A ``Model`` exposes two views used by whole-pass implementations:

- ``ModelStructure`` is immutable topology. Python tuples make the Torch
  lane statically unrollable, while flat integer and floating-point tables
  expose the same topology to device kernels. ``validate_consistency``
  checks the two representations at construction.
- ``ModelValues`` contains differentiable tensor values such as joint
  placements, inertias, limits, gravity, and mimic metadata. It is
  registered as a Torch pytree so a raw pass can accept tensors without
  closing over a mutable ``Model`` object.

The raw Torch passes consume that seam directly:

```python
joint_pose_world, joint_pose_local = forward_kinematics_raw(
    model.structure,
    model.values,
    q,
)

rnea_result = rnea_raw(
    model.structure,
    model.values,
    q,
    v,
    a,
)
```

The public wrappers keep the familiar ``Model`` / ``Data`` API. They call a
whole pass, then populate the mutable workspace. This boundary is where a
kernel lane is selected; Lie functions such as ``se3.compose`` remain direct
Torch operations.

## Whole-pass lane choice

Lane selection is explicit at an integration boundary and coarse enough to
make ownership clear:

```python
if use_warp:
    outputs = try_warp_forward_kinematics(model.structure, model.values, q)
if outputs is None:
    outputs = forward_kinematics_raw(model.structure, model.values, q)
```

``forward_kinematics(..., use_warp=True)`` is the current pass-specific
opt-in. The Torch lane is the correctness oracle and default. The Warp FK
prototype checks its supported joint kinds, device, dtype, and layout before
running. Unsupported inputs intentionally fall back at that whole-pass
boundary; individual math operations never switch lanes mid-pass.

This design avoids several ambiguous states:

- no global selector captured accidentally by ``torch.compile``;
- no per-operation mixture of Torch and Warp within one pass;
- no runtime objects from an optional compute package in the public API;
- no second implementation of elementary Lie algebra chosen at runtime.

## Flat execution batches

``ExecutionBatch`` is the stable ABI for kernels that flatten a broadcast
batch to ``E`` execution elements. Each unique physical input remains stored
once; an integer map associates execution rows with source rows. That avoids
stride-zero views and avoids physically repeating shared model values.

For example, a query batch ``(Bq, nq)`` and a compatible values batch can be
broadcast, flattened to ``(E, ...)``, executed, and unflattened back to the
broadcast batch shape. Backward reduces per-execution gradients to each
input's unique rows with ``index_add_``. The ABI does not yet promise a
heterogeneous collection of robot topologies; each pass still receives one
``ModelStructure``.

## Device and dtype

``Model.to(device, dtype)`` returns a new model. ``ModelStructure.to`` moves
its device tables while preserving integer dtypes, and ``ModelValues.to``
moves the floating-point tensor pytree. The Python topology remains static.

``Data`` follows the query tensor. Supported paths preserve the query device
and working dtype. fp32 is primary and fp64 is used for derivative checks;
fp16 and bf16 are outside the numerical support contract even where an eager
Torch operation happens to execute.

## ``torch.compile`` discipline

The package requires Torch 2.4 or newer. The floor covers the functional
custom-op registration used by the opt-in Warp bridge as well as the
documented compilation path.

The raw Torch passes are the compilation boundary:

- topology loops use ``ModelStructure`` Python tuples;
- there is no Python branching on tensor values;
- hot paths avoid ``.item()`` and host transfers;
- joint dispatch is derived from stable structure data;
- values arrive as tensor leaves rather than through hidden global state.

Automatic compilation is not installed. Callers may compile documented raw
passes explicitly; see {doc}`/conventions/performance` for current coverage.

## CUDA graph capture

CUDA graph capture is roadmap work. A capture-safe solver must use fixed
storage and record forward **and backward together** so replay preserves the
intended differentiation lifecycle. BetterRobot does not currently ship a
capture decorator or context manager. Named-block LM's tensor-only state and
pure, fixed-shape `update` are capture-ready by construction, but a CPU
fullgraph smoke test is not certification. Capture remains opt-in until M6's
actual replay-parity harness covers the solver lifecycle and kernel adjoints.

## Requirements for an opt-in kernel lane

A whole-pass kernel is eligible only when it provides:

1. Torch-compatible tensor inputs and outputs at the boundary.
2. Explicit support checks for joint kinds, dtype, device, and shapes.
3. Forward parity against the raw Torch pass.
4. A tested adjoint exposed through Torch autograd, without leaking optional
   runtime array types.
5. Stable allocation and capture behavior for repeated shapes.

The optional runtime import stays local to the kernel integration. Elementary
Lie operations and ordinary public calls remain available without it.

## Testing the contract

Every public tensor routine needs shape, device, dtype, and gradient tests in
proportion to its guarantees. Every opt-in lane additionally needs the same
forward and backward fixtures as the Torch oracle, including rejection or
fallback tests for unsupported structures.

## Sharp edges

- Quaternions are scalar-last ``[qx, qy, qz, qw]``.
- Spatial Jacobians store linear rows above angular rows.
- ``torch.compile`` may specialise on shape, dtype, device, and topology.
- Mixed precision is not a supported robotics-numerics contract.
- Lane choice belongs at a whole-pass integration point, never inside a Lie
  primitive or mutable process-global setting.

## Where to look next

- {doc}`architecture` — the layered dependency graph and ownership rules.
- {doc}`lie_and_spatial` — the direct Torch math layer.
- {doc}`/conventions/performance` — compile, capture, and kernel requirements.
- {doc}`/conventions/extension` — how an experimental whole-pass lane is
  integrated without changing the public API.
