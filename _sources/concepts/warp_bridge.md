# Torch–Warp bridge and opt-in FK lane

The Torch–Warp bridge is the shipped pattern for driving a fused whole pass
from the structure/value seam. The FK implementation is CUDA-validated but
remains opt-in: it is not a global backend and it is not selected for ordinary
calls. ``forward_kinematics(..., use_warp=True)`` attempts the fused lane and
falls back to the Torch raw pass when a joint kind, dtype, or layout is not
supported. Other Warp passes remain unimplemented.

## Functional boundary

The forward boundary is a pass-specific functional
``torch.library.custom_op``. Torch allocates fresh output tensors; Warp aliases
those tensors and the Torch inputs without copying, then launches on Torch's
current CUDA stream through ``wp.stream_from_torch``. Caller-owned mutable
output buffers are deliberately excluded because Torch cannot register an
autograd formula for a non-functional custom operator.

| Choice | M1 decision | Evidence |
|---|---|---|
| Public integration | One plain ``use_warp`` branch on FK | No registry, Protocol, or global selector |
| Operator surface | Pass-specific functional custom op | FakeTensor shapes and ``torch.compile(fullgraph=True)`` test |
| Allocation | Fresh Torch outputs, then zero-copy Warp views | Pointer/stride layout tests; capture uses Torch's graph-pool-aware allocator |
| Stream | Torch current stream passed to Warp | Non-default-stream CUDA ordering test |
| Warp gradient ownership | ``wp.from_torch(..., requires_grad=False)`` | Avoids deferred Warp ``.grad`` synchronization and double accumulation |
| First-order VJP | Recompute the Torch FK table pass in the registered autograd formula | Gradcheck for q, joint placements, frame placements, and shared-value reduction |
| Second order | Grad-enabled backward recomputes with ``create_graph=True`` | Public bridge gradgradcheck, including zero joint angle |
| Capture | Private bridge selector and direct functional forward op are CUDA-graph replayable | CUDA replay parity tests; unsupported private-selector fallbacks hard-error during capture |

The registered forward formula performs its Torch-lane recomputation directly.
A custom-op implementation executes below Torch's Autograd dispatch key, so it
cannot record the Torch oracle needed to build this VJP. No separate backward
custom-op is registered until a hand-written Warp VJP has an in-tree caller and
its own parity evidence.

## Execution-batch and layout rules

The bridge consumes the flat-``E`` ``ExecutionBatch`` ABI. Query and value
batches remain physically unique and maps select their row for each execution
element. The Torch recompute uses ``index_select``, so autograd reduces a model
value shared by multiple execution rows back to its original shape.

Poses stay scalar-last ``[tx, ty, tz, qx, qy, qz, qw]``. Contiguous fp32 pose
rows alias ``wp.transformf`` and fp64 rows alias ``wp.transformd``. An
unsupported inner stride falls back to Torch; the bridge never inserts a
silent ``contiguous()`` copy. BetterRobot spatial vectors remain
``[linear, angular]`` and are never reinterpreted as Warp's angular-first
spatial type.

## FK kernel decision record

| Pass | Thread mapping | Differentiable inputs | Adjoint strategy | Status |
|---|---|---|---|---|
| Fused FK + frames | One thread per execution row; serial topological loop | q, joint placements, frame placements | Torch-lane recompute; Warp generated adjoint rejected because dynamic-loop locals are not replayed reliably | CUDA-validated opt-in; default review pending |
| Jacobian / RNEA / ABA / CRBA | Undecided | Must be declared per pass | Must be re-evaluated per pass | Not implemented |

The FK kernel uses stable int8 kind codes and flat topology/value tables. Tests
cover fp32/fp64, branches, a chain deeper than 16 joints, free-flyer and
spherical joints, broadcast maps, shared gradients, and the zero-angle seam.
CUDA validation on RTX 6000 Ada covers fp32/fp64, fixed/free bases, branched
and >16-deep trees, multi-axis/value batches, q and placement VJPs, a numerical
zero-angle gradcheck, current-stream ordering, and graph replay. The committed
SMPL FK cases in `tests/bench/baselines/warp_fk_cuda_rtx6000_ada_b*.json` report
forward-only wins over compiled Torch at every measured batch size. Any
default-on decision remains an owner review because the backward path is a
Torch recomputation and was not part of that timing.

## Known limits

- The Torch VJP currently reads immutable topology tables to the host. That is
  correct but is not suitable for a captured hot backward loop.
- The private bridge selection path and direct custom op are capture-tested.
  An otherwise-silent dtype, model, batch, or layout fallback hard-errors
  during capture instead of baking a hidden lane change into the graph.
- The public ``forward_kinematics(..., use_warp=True)`` facade is not itself
  exercised inside a CUDA graph, so the private bridge evidence must not be
  described as public-facade capture certification.
- FK CUDA graph replay has been validated for the forward op only. The
  separate internal solver harness captures nonlinear jacrev work, but the
  public solver does not yet capture an end-to-end FK/backward/iteration
  lifecycle.
- Warp is an optional dependency. Importing BetterRobot without the ``warp``
  extra leaves the Torch lane fully functional.
