# Engineering Contract

> **Status:** normative for new work. This document separates the intended
> contract from what the current implementation actually enforces. A policy
> marked **gap** is not a shipped guarantee until its boundary check and test
> land.

These rules constrain both compute lanes and the solver redesign. They are
deliberately small: unsupported behavior fails clearly instead of becoming an
accidental API.

## Implementation status

| Area | Status at M1 | Evidence / limitation |
|------|--------------|-----------------------|
| fp32/fp64 dtype preservation | Enforced on covered paths | FK and `solve_ik` have fp32/fp64 preservation tests. |
| fp16/bf16 rejection | Enforced at the FK/pass boundary | FK and every dynamics pass that enters through FK raise `DtypeMismatchError`. Other standalone math primitives remain tensor operations. |
| Model/query dtype matching | Enforced at the FK/pass boundary | Query and joint-placement dtypes must agree before a hot pass begins. |
| Deeply immutable model state | **Gap** | `Model` is a frozen dataclass, but its tensors, dictionaries, and `meta` members remain mutable. |
| Versioned serialization | **Gap** | There is no structure/values `state_dict` API; `Model.meta` retains builder IR and resolver objects. |
| Differentiable solve | Unsupported | `solve_ik` detaches its initial iterate and has data-dependent control flow. |
| Warp differentiation/capture | Not yet a production contract | Promotion requires the bridge and kernel acceptance tests described below. |

## Dtype and numerics

- fp32 is the primary working dtype; fp64 is supported for accuracy and
  derivative checks. Public numerical entry points accept only these two.
- fp16 and bf16 raise `DtypeMismatchError` at the FK/pass boundary. Warp has
  no bf16 value type, and neither low dtype has a validated convergence budget.
- Query tensors and model values must have one device and one dtype. Boundaries
  reject a mismatch rather than silently moving or casting either side.
- Floating outputs, residuals, Jacobians, accumulations, and factorizations use
  the working input dtype. An algorithm that deliberately promotes an internal
  reduction must document and test that exception; the public result still
  preserves the working dtype.
- Small-angle Lie branches use a cutoff on squared angle: `1e-5` for fp32 and
  `1e-8` for fp64. Accuracy assertions use routine- and dtype-specific
  tolerances; a single tolerance for both dtypes is not acceptable.
- BetterRobot does not change PyTorch's process-global TF32 flags. Numerical
  guarantees for solver normal equations, dynamics reductions, and Jacobian
  comparisons assume TF32 is disabled. A benchmark may opt into TF32 only when
  it labels the mode and reports convergence/accuracy separately. Enforcing
  this locally on convergence-critical CUDA operations remains a gap.

“Preserve input dtype” means no hidden `.float()`, default-dtype constructor, or
return cast anywhere on a supported path. Tests must cover both fp32 and fp64.

## Quaternion double cover and continuity

Unit quaternions represent rotations modulo `q ~ -q`; raw component distance
is therefore never a rotation residual.

- SO(3)/SE(3) logarithms use the principal branch: flip when `qw < 0`, yielding
  an angle in `[0, pi]`. Pose/orientation residuals and manifold priors operate
  on a relative transform and then apply this log.
- Interpolation chooses the shortest arc by aligning the second endpoint so
  `dot(q0, q1) >= 0`. A temporal sequence is aligned successively to its
  preceding sample before any component-space filtering or parameter fitting.
- Rest, reference-trajectory, velocity, and acceleration terms use
  `Model.difference`; they must not subtract quaternion components directly.
  BetterHuman pose parameters follow the same rule when migrated in M4.
- Exactly at angle `pi`, the log axis has an unavoidable sign ambiguity. With
  `qw == 0`, the current principal-log convention keeps the input sign.
  Rotation values remain valid there, but continuity and derivatives of the
  chosen tangent are not guaranteed. Tests near the cut use one-sided samples
  and sign-invariant rotation comparisons; they do not demand a derivative at
  the discontinuity.

Serialization may retain a quaternion's sign. Canonicalization is an operation
at a metric/interpolation boundary, not a mutation of stored user data.

## Threading, reentrancy, and streams

- `ModelStructure` and `ModelValues` are read-only after construction. “Frozen
  dataclass” alone is insufficient: exported mappings cannot be mutable and
  callers must not mutate shared tensor storage.
- `Data`, residual evaluation caches, solver state, histories, and warm starts
  belong to one evaluation. Concurrent evaluations never share them. Cached
  basis objects and other mutable helpers also require one instance per solve.
- Process-wide compile/kernel caches may contain immutable programs only. Cache
  insertion must be race-safe, and a cache hit cannot expose mutable evaluation
  buffers. BetterRobot never calls `torch.set_num_threads`.
- CUDA work is enqueued on the caller's current PyTorch stream. A Warp launch
  bridges that exact stream with `wp.stream_from_torch`; it does not use an
  implicit default stream or synchronize unless an API explicitly documents a
  host-visible check.
- Multiprocessing with an accelerator supports the `spawn` start method. Each
  child initializes Torch/Warp and reconstructs model values from serialized
  state. Forking a CUDA-initialized process and sharing `Data`, solver objects,
  streams, or graph executables is unsupported.

The current shallowly frozen `Model` is safe to share only if the application
treats every contained tensor/dictionary as read-only. Deep immutability is a
seam acceptance item, not something the type annotation already proves.

## Serialization

The stable checkpoint boundary is a versioned structure/values state mapping,
not a pickled Python object:

- The mapping carries a `format_version`, static structure fields, tensor model
  values, and a small allow-listed JSON-compatible metadata section.
- `map_location` applies to every tensor value while static topology remains on
  the host. Loading validates the format version and structure/value shapes
  before constructing a model.
- Optimizer state and warm-start state are separate optional mappings with
  their own versions. Evaluation caches, compiled programs, streams, graph
  executables, and solver histories are never model state.
- `Model.meta` is not serialized wholesale. Builder IR, asset resolvers,
  callables, open handles, and environment-specific paths must not enter a
  checkpoint. Provenance fields are copied only through an explicit allow-list.
- Pickle is neither a stable format nor a trusted-input boundary. Pickled
  `Model`/`Data` objects receive no compatibility guarantee and must never be
  loaded from an untrusted source.

Until this API lands, persist source assets plus explicit tensor parameters;
do not advertise `torch.save(model)` as a supported checkpoint.

## Differentiation

Guarantees are per lane and per input, not implied by “PyTorch-native”:

| Input / path | First order | Second order |
|--------------|-------------|--------------|
| Torch Lie maps and FK with respect to `q` | Required; covered by eager tests | Required only where gradgradcheck exists, including the zero-angle Lie seam |
| Torch RNEA/ABA with respect to state/control tensors | Required; eager gradcheck coverage | Not guaranteed |
| `ModelValues` (placements, inertias, limits) | Required after the seam's per-input matrix is complete | Not guaranteed unless named by a kernel test |
| Tensor residual parameters | Required for parameters declared differentiable by that residual | Not guaranteed by default |
| Python weights/configuration and solver hyperparameters | Not differentiable | Not differentiable |
| Current `solve_ik` result with respect to its inputs | Not guaranteed | Not guaranteed |
| Warp custom-op path | Required only after torch-lane parity + gradcheck | Recompute the torch VJP in backward and pass gradgradcheck before promotion |

Unrolled differentiation through a future fixed-iteration solver must be an
explicit mode and return the derivative of the iterations actually executed.
Implicit differentiation is a separate API and is valid only for a converged
solution satisfying its linear-solve conditions. A non-converged solve returns
its per-element status; gradients through it are unavailable unless that API
explicitly defines an unrolled failure behavior. No path silently substitutes a
zero or stale gradient.

Branching, mutation, `.detach()`, and custom operators are reviewed against this
matrix. Every new Warp kernel records its differentiable inputs and adjoint
strategy next to its parity tests.

## Compile lifecycle

- Topology, joint kinds, feature widths, batch rank, lane, dtype, device,
  layout/strides, and compile options are static specialization inputs. The
  flattened execution size `E` is the only candidate dynamic tensor dimension;
  if a compiler cannot keep it dynamic, `E` becomes part of the cache key.
- A graph-cache key includes a topology fingerprint, the static inputs above,
  BetterRobot/Torch/Warp versions, and a kernel/source hash. Cache limits are
  bounded and observable; eviction changes latency, never results.
- Cold compilation is part of performance reporting. The historical first FK
  compile was about 31 seconds on one measured setup; that is evidence, not an
  SLA. Benchmarks report cold codegen separately from warmed execution.
- Phase changes keep tensor shapes fixed and use zero weights when semantics
  allow it. A genuinely different residual structure uses a separately compiled
  program rather than disguising a shape change inside a captured graph.
- Warp CI uses an explicit disposable cache directory. Cached kernels are reused
  only under the complete version/source key, and first-call codegen latency is
  measured before a kernel can be called production-ready.
- A CUDA graph is re-recorded when storage addresses, shapes/strides, dtype,
  device, topology/layout, stream, launch geometry, or program sequence changes.
  Changing tensor values alone does not require re-recording. Capture performs
  no allocation, fallback, compilation, or host synchronization.

The one current full-graph FK test proves only its named shape/model case. It is
not evidence that the broader lifecycle or cache policy is implemented.

## Provenance gate

No external implementation is copied or ported until the owner chooses a
project license and its source-ledger entry is reviewed. See
{doc}`source_and_license` for the decision memo and mandatory ledger fields.
