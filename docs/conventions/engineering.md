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

## Differentiation contract

This section freezes the semantics that the M2 problem and solver state must
preserve; the implicit-solve implementation remains an M6 deliverable. The
current `solve_ik` is not differentiable and receives no guarantee from this
future contract. The owner approved these M2a freeze decisions on 2026-07-17.

Three input roles are distinct:

- **Optimized values** are the named `Values` blocks stepped by the solver.
  Their derivatives during direct problem evaluation live in each block's
  local tangent coordinates, after fixed-coordinate mask elimination.
- **External tensor parameters** affect the problem but are not stepped. A
  `Problem` must enumerate them by stable name in a tensor pytree and state
  whether each is differentiable. They include continuous `ModelValues` and
  residual parameters such as targets, tensor weights, and robust-kernel
  scales.
- **Static configuration** includes `ModelStructure`, names and index tables,
  masks, algorithm choices, and frozen solver hyperparameters. It is never a
  differentiation input.

A tensor occupies exactly one of the first two roles in one solve. Passing a
model placement as an optimized block, for example, makes it part of the
solution; passing it as an enumerated external `ModelValues` field makes it a
parameter of that solution. The role is declared, never inferred from
`requires_grad` or from an unreachable Python attribute.

### Guaranteed inputs

| Input category | Direct Torch problem evaluation | Future implicit solve |
|---|---|---|
| Active optimized `Values` blocks | First-order residual, objective, and Jacobian derivatives in local tangent coordinates | The returned optimum accepts a cotangent, but there is no implicit gradient to the initial guess; the trajectory is intentionally ignored |
| Continuous external `ModelValues` such as placements and inertias | First order when the field is declared differentiable and used by the evaluated path | First order when explicitly enumerated by `Problem` |
| Residual targets, tensor weights, and tensor kernel scales | First order when declared differentiable by the residual/kernel | First order when explicitly enumerated and locally smooth at the solution |
| Provider inputs | Inherit the guarantee above only when the provider preserves the graph and declares the dependency | Same; a deliberately detached provider output is a declared stop-gradient boundary |
| Initial optimized values | Ordinary direct derivatives apply before solving | No implicit gradient; an optimum is not a function of the initialization under the implicit contract |
| Bounds/limits, boolean masks, topology, names, integer tables | Not differentiable | Not differentiable |
| Frozen solver hyperparameters (`max_iter`, tolerances, damping policy, linear solver, step caps) | Not differentiable | Not differentiable |

Python-number weights and scales are static configuration. A caller that needs
a derivative with respect to a weight or kernel scale must supply it through a
residual's declared external tensor parameters. Solver hyperparameters do not
become differentiable merely by wrapping them in tensors; accepting a
`requires_grad=True` hyperparameter must fail clearly rather than silently
discarding its gradient.

Limits are excluded from the initial solve-gradient contract. At a stable
active set, an implicit backward may differentiate the solution with respect
to other declared external parameters while holding the bound values fixed.
Sensitivity with respect to the bound values themselves, discrete topology,
or an active-set change is not guaranteed.

### Derivative order and lane

Guarantees are per path, not implied by “PyTorch-native”:

| Path | First order | Second order |
|---|---|---|
| Torch Lie maps and FK with respect to `q` | Required; covered by eager tests | Required only where gradgradcheck exists, including the zero-angle Lie seam |
| Torch RNEA/ABA with respect to state/control tensors | Required; eager gradcheck coverage | Not guaranteed |
| M2 tangent-autograd helper and AD-generated Jacobian blocks | Required for active optimized blocks and declared external tensors | `create_graph=True` must retain a usable graph on the Torch lane; numerical correctness is guaranteed only for residuals/operations with a named gradgrad test |
| Continuous `ModelValues` | Required according to the per-input matrix above | Not guaranteed unless a named path has gradgrad coverage |
| Custom residual/provider code | First order is a declared capability and must be tested | Opt-in capability; unsupported `create_graph=True` raises instead of returning detached blocks |
| Future implicit solve | First order to declared external tensor parameters, on the Torch lane | Not part of the initial M6 contract |
| Future unrolled solve | Not a stable public guarantee; see below | Not guaranteed |
| Warp custom-op path | Required only after Torch-lane parity and gradcheck | Recompute the Torch VJP in backward and pass public gradgradcheck before promotion |

`create_graph=False` is the normal evaluation mode and must release graphs at
the end of the evaluation. `create_graph=True` is a capability request, not
permission to return a partially detached result. An analytic block or custom
provider that cannot honor it raises an actionable unsupported-operation error.
Second derivatives are not guaranteed at genuine nonsmooth boundaries,
including the quaternion branch cut, robust-kernel kinks, active-set changes,
or discrete mask/topology changes.

### Solver differentiation modes

The names below describe frozen semantics; their final API spelling is not yet
public.

1. **Detached (default).** `run` does not retain an iteration graph. Loop-carry
   values, accepted-state artifacts, warm starts, and returned state are
   detached, while local autograd may still be used to construct one
   iteration's derivatives. Backpropagating through the returned solution is
   unsupported and must not appear to succeed with stale graph fragments.
2. **Implicit (future explicit opt-in).** M6 differentiates the robustified
   tangent-space optimality/KKT system at the terminal solution with a custom
   backward. The forward trajectory is not retained. This mode returns
   first-order gradients only to the declared external tensor parameters; it
   returns no gradient to initialization or solver hyperparameters. Its
   backward must recompute or recover the final-point system rather than reuse
   a damped or stale trial-point factorization.
3. **Unrolled (not a stable solver guarantee).** Pure `update` and the
   evaluation protocol must not preclude a fixed-trip, masked-no-op unroll used
   as a small-problem oracle and research tool. If a public mode is added
   later, it is explicit, first-order-only initially, and differentiates the
   exact iterations executed with respect to initialization and declared
   tensor parameters. That derivative is an algorithm-trajectory derivative,
   not an implicit derivative of the converged optimum. A data-dependent
   Python early break cannot define this mode.

### Terminal and nonsmooth cases

Implicit differentiation is valid per batch element only when the terminal
point satisfies the relevant optimality conditions and the backward linear
system succeeds:

- `converged` is eligible after the final optimality check.
- `stalled_at_bounds` is eligible only when it denotes KKT satisfaction, the
  active set is locally stable, and the backward uses the free-coordinate/KKT
  system. Active coordinates then have zero sensitivity to other external
  parameters while their fixed bound values remain nondifferentiable.
- `maxiter`, `failed`, a non-KKT bounds stall, a changing/degenerate active
  set, a nonsmooth optimality point, or a singular backward system is
  gradient-invalid.

The initial implicit API is strict: if a requested batch contains any
gradient-invalid element, backward raises a documented differentiation error
that identifies the per-element statuses. It does not return zero,
best-effort, or last-iterate gradients. A future partial-gradient mode may mask
invalid elements only as an explicit opt-in and must return the validity mask;
zeroing is never the silent default. A future unrolled mode may differentiate
a non-converged last iterate because it promises the trajectory derivative,
but it must preserve the non-converged status alongside that result.

### Requirements on M2 problem and solver state

M2b must preserve the following shape and graph-lifetime properties even
though it does not implement implicit backward:

1. Solver state is a fixed-structure, plain tensor pytree. Every per-element
   mutable field carries the full leading batch shape `B...`; a shared
   iteration counter may be a scalar tensor. There are no Python floats,
   tensor-valued fields hidden on residual objects, or per-element Python
   containers.
2. Per-element `status`, convergence/KKT information, and implicit-gradient
   eligibility have shape `B...`; M6's actual backward-validity result uses
   the same shape. Active/free-coordinate masks and projected optimality data
   have shape `B... + (nt,)`, where `nt` is the reduced total tangent
   dimension. Robust weights retain the residual's fixed semantic group/row
   structure under the same leading batch.
3. The returned final `Values` and state must identify one consistent terminal
   point. Its residual, robust weights, Jacobian blocks, active-bound mask, and
   optimality/KKT residual are either stored as detached tensors for that point
   or deterministically recomputable by a canonical final evaluation. State
   must never label a candidate residual or previous-iterate Jacobian as final.
4. `Problem` exposes a stable, named tensor pytree of external parameters and
   their differentiability declarations. Provider dependencies preserve that
   lineage. Static configuration and optimized blocks are separately
   enumerable, so implicit backward never guesses roles from object identity.
5. Broadcasted/shared external tensors remain shared inputs. Their
   per-element VJP contributions reduce back to the original input shape;
   state must not replace a shared parameter with an expanded, independently
   differentiable copy.
6. Evaluation caches are evaluation-local. Default `run` retains only
   detached artifacts and warm-start state, while the pure evaluation/update
   substrate must not hard-code a detach that prevents the explicit unrolled
   oracle or the M6 final-point recomputation.

Branching, mutation, `.detach()`, and custom operators are reviewed against
this contract. Every new Warp kernel records its differentiable inputs and
adjoint strategy next to its parity tests.

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
