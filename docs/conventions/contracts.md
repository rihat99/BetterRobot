# Contracts

This page draws the line between a caller's responsibility and the library's.
Public boundaries check tensor structure once and raise a useful error. The
math below those boundaries trusts the checked inputs.

## Tensor shapes and batching

A supported batched tensor uses `(B..., event)`:

- `B...` is any number of leading batch axes;
- the trailing event axes describe one robot quantity.

Common events are `(nq,)` for a configuration, `(nv,)` for a tangent or
generalized force, `(7,)` for a pose, and `(6, nv)` for a spatial
Jacobian.

When several inputs participate in a call, their leading axes follow
right-aligned PyTorch broadcasting. The return keeps the resolved batch.
A size-one batch remains a batch; BetterRobot does not silently squeeze it.
See {doc}`/concepts/the_compute_seam` for the execution model.

## Devices and dtypes

- `Model.to(device, dtype)` returns a moved copy.
- Query tensors and model values must share a device and floating dtype.
- The validated kinematics and dynamics paths support `torch.float32` and
  `torch.float64`.
- Those paths reject `torch.float16` and `torch.bfloat16`.
- A public operation does not silently move or cast one side to match the
  other.

Supported floating results preserve the working input dtype. If an algorithm
ever needs a wider internal reduction, that exception must be documented and
tested.

BetterRobot does not change PyTorch's process-wide TF32 or deterministic
settings. A caller that enables them owns the accuracy and reproducibility
check for that workload.

## Pose and tangent storage

Every quaternion is scalar-last:

```text
rotation     [qx, qy, qz, qw]
SE(3) pose   [tx, ty, tz, qx, qy, qz, qw]
twist        [vx, vy, vz, wx, wy, wz]
```

Angles are radians and public physical quantities use SI units.

Input quaternions are expected to have unit norm. FK does not normalize them
by default. For free-flyer diagnostics,
`forward_kinematics(..., check_quaternion_norm=True)` checks the norm and
raises `QuaternionNormError` outside `[0.9, 1.1]`. The optional check can
synchronize accelerator work, so it is not part of the normal hot path.

A quaternion and its negation represent the same rotation. Rotation tests and
residuals compare relative rotations, not raw quaternion components. The
SO(3) logarithm uses a principal branch. At an angle of exactly pi the chosen
axis has an unavoidable sign ambiguity, so the tangent derivative is not
guaranteed there.

## Joint limits

FK accepts configurations outside the model's position limits. It computes
the pose represented by the input; it is not a feasibility checker.

`solve_ik` gives its robot variable hard position bounds.
`IKCostConfig.limit_weight` is a tuning scale for an additional soft
residual, not those hard bounds; the task squares it into the residual's outer
L2 coefficient. Continuous revolute joints use infinite limits. Quaternion
coordinates live on a manifold and do not have meaningful rectangular bounds;
use an orientation residual to constrain them.

## Model consistency

`build_model` validates topology once. Among other checks:

- joint zero is the universe and has parent `-1`;
- each later parent precedes its child;
- joint widths add up to `nq` and `nv`;
- public and full-coordinate slice tables are consistent; and
- structure tables agree with model-value table sizes.

Malformed input raises `ModelInconsistencyError` while building the model,
not during every query.

## State ownership

| Object | Ownership rule |
|---|---|
| `ModelStructure` | immutable topology; safe to share |
| `ModelValues` | treat tensor storage as read-only after attachment |
| `Model` | shallowly frozen; contained tensors and metadata are not deeply frozen |
| `Data` | mutable cache for one evaluation; do not share across concurrent evaluations |
| solver state and histories | belong to one solve; carry the returned state forward |
| `IKResult` and `Trajectory` | caller-owned dataclasses containing tensors |

`Data.require(level)` raises `StaleCacheError` when a caller asks for a
result that has not been computed. Run the required kinematic pass and keep
the returned `Data`.

BetterRobot never changes `torch.set_num_threads`. CUDA work uses the caller's
current PyTorch stream. The optional Warp FK and RNEA paths bridge that stream
rather than choosing a separate default stream.

For accelerator multiprocessing, use `spawn` and initialize each child
independently. Sharing mutable `Data`, solver objects, streams, or captured
graphs between processes is unsupported.

## Numerical behavior

BetterRobot promises results within the routine- and dtype-specific
tolerances exercised by its tests. It does not promise bitwise equality
across devices, compiler choices, dependency versions, or CUDA streams.

Small-angle Lie formulas use stable series branches. Dense Cholesky reports a
failed factorization to the solver instead of hiding a rank-deficient fallback.
LM can then increase damping for the affected batch element.

Regression and parity evidence records its model, dtype, device, inputs, and
tolerances. One tight fp64 FK test is not a universal error bound for every
robot or chain length.

## Residual objective algebra

### Optimization graph and static inputs

`Problem` harvests residual dependencies transitively. A `Node` may read
Variables and child Nodes. `node.nodes` contains its direct children;
`node.variables` contains all leaf Variables in stable first-seen order,
deduplicated by identity. Freeze rejects a node cycle, applies identity-only
node merging at every depth, and registers every discovered node in the same
evaluation scope. A node memo is valid only within that scope.

Inputs that change between evaluations must be explicit named
`Variable(..., trainable=False)` objects and be replaced through
`Problem.update()`. Static Variables may hold floating, boolean, or integer
tensors; trainable Variables must use float32 or float64. Every Variable in a
problem shares one device, and all floating Variables share one dtype. Static
bool or integer data never enters tangent or retraction paths.

A bare tensor accepted by a residual or node constructor is a construction-time
constant. It is not auto-wrapped, harvested, or addressable by
`Problem.update()`. Mutability is never inferred from `requires_grad`.

### Vector residuals

For every residual, the objective contract is:

```text
rows = row_weight.apply(error())
cost = Σ_k active_k · w_k · ρ(‖rows_k‖²) · norm
```

The symbols have these exact meanings:

- `row_weight` is square-root-information whitening. A scalar, a compatible
  tensor, `ScaleWeight`, or `DiagonalWeight` applies identically to error and
  Jacobian rows.
- `weight` supplies the non-negative outer coefficient `w_k`; it is not a row
  multiplier and never changes the kernel argument. A tensor coefficient must
  have shape `()`, the exact trainable batch shape, or
  `(*batch_shape, n_groups)` and must preserve working dtype and device.
  Negative Python values raise at assignment. Tensor non-negativity is the
  caller's domain responsibility and is not scanned in the hot path.
- `active_k` comes from the detached boolean tensor returned by
  `active_groups()`, with exact shape `(*batch_shape, n_groups)`. It is
  authoritative even when `ρ(0)` is nonzero. Built-in gated residuals also
  return finite zero rows for inactive groups, but zero rows alone do not
  declare inactivity.
- `norm` is `1` for `reduce="sum"`, `1 / n_groups` for `"mean"`, and
  `1 / clamp(Σ_k active_k, 1)` for `"mean_active"`. The active count is
  detached per batch element.

The L2 kernel is `ρ(s) = 0.5 · s`, so its exact objective contribution is
`0.5 · Σ_k active_k · w_k · ‖rows_k‖² · norm`.
`enabled=False` and a Python-zero outer weight skip evaluation but retain the
fixed row layout and optimizer state. A tensor zero stays graph-visible.

`Problem.error()` returns concatenated `row_weight`-whitened rows only; outer
weights, reduction, group activity, and robust-kernel scaling are excluded.
The `residual` fields of `IKResult`, `TrajOptResult`, and
`ContactForceResult` have the same diagnostic meaning. Use
`Problem.objective()` for total cost and `Problem.term_costs()` for named term
contributions.

LM and GN use uncorrected IRLS with group row scale
`sqrt(active_k · w_k · norm · kernel.weight(‖rows_k‖²))`. There is no
Triggs second-order correction. This approximation is gradient-consistent
only on a fixed active set. Activity thresholds are non-differentiable because
the mask and `mean_active` count are detached.

### Scalar costs

`ScalarCost(fn, *reads, weight=w)` requires `fn` to return one batch-shaped
scalar `f` on the caller-guaranteed domain `f >= 0`. Reads may be Variables or
Nodes. Its one row is `sqrt(2f)` for `f > 0` and exactly zero for `f <= 0`,
using masked branches that avoid a NaN backward. With L2, the on-domain
objective contribution is exactly `w * f`.

At exactly zero, that row has zero gradient. For positive values approaching
zero, the Gauss--Newton column `grad(f) / sqrt(2f)` may grow; damping covers the
tail. A problem containing `ScalarCost` is implicit-ineligible.

## Automatic differentiation

Differentiability is a property of a path, not of the package name.

- Torch Lie functions, FK, and the main rigid-body passes have focused
  first-order tests.
- `Problem` can form analytic, `jacrev`, `jacfwd`, or explicit
  finite-difference Jacobians.
- A custom residual or provider preserves gradients only if its implementation
  does so and its tests prove that claim.
- Second derivatives are promised only for paths with a focused
  `gradgradcheck`.

Task helpers return detached results by default. `solve_ik` can opt into its
guarded implicit path with `differentiable=True`. The generic
`LevenbergMarquardt.run` and `GaussNewton.run` loops remain detached.

For a small eligible problem,
`solve(..., differentiate="implicit")` attaches a guarded first-order
backward at the final optimum. It differentiates with respect to declared
external tensor parameters, not the initial guess or solver settings. Invalid
terminal states, unstable active bounds, nonsmooth robust points, quaternion
branch cuts, oversize systems, and singular backward systems raise
`ImplicitDifferentiationError` instead of returning a guessed gradient.

A problem is implicit-ineligible if any residual uses
`reduce="mean_active"`, overrides `active_groups()`, or is a `ScalarCost`. The
implicit entry point raises an actionable error naming that residual.

Optimized values, external tensor parameters, and static configuration are
three different roles. A tensor has one role in a solve. The API never infers
that role from `requires_grad` alone.

## Compilation and capture

Callers may compile compatible raw functions with `torch.compile`.
Specialization belongs to PyTorch and may depend on shape, dtype, device,
layout, static topology, and compiler options. BetterRobot does not install a
package-wide graph cache or dynamic-shape policy.

No public captured solver driver ships. Experimental CUDA graph use must own
stable storage, warmup, invalidation, replay parity, and both forward and
backward work.

## Serialization

There is no stable `Model.state_dict` or versioned checkpoint format yet.
Persist the source robot description and any explicit tensor parameters.

Do not advertise `torch.save(model)` as a portable or trusted checkpoint.
`Model.meta` may contain builder objects, asset resolvers, callables, and
environment-specific paths. Evaluation caches, compiled programs, streams,
and solver histories are not model state.

## Exceptions

| Exception | Meaning |
|---|---|
| `ModelInconsistencyError` | parsed or built topology is inconsistent |
| `DeviceMismatchError` | tensors that must interact are on different devices |
| `DtypeMismatchError` | a tensor has an unsupported or mismatched dtype |
| `QuaternionNormError` | the optional free-flyer norm check failed |
| `ShapeError` | a tensor has the wrong event shape or incompatible batch |
| `StaleCacheError` | requested `Data` was not computed to the required level |
| `BackendNotAvailableError` | an optional integration dependency is unavailable |
| `ConvergenceError` | available for callers that promote a returned failure status |
| `UnsupportedJointError` | reserved compatibility error for an unsupported joint |

`SingularityWarning` is a warning type, not an exception, and is not emitted
automatically today. Public code may use `TypeError` or `ValueError` for a
boundary that has no more specific BetterRobot exception.

Error messages name the input, the rule, and the received value. Internal
impossibilities use assertions; user mistakes do not.

## API stability

Before 1.0, minor releases may change the public surface. Release notes must
state replacements plainly. The pose layout, tensor batch convention, and
current public behavior remain deliberate contracts even during that period.

After 1.0, these surfaces are intended to be stable:

- the root public API and documented qualified imports;
- `Model`, `Data`, and their public fields;
- the pose and tangent layouts;
- required members of public protocols; and
- the package dependency direction.

Collision computation and some viewer surfaces remain experimental. See
{doc}`packaging` for release rules.

## Caller checklist

1. Give `q` shape `(B..., nq)` and tangent quantities
   `(B..., nv)`.
2. Match the model's device and use fp32 or fp64.
3. Use scalar-last unit quaternions.
4. Treat model contents as read-only.
5. Keep mutable caches and solver state local to one evaluation.
6. Inspect returned convergence status instead of assuming every solve
   succeeded.
