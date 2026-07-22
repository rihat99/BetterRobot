# `residuals/` — Object-Referenced Error Terms

## Residual protocol

A residual subclasses `Residual`, receives every variable or shared node it
reads at construction, and returns raw `(..., dim)` rows from `error()`.
`Residual` owns the stable name, fixed positive dimension, outer `weight`,
square-root-information `row_weight`, reduction, activity, robust kernel,
group size, and ordered variable references. Missing analytic blocks use the
`Problem` Jacobian strategy; finite differences remain an explicit debug
choice.

`jacobian()` returns a tuple of complete full-tangent blocks ordered like the
residual's trainable variable dependencies. Optimizer plumbing centrally
gathers any construction-time free subset. A residual with shared work lists
its direct `Node` objects in `nodes`. A node may read Variables and other
nodes; `node.nodes` holds its direct children and `node.variables` is the
order-stable, identity-deduplicated tuple of all transitive leaf Variables.
`compute()` may call `child.value()`. `Problem` walks and evaluation-scopes the
whole node DAG, rejects cycles, and applies identity-only `merge_key` sharing
to nested nodes too. Graph-bearing results are memoized for one evaluation
scope only. `RobotState` is the standard lazy FK node.

Inputs that may change between evaluations are named
`Variable(..., trainable=False)` objects and are replaced with
`Problem.update()`. Static Variables may use bool or integer dtypes; trainable
Variables remain float32/float64. All Variables in a problem share one device,
and all floating Variables share one dtype. A bare tensor accepted by a
constructor is a construction-time constant: it is not harvested, auto-wrapped,
or addressable by `Problem.update()`. Constructors that accept either form
must document this distinction.

## Objective algebra

For each residual, `Problem` applies exactly this formula:

```text
rows = row_weight.apply(error())
cost = Σ_k active_k · w_k · ρ(‖rows_k‖²) · norm
```

`weight` supplies the non-negative outer coefficient `w_k`; it never changes
the kernel argument. `row_weight` alone whitens errors and Jacobian rows.
`norm` is `1` for `reduce="sum"`, `1 / n_groups` for `"mean"`, and
`1 / clamp(Σ_k active_k, 1)` for `"mean_active"`; the active count and mask
are detached. With L2, `ρ(s) = 0.5 · s`, so the term is
`0.5 · Σ_k active_k · w_k · ‖rows_k‖² · norm`.

`active_groups()` may return the authoritative boolean group mask. It must be
fixed-shape; shipped gated residuals also return zero rows for inactive groups
as an authoring convention. `enabled=False` and a Python-zero outer weight
skip a term without changing its reserved layout; tensor zero remains in the
graph.

`Problem.error()` exposes the concatenated whitened rows, not objective-scaled
rows. Outer coefficients, reduction, per-group activity, and robust-kernel
scaling belong to `objective()` and `term_costs()`. Task-result `residual`
fields reuse the whitened-row meaning.

`reduce="mean_active"` and any `active_groups()` override are
implicit-ineligible because their detached state-dependent activity has no
consistent implicit derivative.

## Scalar costs

`ScalarCost(fn, *reads, weight=..., name=...)` adapts a non-negative scalar
penalty `f` into one safe `sqrt(2f)` residual row, so the default L2 objective
contribution is exactly `weight * f`. Reads may be Variables or Nodes. The
callable contract is `f >= 0`; no hot-path value scan enforces it. At exactly
zero the masked row has zero gradient, while the Gauss--Newton column
`grad(f) / sqrt(2f)` grows as positive `f` approaches zero. Damping covers the
tail, but authors should use this class for penalties whose vanishing means
convergence or whose values stay away from zero. Any `ScalarCost` makes a
problem implicit-ineligible.

## Temporal structure

`TemporalPattern` is the optimizer-independent declaration for a residual over
one `Variable(..., time_axis=0)`. For row group `r`, offset `o` identifies knot
`r + row_origin + o`; `rows * row_width` must equal the residual's static
`dim`. Offsets are sorted, unique, non-empty integers and every referenced
knot must lie inside the declared horizon.

A structured residual implements both optional hooks:

- `temporal_structure(variable) -> TemporalPattern | None` declares support
  without inspecting tensor values;
- `temporal_jacobian_blocks(variable)` returns `offset -> Tensor` with shape
  `(B..., rows, row_width, tangent_width_per_knot)`.

Numeric blocks contain only the raw residual derivative. `Problem` applies
`row_weight` to those blocks and applies outer, activity, reduction, and
robust IRLS scaling once. A declaration without numeric blocks is not
direct-banded eligible. Missing declarations make automatic LM use dense
assembly; forced structured routing fails actionably. A zero weight never
grants eligibility.

Velocity and higher-order smoothness may declare constant temporal blocks
only when every moving joint has an affine scalar difference. Spherical,
free-flyer, and other manifold logarithms have state-dependent derivatives;
those models must warn under automatic differentiation and route dense until
exact log-Jacobian blocks have their own reviewed contract.

## Vision and point-cloud pack

- `projection.py`: camera-thin `ProjectionResidual` over model frame-table
  rows. The complete analytic block is projection derivative × camera rotation
  × frame Jacobian. `PointProjectionResidual` applies the same camera
  convention to Variable-, Node-, or tensor-valued point sets. Its explicit
  `time_axis` keeps trajectory `T` and point `P` as event axes, confidence is a
  per-group outer coefficient, and visibility is detached group activity.
  Observation tensors must already match the evaluated dtype/device; no hidden
  hot-path transfers.
- `chamfer.py`: `MaskedChamferResidual` on fixed padded point clouds and bool
  masks. Clouds, masks, and confidence may come from Variables or Nodes.
  Nearest indices and confidence are detached while selected distances remain
  differentiable; safe square-root row scaling makes confidence linear in the
  L2 objective.
- `scene_sdf.py`: `SceneSDFState` performs one detached nearest-neighbour pass
  and feeds penetration, attraction, and clearance residuals. The shared result
  carries signed distance, nearest distance, confidence, and validity and may
  use point or point-to-plane distance. Each penalty owns its detached trust
  gates and exact-zero inactive rows because the three heads have different
  active sets; confidence remains shared geometry quality and enters linearly.
- `_point_cloud.py`: private chunked nearest-correspondence helpers. The public
  ragged convention is `(padded_tensor, validity_mask)`; invalid rows are finite
  zeros.

Robust kernels live on each `Residual`. Use `group_size=2` for per-point
projection kernels such as Geman–McClure. LM/GN use uncorrected IRLS, with row
scale `sqrt(active_k · w_k · norm · kernel.weight(s_k))`; there is no
Triggs second-order correction. This linearization is gradient-consistent on
a fixed active set. Activity thresholds are deliberately non-differentiable.

## Existing library

- `pose.py`: pose, position, and orientation targets using `RobotState`.
- `limits.py`, `human.py`: joint and swing/twist limits.
- `regularization.py`: rest, spherical-joint prior, and reference-trajectory
  terms.
- `smoothness.py`, `temporal.py`: trajectory differences and time indexing.
- `contact.py`: contact-consistency residual.
- Collision residuals are not exported; the separate collision package remains
  owner-gated and must not be partially implemented without reviewed scope.

## Author checklist

1. Pass every dependency at construction and give the residual a stable name
   and fixed dimension. Use explicit non-trainable Variables for values that
   must change after construction; never add setter or auto-wrapping APIs.
2. Preserve arbitrary leading execution batches and return raw residual rows.
3. When a constructor exposes `weight`, `row_weight`, `reduce`, or `enabled`,
   forward that control unchanged to `Residual`.
4. Require compatible input dtype/device instead of coercing during an
   evaluation.
5. Make masks fixed-shape and explicit; never infer structure from tensor
   values.
6. If analytic blocks are advertised, provide one for every trainable
   dependency and test them against both AD directions.
7. Record detached choices such as nearest-neighbour indices in the docstring
   and test that gradients reach only selected continuous values.
8. Put shared FK, dynamics, or NN work in a composable `Node`; list only direct
   child nodes and let the base collect leaf Variables. Add a counting test for
   one computation per node per evaluation scope.
9. For time-local support, test dense blocks, JVP, VJP, normal bands, arbitrary
   leading batches, and short-horizon constructor failures. Never infer
   support from numerical zeros.
