# Write a custom block residual

This guide is the consumer-author contract for the named-block optimization
API. A residual is structural: it does not inherit from a BetterRobot base
class and does not import `better_robot.optim`. It declares what it reads and
returns a tensor. The vertical-slice test extracts the worked residual below
directly from this file, so the example and its executable use cannot drift.

Start the consumer module with the public optimization imports used to assemble
the problem:

```python
import torch

from better_robot.optim import Problem, ResidualItem, VarSpec
```

The residual itself remains structural and does not depend on those classes.
`Problem`, `ResidualItem`, and `VarSpec` are used by the surrounding consumer
setup; the marked residual definition below intentionally uses only `torch`.

## 1. Declare names, dependencies, and a static dimension

Give the object three attributes:

- `name`: unique within a `Problem`;
- `reads`: a tuple of variable, external-parameter, or provider-output names;
- `dim`: the final residual dimension, fixed for every evaluation.

Implement `__call__(ctx)`. Read only the names declared in `reads`; the context
is read-only. The result shape is `(B..., dim)`. `B...` contains independent
batch axes. Event axes such as time, points, or joints must be reduced or
flattened into the final `dim` axis exactly as declared.

The slice uses a signed-distance provider with output shape
`(B..., time, points)`. This custom residual penalizes penetration:

<!-- custom-residual-example:start -->
```python
class PenetrationResidual:
    """Positive depth for samples lying behind the synthetic surface."""

    name = "penetration"
    reads = ("signed_distance",)

    def __init__(self, time: int, points: int) -> None:
        self.time = time
        self.points = points
        self.dim = time * points

    def __call__(self, ctx):
        signed = ctx["signed_distance"]
        penetration = torch.relu(-signed)
        return penetration.reshape(*signed.shape[:-2], self.dim)
```
<!-- custom-residual-example:end -->

The class deliberately needs only `torch` in its defining namespace. Register
an instance as `ResidualItem("penetration", PenetrationResidual(T, N))`.
The wrapper name and residual name must match; this catches accidental report
and weight-column mismatches at problem construction.

Do not infer or mutate `dim` during `__call__`. Pad a variable-size observation
set to a fixed maximum and carry a mask in the context instead.

## 2. Choose robust-kernel groups deliberately

`ResidualItem.group_size` defines consecutive semantic groups on the final
axis. Its default is `1`, matching scalar-row robust weighting. Set
`group_size=3` for a flattened sequence of 3D displacement vectors, for
example; `dim` must be divisible by the group size. One robust kernel is stored
per residual item and applies independently to each contiguous group.

Do not put unrelated units into one group. Pixel `x/y` error may be a 2-vector;
3D point error may be a 3-vector; a scalar hinge remains a scalar group. Split
terms with different kernels, scales, or meanings into separate residual
items. Direct `Problem.residual()` remains the weighted raw vector.
`Problem.objective()` sums grouped `kernel.rho(squared_norm)`, and
`Problem.gradient()` differentiates that same objective. Named-block LM/GN
applies `sqrt(kernel.weight(squared_norm))` to each group's residual and
Jacobian rows using the matching normalized `weight = 2*rho'` convention.

## 3. Add analytic blocks only when they are complete

Autodiff is the default fallback. An optional
`jacobian_blocks(ctx) -> dict[str, Tensor]` can provide selected variable
blocks. Keys are variable names, not provider-output names. Each value has
shape `(B..., dim, free_dim)` where `free_dim` is already reduced by the
problem's mask. Obtain the retained full-tangent indices structurally with
`ctx.free_indices("q")`; no optimizer import is needed.

Return no key for a structurally zero/unread variable. A supplied block is a
contract: if it raises or has the wrong shape, evaluation raises. BetterRobot
never catches an analytic error and silently substitutes autodiff. Test an
analytic block against both forced `jacrev` and forced `jacfwd` at fp32.

## 4. Request shared provider outputs

Put a provider output name in `reads`; do not call the expensive function from
the residual. Providers declare `inputs` and `outputs`, form an acyclic graph,
and run lazily at most once in one `EvaluationContext`. Three residual items
reading the same nearest-neighbour output therefore share one pass during a
residual, objective, gradient, or analytic-Jacobian context. AD-generated
Jacobian blocks use fresh transform-local contexts, so the provider may run
once for each missing `(residual, variable)` block. A zero Python weight keeps
the item inactive and does not trigger its providers.

Nothing may cache a context, provider result, graph tensor, tensor identity, or
mutation version across evaluations. A second `residual()` or `gradient()` call
creates a new context and recomputes requested outputs. A provider may detach a
discrete operation such as nearest-neighbour `argmin`; document that choice and
reconstruct any continuous quantity from graph-carrying inputs after the
detached selection, as the slice does for signed distance.

External differentiable tensors such as targets, tensor weights, and kernel
scales belong in `Problem(parameters={...},
differentiable_parameters=(...names...))`. The second argument is an explicit
future implicit-gradient declaration; BetterRobot never guesses it from
`requires_grad`. Hidden residual attributes are static configuration and are
not promised an implicit gradient.

## 5. Signal per-element invalidity without killing the batch

The v1 decision is **NaN rows**, not exceptions or an additional validity-mask
protocol. If one batch element cannot produce a meaningful residual, return
NaN for all of that element's affected rows while leaving other elements
intact. Do not raise from data-dependent validity checks: that would discard
valid neighbors in the batch. Named-block LM/GN detects non-finite rows before
factorization, marks only those elements `LMStatus.FAILED`, and records their
implicit-gradient eligibility as false.

NaN is a failure signal, not padding. Padded or missing observations that are a
valid no-op use finite zero rows plus an ordinary observation mask. A scalar
objective term follows the same per-element convention with a NaN scalar.

## 6. Decide whether capture eligibility matters

Every structural residual that satisfies the eager evaluation contract works
with the named-block solver. CUDA graph eligibility is narrower. A custom
residual and every provider it requests are capture-eligible only when their
forward/Jacobian work is:

- fixed-shape for a fixed `Problem` and `Values` layout, with padding and
  finite masks used for variable-size observation sets;
- sync-free: no `.item()`, `.cpu()`, `float(tensor)`, `bool(tensor)`,
  `int(tensor)`, or other tensor-dependent Python decisions;
- free of data-dependent indexing that changes output shape or storage
  identity; and
- allocation-bounded, with no Python-side cache keyed by a tensor object,
  pointer, mutation counter, or runtime value.

Static loops over declared residuals, providers, joints, or a padded maximum
are allowed. Tensor-dependent choices use fixed-shape operations such as
`torch.where`. The same rules apply to an analytic `jacobian_blocks` method;
when autodiff supplies a block, its transformed residual/provider path must
also satisfy them.

A non-eligible residual is still supported by the detached eager `run` loop;
it simply must not be placed in a future captured driver. The M2b update's
hot-path lint and a CPU `torch.compile(fullgraph=True)` smoke test catch common
graph breaks, but neither proves CUDA graph safety. M6 owns certification via
warmup plus actual capture/replay parity, including any custom-kernel adjoints.

## Verification checklist

Before shipping a custom residual:

1. assert unbatched, one-axis-batched, and multi-axis-batched output shapes;
2. compare batched values and tangent gradients with sequential evaluation;
3. count every shared provider (once per context, zero when inactive, and once
   per transformed block for AD-generated Jacobians);
4. compare analytic blocks with forced reverse and forward AD, if supplied;
5. test mask-reduced columns and robust `group_size` boundaries;
6. test one invalid batch element without raising or corrupting its neighbors;
7. if capture matters, run the fixed-shape fullgraph smoke and M6 replay-parity
   harness rather than treating eager success as certification.
