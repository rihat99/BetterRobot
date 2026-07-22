# Extension points

BetterRobot grows through small, explicit interfaces. Most extensions are
ordinary objects passed to the code that uses them. Parser suffixes are the
one public feature discovered through registration.

If an interface is not described here, do not assume that an internal
dictionary is a public plugin API.

## Quick map

| Goal | Public extension point |
|---|---|
| add an objective | `Residual` subclass or `@residual` tensor function |
| add a robot joint | `JointModel` protocol and `ModelBuilder` |
| add shared residual work | `Node` subclass |
| add a first-order optimizer | `TorchOptimizer` with a `torch.optim.Optimizer` class or factory |
| add a nonlinear optimizer | `Optimizer` subclass owning one `Problem` |
| add a linear solver | `LinearSolver` protocol |
| add a robust loss | `RobustKernel` protocol |
| add a file format | `register_parser(suffix, function)` |
| add a viewer mode | `RenderMode` protocol and `Scene.add_mode` |
| add an asset source | `AssetResolver` protocol |
| add a fused compute implementation | internal branch at one complete pass |

## Residuals

Start with {doc}`/guides/custom_residual`. A residual subclasses `Residual`
and declares:

| Member | Meaning |
|---|---|
| `name` | stable diagnostic name |
| `variables` | ordered object references read by the residual |
| `dim` | fixed number of output rows |
| `row_weight` | square-root-information scaling for error and Jacobian rows |
| `weight`, `reduce` | non-negative outer importance and group reduction |
| `kernel`, `group_size` | robust loss and contiguous row grouping |
| `enabled` | whole-term activity without a layout change |
| `error()` | returns raw `(B..., dim)` rows |
| `jacobian()` | optional complete tangent blocks |
| `active_groups()` | optional authoritative boolean `(B..., n_groups)` mask |

For example:

```{testcode}
import torch
from better_robot.optim import Problem, Residual, Variable


class PointAtOrigin(Residual):
    def __init__(self, point):
        self.point = point
        super().__init__(point, dim=3, name="point_at_origin")

    def error(self) -> torch.Tensor:
        return self.point.tensor


point = Variable(torch.ones(3), name="point")
problem = Problem([PointAtOrigin(point)])
print(problem.error().tolist())
```

```{testoutput}
[1.0, 1.0, 1.0]
```

Pass residual instances to `Problem`; it harvests their variable and node
references. The `@residual(variable, ..., dim=...)` adapter is the concise
choice when automatic differentiation is sufficient. There is no process-wide
residual registry.

Use an explicit named `Variable(..., trainable=False)` for any target, mask,
label, or observation that will be replaced through `Problem.update()`. A bare
tensor accepted by a constructor is a construction-time constant and is not
part of the harvested graph.

A custom constructor that exposes `weight`, `row_weight`, `reduce`, or
`enabled` forwards that control to `Residual`. Do not apply outer importance
inside `error()` or an analytic `jacobian()`; `Problem` applies it once after
row whitening and robust grouping. `Problem.error()` returns the whitened
rows, while `objective()` and `term_costs()` expose costs.

Built-in residual constructors expose `row_weight` alongside outer `weight`.
Where a built-in does not expose `reduce` or `enabled` as a keyword, set the
inherited field after construction.

## Shared nodes

A `Node` computes shared work lazily from Variables and child Nodes supplied to
its constructor. `node.nodes` is the tuple of direct children;
`node.variables` is the order-stable, identity-deduplicated tuple of transitive
leaf Variables. Implement `compute()` and call `child.value()` for child
outputs. Optionally provide an identity-only `merge_key` when equivalent
instances may share one memo.

Residuals list their direct nodes in `nodes` and call `node.value()`.
`Problem` walks the complete acyclic graph, applies merge keys at every depth,
and scopes every discovered node. Each node computes once per evaluation, and
no graph-bearing value leaks to another candidate. `RobotState` is the built-in
FK example.

## Joint models

A custom joint implements `JointModel` structurally; inheritance is not
required.

| Member | Contract |
|---|---|
| `kind`, `nq`, `nv`, `axis` | static joint description |
| `joint_transform(q)` | `(B..., nq) -> (B..., 7)` |
| `joint_motion_subspace(q)` | `(B..., nq) -> (B..., 6, nv)` |
| `joint_velocity(q, v)` | joint-frame spatial velocity |
| `integrate(q, v)` | manifold retraction |
| `difference(q0, q1)` | local tangent from `q0` to `q1` |
| `random_configuration(...)` | one valid configuration |
| `neutral()` | identity or default configuration |

Optional dynamics hooks can provide joint bias acceleration or the derivative
of the motion subspace. When absent, current built-in dispatch uses zero,
which is correct only for joints with the corresponding constant behavior.

There is no joint registry. Pass the custom joint object to a programmatic
`ModelBuilder`; the built model keeps that exact object. The exported
`JointKind` literal lists built-ins and does not become open merely because
runtime construction accepts a custom string.

## Nonlinear optimizers

Every `Optimizer` owns a `Problem` and exposes this public lifecycle:

| Method | Purpose |
|---|---|
| `step()` | update referenced trainable variables once and return `OptimizerInfo` |
| `optimize()` | run the eager loop and return final public diagnostics |
| `resume()` | return terminal elements to RUNNING while keeping compatible algorithm state and cumulative counts |
| `reset()` | clear optimizer state while retaining variable values |

Subclass `Optimizer` for a genuinely different nonlinear driver and implement
`step`, `resume`, `reset`, and the initial-info hook. Solved values remain on variables;
the shared `OptimizerInfo` contains status, iterations, cost, and derived
convergence. Task helpers accept custom optimizer factories only where their
documented signature says so; otherwise call the custom optimizer directly.

For first-order methods, construct `TorchOptimizer(problem, optimizer_cls,
scheduler=..., **kwargs)`. BetterRobot owns tangent retraction and bounds while
PyTorch owns Adam, SGD, or another compatible update rule. The optional
scheduler is a factory from that inner optimizer to a no-argument-step
`torch.optim.lr_scheduler.LRScheduler`; it advances once only when a public
step performs an optimizer update.

The generated {doc}`/reference/api/better_robot/better_robot.optim` page is
the exact signature reference. The least-squares reasoning and damping
intuition live in {doc}`/concepts/residuals_costs_and_solvers`.

## Linear solvers

A `LinearSolver` implements a `solve` method:

```{testcode}
import torch


class DenseSolver:
    supported_systems = frozenset({"dense"})

    def solve(self, A, b, ridge=None):
        matrix = A.clone()
        if ridge is not None:
            value = torch.as_tensor(ridge, dtype=A.dtype, device=A.device)
            matrix.diagonal(dim1=-2, dim2=-1).add_(value[..., None])
        return torch.linalg.solve(matrix, b)


A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
b = torch.tensor([9.0, 8.0])
print(DenseSolver().solve(A, b).tolist())
```

```{testoutput}
[2.0, 3.0]
```

`b` has shape `(B..., n)`; `ridge` is a scalar, `(B...,)`, or
`None`. A solver that supports only dense tensors can omit
`supported_systems`. A solver that accepts block-banded storage declares
`supported_systems = frozenset({"banded"})` or both supported forms.

The built-ins are dense `Cholesky` and `BandedCholesky`. Do not
claim a rank-deficient fallback unless the implementation and result
diagnostics prove it.

## Robust kernels

A `RobustKernel` implements `rho(s)` for the objective and `weight(s)`
for the iteratively reweighted normal equations. Both methods preserve the
input shape. BetterRobot uses `weight(s) = 2 * rho'(s)`.

The LM/GN approximation is uncorrected IRLS on a fixed active set; a kernel
must not bake the residual's outer coefficient or reduction into either
method.

Built-ins are `L2`, `Huber`, `Cauchy`, `Tukey`, and
`GemanMcClure`. Pass a kernel on the residual; there is no global
kernel selector.

## Parser formats

`register_parser(suffix, function)` is the only public registration API.
The suffix omits its leading dot and is matched case-insensitively after
`load` inspects the path.

The parser receives the source and returns an `IRModel`; `load` then calls
`build_model`. Registration is local to the process and takes effect after
the extension module is imported. A later registration for the same suffix
replaces the earlier callable.

See {doc}`/concepts/parsers_and_ir` for the intermediate model and
{doc}`/guides/load_a_robot` for normal loading.

## Viewer modes

A `RenderMode` provides `name`, `description`,
`is_available(model, data)`, `attach(context, model, data)`,
`update(data)`, `set_visible(visible)`, and `detach()`.

Attach an instance with `Scene.add_mode`. Third-party modes are not found
through the viewer's internal tables. See {doc}`/guides/visualize` for a
complete viewer example.

## Asset resolvers

An `AssetResolver` has one method:
`resolve(uri: str) -> pathlib.Path`. Missing assets raise
`FileNotFoundError`.

`load` does not accept a custom resolver directly. Call `parse_urdf` or
`parse_mjcf` with the resolver, then pass the returned intermediate model to
`build_model`. Shipped resolvers cover filesystems, packages, composites,
and cached downloads.

## Complete-pass compute implementations

This is an internal contribution point, not a public plugin protocol. Keep a
specialized implementation beside the Torch function that owns the complete
pass. The Torch version remains the default and numerical reference.

An opt-in implementation must:

1. accept ordinary Torch tensors plus `ModelStructure` and `ModelValues`;
2. reject or fall back for unsupported joints, devices, dtypes, and shapes;
3. match the Torch result on shared fixtures;
4. provide and test its gradient strategy;
5. preserve broadcasted execution batches; and
6. keep optional imports local.

Do not switch individual quaternion or spatial operations at runtime. The
reasoning is in {doc}`/concepts/the_compute_seam`.

## Features that are not extension points

- Collision distance and robot collision queries are not implemented. See
  {doc}`/reference/collision_and_geometry`.
- B-spline robot trajectory parameterization remains deferred because
  manifold interpolation and bound handling need a larger contract.
- No actuator or muscle protocol is shipped. External code can form actuator
  torque and pass it to dynamics explicitly.
- `Model`, `Data`, pose storage, and the package dependency direction are
  fixed foundations, not plugin interfaces.
- Internal dictionaries are not consumer APIs.

## Contribution checklist

1. Use an extension point listed above.
2. Test the happy path and at least one bad input.
3. Test `isinstance(value, Protocol)` when the public protocol is runtime
   checkable.
4. Document shapes, units, frames, and ownership.
5. Keep optional dependencies out of the core import path.
6. Update the generated reference through docstrings when a public signature
   changes.
