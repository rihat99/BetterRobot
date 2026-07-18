# Extension points

BetterRobot grows through small, explicit interfaces. Most extensions are
ordinary objects passed to the code that uses them. Parser suffixes are the
one public feature discovered through registration.

If an interface is not described here, do not assume that an internal
dictionary is a public plugin API.

## Quick map

| Goal | Public extension point |
|---|---|
| add an objective | residual callable, usually wrapped by `ResidualItem` |
| add a robot joint | `JointModel` protocol and `ModelBuilder` |
| add a first-order optimizer | a `torch.optim.Optimizer` factory |
| add a least-squares solver | call a solver object directly |
| add a linear solver | `LinearSolver` protocol |
| add a robust loss | `RobustKernel` protocol |
| add a file format | `register_parser(suffix, function)` |
| add a viewer mode | `RenderMode` protocol and `Scene.add_mode` |
| add an asset source | `AssetResolver` protocol |
| add a fused compute implementation | internal branch at one complete pass |

## Residuals

Start with {doc}`/guides/custom_residual`. A residual declares:

| Member | Meaning |
|---|---|
| `name` | stable diagnostic name |
| `reads` | context entries needed by the calculation |
| `dim` | fixed number of output rows |
| `__call__(ctx)` | returns `(B..., dim)` |
| `jacobian_blocks(ctx)` | optional complete analytic blocks |

For example:

```{testcode}
import torch


class PointAtOrigin:
    name = "point_at_origin"
    reads = ("point",)
    dim = 3

    def __call__(self, ctx) -> torch.Tensor:
        return ctx["point"]
```

Add an instance explicitly with `Problem.add_residual` or wrap it in a
`ResidualItem`. The item owns its weight and robust kernel. There is no
process-wide residual registry.

A provider may compute shared context entries lazily. It declares `name`,
`reads`, `outputs`, and `__call__(ctx)`. Provider caches last for one
problem evaluation only.

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

## Nonlinear solvers

`LevenbergMarquardt` and `GaussNewton` expose the lifecycle used by direct
callers:

| Method | Purpose |
|---|---|
| `init_state(values, problem)` | validate the solve and create tensor state |
| `update(values, state, problem)` | perform one fixed-shape step |
| `finalize(values, state, problem)` | refresh diagnostics at the returned point |
| `run(values, problem, state=None)` | run the detached eager loop |
| `solve(..., differentiate=...)` | choose detached or eligible implicit differentiation |

A prior state can warm-start solver state. Task helpers do not discover custom
solver classes; call a custom solver directly.

For first-order methods, pass a standard optimizer factory to
`run_first_order`. BetterRobot owns tangent retraction and bounds while
PyTorch owns Adam, SGD, or another compatible update rule.

The generated {doc}`/reference/api/better_robot/better_robot.optim` page is
the exact signature reference. The least-squares reasoning and damping
intuition live in {doc}`/concepts/residuals_costs_and_solvers`.

## Linear solvers

A `LinearSolver` implements:

```{testcode}
import torch


def solve(A, b, ridge=None):
    return torch.linalg.solve(A, b)
```

`b` has shape `(B..., n)`; `ridge` is a scalar, `(B...,)`, or
`None`. A solver that supports only dense tensors can omit
`supported_systems`. A solver that accepts block-banded storage declares
`supported_systems = frozenset({"banded"})` or both supported forms.

The built-ins are dense `Cholesky` and `BandedCholesky`. Do not claim a
rank-deficient fallback unless the implementation and result diagnostics prove
it.

## Robust kernels

A `RobustKernel` implements `rho(s)` for the objective and `weight(s)`
for the iteratively reweighted normal equations. Both methods preserve the
input shape. BetterRobot uses `weight(s) = 2 * rho'(s)`.

Built-ins are `L2`, `Huber`, `Cauchy`, `Tukey`, and
`GemanMcClure`. Pass a kernel on the residual item; there is no global
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
- `BSplineTrajectory` is a Euclidean numerical utility. Robot trajectory
  optimization currently accepts `KnotTrajectory` because manifold
  interpolation and bound handling need a larger contract.
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
