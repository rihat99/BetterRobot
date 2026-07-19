# Write a custom residual

A residual is a vector of errors that an optimizer tries to make small. A
position target might return three errors; a signed-distance term might return
one error per sampled point. BetterRobot squares and combines those rows for
least-squares optimization.

You do not need to subclass a library base class. A residual object declares:

- `name`, unique within one problem;
- `reads`, the context entries it uses;
- `dim`, the fixed number of output rows; and
- `__call__(ctx)`, which returns a tensor shaped `(*batch, dim)`.

This residual turns negative signed distances into positive penetration
depths. The example also evaluates the exact class through a public
{py:class}`better_robot.optim.Problem`.

<!-- custom-residual-example:start -->
```{testcode}
import torch
from better_robot.optim import Problem, ResidualItem, VarSpec


class PenetrationResidual:
    """Positive depth for samples lying behind a surface."""

    name = "penetration"
    reads = ("signed_distance",)

    def __init__(self, time: int, points: int) -> None:
        self.dim = time * points

    def __call__(self, ctx):
        signed = ctx["signed_distance"]
        penetration = torch.relu(-signed)
        return penetration.reshape(*signed.shape[:-2], self.dim)


signed_distance = torch.tensor([[-0.2, 0.1], [-0.4, 0.3]])
residual = PenetrationResidual(time=2, points=2)
problem = Problem(
    vars=(VarSpec("dummy", (1,)),),
    residuals=(ResidualItem("penetration", residual),),
    parameters={"signed_distance": signed_distance},
)
rows = problem.residual({"dummy": torch.zeros(1)})

torch.testing.assert_close(rows, torch.tensor([0.2, 0.0, 0.4, 0.0]))
```
<!-- custom-residual-example:end -->

The `dummy` variable only makes this tiny example a complete optimization
problem; a real residual normally reads an optimized variable or a provider
output. `reads` tells BetterRobot which variables can affect each Jacobian
block. It is a structural declaration, so keep it accurate even though the
context behaves like a mapping.

## Shared calculations

If several residuals need the same expensive calculation, put that work in a
provider. A provider declares its own `reads` plus one or more `outputs`.
BetterRobot computes a requested provider once during one problem evaluation
and shares the result. Kinematics is the common example: several frame errors
can reuse one FK pass.

Do not cache a provider result on the residual object. A cached tensor can
belong to an old input or an old autograd graph. Let the evaluation context own
that short-lived cache.

## Jacobians and robust groups

PyTorch automatic differentiation is the default. Add
`jacobian_blocks(ctx)` only when you have a complete, tested analytic formula.
Compare every analytic block with `jacrev` or `jacfwd` before relying on it.

`ResidualItem.group_size` says which consecutive rows form one robust group.
Scalar penetration depths use the default `1`. A flattened list of 3D point
errors can use `group_size=3`. Non-default groups require direct
`ResidualItem(..., group_size=3)` construction; `Problem.add_residual` is the
short path for the default scalar grouping.

Variable-size observations still need a fixed output shape. Pad to a declared
maximum and return finite zero rows for missing observations. Reserve NaN rows
for a genuinely invalid batch element, not ordinary padding.
