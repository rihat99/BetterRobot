# Write a custom residual

A residual is a vector of errors that an optimizer tries to make small. A
position target might return three errors; a signed-distance term might return
one error per sampled point. BetterRobot squares and combines those rows for
least-squares optimization.

Subclass {py:class}`better_robot.optim.Residual`, keep direct references to the
variables or nodes it reads, and implement `error()`. The base constructor
declares:

- `name`, unique within one problem;
- `dim`, the fixed number of output rows; and
- the variables, weight, robust kernel, and robust `group_size`.

This residual turns negative signed distances into positive penetration
depths. The example also evaluates the exact class through a public
{py:class}`better_robot.optim.Problem`.

<!-- custom-residual-example:start -->
```{testcode}
import torch
from better_robot.optim import Problem, Residual, Variable
from better_robot.residuals import Node


class PenetrationResidual(Residual):
    """Positive depth for samples lying behind a surface."""

    def __init__(self, signed_distance: Node, *, time: int, points: int, weight=1.0) -> None:
        self.signed_distance = signed_distance
        self.nodes = (signed_distance,)
        super().__init__(
            *signed_distance.variables,
            dim=time * points,
            name="penetration",
            weight=weight,
        )

    def error(self):
        signed = self.signed_distance.value()["signed_distance"]
        penetration = torch.relu(-signed)
        return penetration.reshape(*signed.shape[:-2], self.dim)


class SignedDistance(Node):
    def __init__(self, value):
        self.distance = Variable(value, name="signed_distance", trainable=False)
        super().__init__(self.distance)

    def compute(self):
        return {"signed_distance": self.distance.tensor}


signed_distance = torch.tensor([[-0.2, 0.1], [-0.4, 0.3]], dtype=torch.float64)
residual = PenetrationResidual(SignedDistance(signed_distance), time=2, points=2)
problem = Problem([residual])
rows = problem.error()

print(rows.tolist())
```
<!-- custom-residual-example:end -->

```{testoutput}
[0.2, 0.0, 0.4, 0.0]
```

The example uses a static variable because it only evaluates rows. In an
optimization, the node would reference trainable variables instead; the
problem discovers those references through the residual's object graph.

## Shared calculations

If several residuals need the same expensive calculation, put that work in a
{py:class}`better_robot.residuals.Node`. BetterRobot computes a referenced node
once during one problem evaluation and shares the result. Kinematics is the
common example: several frame errors can reuse one FK pass.

Do not cache a computed tensor on the residual object. A cached tensor can
belong to an old input or an old autograd graph. `Node.value()` owns the
evaluation-epoch memo and invalidates it between graph evaluations.

## Jacobians and robust groups

PyTorch automatic differentiation is the default. Override `jacobian()` only
when you have a complete, tested analytic formula. Return blocks in the same
order as the residual's trainable variable references, and compare them with
`jacrev` or `jacfwd` before relying on them.

`Residual.group_size` says which consecutive rows form one robust group.
Scalar penetration depths use the default `1`. A flattened list of 3D point
errors can pass `group_size=3` to `Residual.__init__`.

Variable-size observations still need a fixed output shape. Pad to a declared
maximum and return finite zero rows for missing observations. Reserve NaN rows
for a genuinely invalid batch element, not ordinary padding.
