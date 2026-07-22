# Run a staged fit

Use a staged fit when an early objective should establish a stable pose before
later observations and penalties become active. Keep phase policy in ordinary
application data, and use optimizer lifecycle methods to decide which numerical
state survives each boundary.

This example first places a floating root while its articulated joint is frozen.
Because a frozen tangent layout is immutable, that warm-up owns a separate
`RobotVariable` and `Problem`. The main fit then builds one unfrozen problem,
keeps one Adam optimizer and scheduler across two phases, and finishes with a
new L-BFGS optimizer.

```{testcode}
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim import (
    LevenbergMarquardt,
    Problem,
    RobotVariable,
    TorchOptimizer,
    Variable,
    residual,
)

builder = ModelBuilder("staged_fit")
base = builder.add_body("base")
tip = builder.add_body("tip")
builder.add_free_flyer_root("root", child=base)
builder.add_revolute_z("hinge", parent=base, child=tip)
model = build_model(builder.finalize())
neutral = model.q_neutral
final_target = torch.tensor([0.08, -0.04, 0.05, 0.02, -0.015, 0.01, 0.18])

# A changed frozen set is a changed layout, so root placement is its own solve.
root_q = RobotVariable(model, neutral.clone(), name="root_q", frozen_groups=("joints",))


@residual(root_q, dim=6, name="place_root")
def place_root(current):
    return model.difference(neutral, current)[..., :6] - final_target[:6]


LevenbergMarquardt(
    Problem([place_root]),
    max_iterations=30,
    tolerance=1e-6,
    jacobian_strategy="jacrev",
).optimize()
root_tangent = model.difference(neutral, root_q.tensor)

# Transfer the value once, then keep this graph and optimizer across Adam phases.
q = RobotVariable(model, root_q.tensor.detach().clone(), name="q")
observed = Variable(final_target.clone(), name="observed", trainable=False)


@residual(q, observed, dim=model.nv, name="data_fit")
def data_fit(current, target):
    return model.difference(neutral, current) - target


@residual(q, dim=model.nv, name="posture", weight=0.1)
def posture(current):
    return model.difference(neutral, current)


problem = Problem([data_fit, posture])
phases = [
    {
        "target": torch.tensor([0.06, -0.03, 0.04, 0.01, -0.01, 0.005, 0.08]),
        "data_weight": 1.0,
        "posture_enabled": True,
        "iterations": 30,
    },
    {
        "target": final_target,
        "data_weight": 2.0,
        "posture_enabled": False,
        "iterations": 45,
    },
]
adam = TorchOptimizer(
    problem,
    torch.optim.Adam,
    lr=0.05,
    tolerance=0.0,
    scheduler=lambda inner: torch.optim.lr_scheduler.ExponentialLR(inner, gamma=0.98),
)
logged = []
for index, phase in enumerate(phases):
    problem.update({"observed": phase["target"]})
    data_fit.weight = phase["data_weight"]
    posture.enabled = phase["posture_enabled"]
    adam.max_iterations = phase["iterations"]
    if index:
        adam.resume()
    adam.optimize()
    logged.append(problem.term_costs())

# A different algorithm owns different state, so polishing uses a new adapter.
polish = TorchOptimizer(
    problem,
    torch.optim.LBFGS,
    lr=0.8,
    max_iter=20,
    line_search_fn="strong_wolfe",
    max_iterations=3,
    tolerance=1e-6,
)
polish.optimize()
final_tangent = model.difference(neutral, q.tensor)

print("warm-up joint stayed fixed:", bool(root_tangent[-1] == 0.0))
print("logged terms:", sorted(logged[-1]))
print("final target reached:", bool(torch.allclose(final_tangent, final_target, atol=2e-4, rtol=0.0)))
```

```{testoutput}
warm-up joint stayed fixed: True
logged terms: ['data_fit', 'posture']
final target reached: True
```

`Problem.update()` changes named observations without changing this main
layout, so Adam moments and scheduler state survive. `resume()` makes terminal
elements runnable again and keeps cumulative iteration counts. Use `reset()`
instead when you intentionally want to discard optimizer state while retaining
the current variable values. LM also supports `resume()`, but it rebuilds
phase-specific damping and acceptance state because those values describe the
old objective.

BetterRobot does not ship a `Phase` object. Phase entry hooks, iteration budgets,
logging, and application-specific policy remain clearest as a plain loop like
the one above; the library supplies only the numerical lifecycle needed to make
that loop honest.
