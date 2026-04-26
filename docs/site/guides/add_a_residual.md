# Add a residual to an existing IK problem

`solve_ik` builds a `CostStack` from `IKCostConfig`. To layer additional
residuals, drop down to the lower-level API:

```python
import better_robot as br
from better_robot.residuals.position import PositionResidual
from better_robot.costs.cost_stack import CostStack
from better_robot.optim import LeastSquaresProblem
from better_robot.optim.optimizers import LevenbergMarquardt

model = br.load("panda.urdf")
ee = model.frame_id("panda_hand")

stack = CostStack()
stack.add(
    "ee_pos",
    PositionResidual(frame_id=ee, target=torch.tensor([0.4, 0.0, 0.5])),
    weight=1.0,
)
# ... add more residuals here

problem = LeastSquaresProblem(
    cost_stack=stack,
    state_factory=...,        # see solve_ik for the boilerplate
    x0=model.q_neutral.clone(),
    lower=model.lower_pos_limit,
    upper=model.upper_pos_limit,
)
LevenbergMarquardt().minimize(problem, max_iter=200)
```

Custom residuals are written as `Residual` subclasses (see
`tutorials/04_custom_residual.md`).
