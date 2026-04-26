# Define a custom residual

Every IK / trajopt term is a `Residual` subclass. The minimal contract:

```python
import torch
from better_robot.residuals.base import Residual, ResidualState

class FrameZAbove(Residual):
    """Penalise frames whose z-coordinate falls below a threshold."""
    def __init__(self, frame_id: int, *, z_min: float = 0.0, name: str = "z_above"):
        super().__init__(name=name)
        self.frame_id = frame_id
        self.z_min = z_min

    @property
    def dim(self) -> int:
        return 1

    def __call__(self, state: ResidualState) -> torch.Tensor:
        z = state.data.frame_pose_world[..., self.frame_id, 2]
        return torch.clamp(self.z_min - z, min=0.0).unsqueeze(-1)
```

Add it to a `CostStack`:

```python
from better_robot.costs.cost_stack import CostStack

stack = CostStack()
stack.add("z_floor", FrameZAbove(model.frame_id("panda_hand")), weight=10.0)
```

For Jacobians, override `.jacobian(state)`. If you don't, the AUTO
strategy will fall back to central finite differences through
`model.integrate` — correct but slower than analytic.
