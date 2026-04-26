# Batch IK on GPU

Every BetterRobot tensor carries a leading batch dimension. To run a
batch of IK problems on CUDA:

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64, device="cuda")

# Batched targets: 64 different end-effector poses.
targets = torch.zeros(64, 7, dtype=torch.float64, device="cuda")
targets[..., 6] = 1.0  # qw = 1
targets[..., :3] = torch.rand(64, 3, dtype=torch.float64, device="cuda") * 0.4 + 0.3

result = br.solve_ik(model, {"panda_hand": targets})
print(result.q.shape)  # (64, nq)
```

Performance notes: the LM solver runs a single Cholesky per iteration
on the batched normal equations, which is the GPU-friendliest path.
For very large batches, profile against the `tests/bench/` baselines.
