# Run on a GPU and in batches

You learn the two rules that make BetterRobot fast: batch axes lead, event axes trail; and the model and
your tensors share one device and dtype. You run a batched query on CPU and see the CUDA version.

## Batch axes lead, event axes trail

Every public tensor ends in its event shape: `q (..., nq)`, a pose `(..., 7)`, a transform `(..., 3, 4)`.
Everything before that is batch, and you may use any number of batch axes. Results keep them:

```python
import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot.kinematics import forward_kinematics_state, frame_jacobian

model = br.load(panda_description.URDF_PATH, dtype=torch.float32)
hand = model.frame_id("body_panda_hand")

q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
q_batch = q.expand(8, 16, model.nq)  # 8 robots x 16 time steps
state = forward_kinematics_state(model, q_batch)
print(state.frame_pose.shape)                            # (8, 16, nframes, 7)
print(frame_jacobian(model, state, frame="body_panda_hand").shape)  # (8, 16, 6, nv)
```

Batch axes follow PyTorch broadcasting. A `(5, 1, nq)` tensor stays `(5, 1, ...)`: nothing is squeezed.
Prefer one call on a big batch over a Python loop of small calls; each call has a fixed launch cost.

## Model and tensors share a device and dtype

`br.load` takes `device=` and `dtype=` (default `float32`). `Model.to` moves or casts a model. Public
functions check that `q` matches the model and raise instead of casting silently:

```python
q64 = torch.zeros(3, model.nq, dtype=torch.float64)
try:
    forward_kinematics_state(model, q64)
except Exception as error:
    print(type(error).__name__, error)

model64 = model.to(dtype=torch.float64)
print(forward_kinematics_state(model64, q64).frame_pose.dtype)
```

Use `float64` for tests, parity checks and tight IK tolerances; `float32` for throughput.

## Run the same code on CUDA

Only the device changes:

```python no-run
device = torch.device("cuda")
model = br.load(panda_description.URDF_PATH, device=device, dtype=torch.float32)
q_batch = model.q_neutral.expand(4096, model.nq).clone()
state = forward_kinematics_state(model, q_batch)  # runs on the GPU
torch.cuda.synchronize()
```

## What batching buys on a GPU

From `benchmarks/results/p2_world_dynamics_{cpu,cuda}.json`: median ms per forward call, Panda, `float32`.
CPU is an AMD EPYC 9654 on 1 thread; GPU is an NVIDIA RTX PRO 6000 Blackwell. The two files come from
different commits (`bbbb669`, `9f85980`), so treat them as orders of magnitude.

| Algorithm | CPU, batch 1 | CUDA, batch 1 | CPU, batch 4096 | CUDA, batch 4096 |
|---|---|---|---|---|
| inverse dynamics (RNEA) | 1.07 | 2.56 | 52.0 | 2.85 |
| mass matrix (CRBA) | 0.90 | 2.02 | 56.8 | 2.17 |
| forward dynamics (ABA) | 1.30 | 3.19 | 81.0 | 3.50 |

A single query is faster on the CPU. On the GPU the cost hardly grows from 1 to 4096 queries, so large
batches run 18–26x faster than on one CPU thread. [Benchmark and profile](../guides/benchmark-and-profile.md)
shows how to measure your own case.

Next: [visualize a robot](visualize.md).
