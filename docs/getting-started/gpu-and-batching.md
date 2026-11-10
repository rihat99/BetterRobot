# Run on a GPU and in batches

You learn the two rules that make BetterRobot fast: batch axes lead, event axes trail; and the model and
your tensors share one device and dtype. You run a batched query on CPU and see the CUDA version.

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/batched_ik.png" width="960"><source src="../../clips/batched_ik.webm" type="video/webm"></video>
<figcaption>One <code>solve_ik</code> call on a batch of targets: every Panda in the grid is one row of the same <code>q</code> tensor.</figcaption></figure>

## Batch axes lead, event axes trail

Every public tensor ends in its event shape: `q (..., nq)`, a pose `(..., 7)`, a transform `(..., 3, 4)`.
Everything before that is batch, and you may use any number of batch axes. Results keep them:

```python
import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot import forward_kinematics, frame_jacobian

model = br.load(panda_description.URDF_PATH, dtype=torch.float32)
hand = model.frame_id("panda_hand")

q = model.q_neutral
q_batch = q.expand(8, 16, model.nq)  # 8 robots x 16 time steps
state = forward_kinematics(model, q_batch)
print(state.frame_poses.shape)                            # (8, 16, nframes, 7)
print(frame_jacobian(model, state, frame="panda_hand").shape)  # (8, 16, 6, nv)
```
```output
torch.Size([8, 16, 14, 7])
torch.Size([8, 16, 6, 8])
```

Batch axes follow PyTorch broadcasting. A `(5, 1, nq)` tensor stays `(5, 1, ...)`: nothing is squeezed.
Prefer one call on a big batch over a Python loop of small calls; each call has a fixed launch cost.

## Model and tensors share a device and dtype

`br.load` takes `device=` and `dtype=` (default `float32`). `Model.to` moves or casts a model. Public
functions check that `q` matches the model and raise instead of casting silently:

```python
q64 = torch.zeros(3, model.nq, dtype=torch.float64)
try:
    forward_kinematics(model, q64)
except Exception as error:
    print(type(error).__name__, error)

model64 = model.to(dtype=torch.float64)
print(forward_kinematics(model64, q64).frame_poses.dtype)
```
```output
DtypeMismatchError q.dtype=torch.float64 != model.dtype=torch.float32; pass q.to(torch.float32)
torch.float64
```

Use `float64` for tests, parity checks and tight IK tolerances; `float32` for throughput.

## Run the same code on CUDA

Only the device changes:

```python { .no-run data-reason="needs a CUDA device; the docs run on the CPU" }
device = torch.device("cuda")
model = br.load(panda_description.URDF_PATH, device=device, dtype=torch.float32)
q_batch = model.q_neutral.expand(4096, model.nq).clone()
state = forward_kinematics(model, q_batch)  # runs on the GPU
torch.cuda.synchronize()
```

## What batching buys on a GPU

At batch 1 the cost of a call is operator dispatch, which the CPU does faster. On the GPU the cost of a call
hardly grows from one query to thousands, so a large batch runs many times faster than on one CPU thread. Measure
both on your machine (Panda, batches 1 and 4096):

```bash
CUDA_VISIBLE_DEVICES="" uv run python benchmarks/run.py --device cpu --batches 1,4096 --models panda --algos rnea,crba,aba
uv run python benchmarks/run.py --device cuda --batches 1,4096 --models panda --algos rnea,crba,aba
```

[Benchmark and profile](../guides/benchmark-and-profile.md) explains the columns it prints.

!!! tip
    Batch first, then move to CUDA. A batch of one gains nothing from the GPU; a batch of thousands
    costs about the same as a batch of one.

## Where next

- [Visualize a robot](visualize.md) — show one of those batched solutions in the viewer.
- [Benchmark and profile](../guides/benchmark-and-profile.md) — measure your own robot and batch size.
- [Architecture](../concepts/architecture.md) — where the batching and device contract lives in the layers.
