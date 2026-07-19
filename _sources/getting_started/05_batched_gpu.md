# Batching and the GPU

Batching means putting independent problems on leading tensor axes. Instead of
calling FK a thousand times in a Python loop, pass a tensor shaped
`(1000, nq)`. BetterRobot keeps that leading axis on every output.

This example runs on CUDA when it is available and otherwise runs unchanged on
CPU. The model and query are created on the same device because BetterRobot
does not silently copy either one.

```{testcode}
import torch
import better_robot as br
from robot_descriptions import panda_description

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = br.load(panda_description.URDF_PATH, device=device)
q_batch = model.q_neutral.expand(1000, -1).clone()

data = br.forward_kinematics(model, q_batch, compute_frames=True)
hand = data.frame_pose_world[..., model.frame_id("body_panda_hand"), :]

print(q_batch.shape)
print(hand.shape)
print(hand.device.type == device.type)
```

```{testoutput}
torch.Size([1000, 8])
torch.Size([1000, 7])
True
```

The batch prefix may have more than one axis, for example `(people, samples,
nq)`. A single configuration has an empty prefix and shape `(nq,)`; there is no
required batch-of-one wrapper.

Batching reduces Python overhead and gives PyTorch enough parallel work to use
a GPU well. It does not guarantee that every small problem is faster on a GPU:
transfer cost, dtype, robot size, and batch size still matter. Keep the model
and all input tensors on one device, and measure the workload you actually
run.

For the design behind these shape rules, see
{doc}`/concepts/the_compute_seam`.
