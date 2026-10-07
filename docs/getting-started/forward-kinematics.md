# Compute forward kinematics

You compute where every joint and frame of the Panda sits for a configuration, read the hand pose, and
take its Jacobian. You also run the same call on a whole batch of configurations at once.

## Run forward kinematics once

Continue with the Panda from [the previous page](first-model.md):

```python
import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot.kinematics import forward_kinematics_state, frame_jacobian

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
hand = model.frame_id("body_panda_hand")

state = forward_kinematics_state(model, q)
print(state.joint_transform.shape)  # (njoints, 3, 4)
print(state.frame_pose_of(hand))    # (7,)
```
```output
torch.Size([14, 3, 4])
tensor([ 1.0731e-01, -3.2653e-17,  9.2494e-01,  9.2332e-01,  3.8245e-01,
        -3.2237e-02, -1.3353e-02], dtype=torch.float64)
```

`model.fk(q)` is the same call as a method, and `frame_pose_of` also takes a frame name:

```python
state = model.fk(q)
print(state.frame_pose_of("body_panda_hand"))  # (7,)
```
```output
tensor([ 1.0731e-01, -3.2653e-17,  9.2494e-01,  9.2332e-01,  3.8245e-01,
        -3.2237e-02, -1.3353e-02], dtype=torch.float64)
```

`forward_kinematics_state` returns a `KinematicState`: an immutable record of one query. It stores
`joint_transform (..., njoints, 3, 4)` and `frame_transform (..., nframes, 3, 4)`, the world `[R | p]` of
every joint and frame, and derives the rest on demand:

| Field | Shape | Meaning |
|---|---|---|
| `joint_translation`, `frame_translation` | `(..., n, 3)` | world position |
| `joint_rotation`, `frame_rotation` | `(..., n, 3, 3)` | world rotation matrix |
| `joint_pose`, `frame_pose` | `(..., n, 7)` | pose `[tx, ty, tz, qx, qy, qz, qw]` |
| `frame_pose_of(i)` | `(..., 7)` | pose of frame `i` only |

<figure markdown>
  ![The Panda at its neutral configuration, hand frame shown.](../images/panda_rest.png){ width="560" }
  <figcaption>The Panda at its neutral configuration, hand frame shown.</figcaption>
</figure>

## Batch it

Every public function takes arbitrary leading batch axes. A `(4, 10, nq)` tensor gives
`(4, 10, nframes, 7)` frame poses in one call, with no Python loop:

```python
generator = torch.Generator().manual_seed(0)
low, high = model.lower_pos_limit, model.upper_pos_limit
q_batch = low + (high - low) * torch.rand(4, 10, model.nq, generator=generator, dtype=torch.float64)

batch_state = forward_kinematics_state(model, q_batch)
print(batch_state.frame_pose.shape)                       # (4, 10, nframes, 7)
print(batch_state.frame_translation[..., hand, :].shape)  # (4, 10, 3)
```
```output
torch.Size([4, 10, 14, 7])
torch.Size([4, 10, 3])
```

[GPU and batching](gpu-and-batching.md) explains the batch-axis rule in full.

## Take the hand Jacobian

`frame_jacobian` reuses the state, so forward kinematics runs once. The result maps joint velocities
`v (nv,)` to the hand twist `[linear(3), angular(3)]`:

```python
J = frame_jacobian(model, state, frame="body_panda_hand")
print(J.shape)  # (6, nv)
```
```output
torch.Size([6, 8])
```

The default `reference="local_world_aligned"` expresses the twist at the frame origin with world axes.
Pass `"local"` or `"world"` for the other two conventions; see
[Kinematics and Jacobians](../concepts/kinematics-and-jacobians.md).

## Differentiate through it

The whole computation is PyTorch, so autograd works. The gradient of the hand height with respect to `q`:

```python
q_grad = q.clone().requires_grad_(True)
height = forward_kinematics_state(model, q_grad).frame_translation[hand, 2]
height.backward()
print(q_grad.grad)
```
```output
tensor([ 0.0000e+00, -1.0731e-01,  0.0000e+00,  2.4806e-02, -3.3616e-18,
         8.0323e-02,  0.0000e+00,  0.0000e+00], dtype=torch.float64)
```

## Where next

- [Solve inverse kinematics](inverse-kinematics.md) — go the other way: from a hand pose to a configuration.
- [Kinematics and Jacobians](../concepts/kinematics-and-jacobians.md) — the three Jacobian conventions and why the default is local-world-aligned.
- [Run on a GPU and in batches](gpu-and-batching.md) — the batch-axis rule in full.
