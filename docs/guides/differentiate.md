# Differentiate through kinematics and dynamics

Every public kinematics and dynamics function is plain PyTorch, so autograd reaches the configuration
`q`, the velocity `v` and the model's own tensors. This guide shows gradients with respect to each,
and how to use `torch.func.jacrev` for full Jacobians. Tests in `tests/autograd/`,
`tests/kinematics/test_state.py` and `tests/dynamics/test_state_dynamics.py` guard every path shown here.

## Gradients with respect to `q`

```python
import torch
import better_robot as br
from better_robot import kinematics
from better_robot.dynamics import state as dyn
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
hand = model.frame_id("body_panda_hand")
q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit).expand(4, -1).clone()
q.requires_grad_()

state = kinematics.forward_kinematics_state(model, q)
height = state.frame_translation[:, hand, 2]        # (4,)
height.sum().backward()
print(q.grad.shape)                                   # (4, nq)
```

`forward_kinematics_state` returns a `KinematicState` with world transforms `joint_transform
(B..., njoints, 3, 4)` and `frame_transform (B..., nframes, 3, 4)`, plus derived views
(`frame_translation`, `frame_pose`, `frame_pose_of(id)`). All of them carry the autograd graph.

The analytic `frame_jacobian` is itself differentiable, so second-order terms (for example a
manipulability gradient) need no extra code:

```python
q.grad = None
state = kinematics.forward_kinematics_state(model, q)
jacobian = kinematics.frame_jacobian(model, state, hand)    # (4, 6, nv), LOCAL_WORLD_ALIGNED
manipulability = torch.linalg.det(jacobian @ jacobian.mT)    # (4,)
manipulability.sum().backward()
print(q.grad.abs().max() > 0)
```

## Gradients with respect to `v` and `a`

`better_robot.dynamics.state` takes the same `KinematicState`, so one forward pass serves both
kinematics and dynamics. Its outputs are differentiable in `q` (through the state), `v`, `a` and the
external wrenches `fext`.

```python
v = torch.zeros(4, model.nv, dtype=torch.float64, requires_grad=True)
a = torch.zeros(4, model.nv, dtype=torch.float64)
state = kinematics.forward_kinematics_state(model, q)
tau = dyn.inverse_dynamics(model, state, v, a)               # (4, nv)
(tau ** 2).sum().backward()
print(v.grad.shape)                                          # (4, nv)
```

## Gradients with respect to model parameters

`model.values` holds the differentiable model tensors: `joint_placements`, `body_inertias` and
`frame_placements`. `model.with_values(...)` returns a new `Model` with the same topology and the
tensors you pass, checked for shape, dtype and device. The original model is unchanged.

```python
placements = model.values.joint_placements.clone().requires_grad_()
tuned = model.with_values(joint_placements=placements)
q_fixed = q.detach()
reach = kinematics.forward_kinematics_state(tuned, q_fixed).frame_translation[:, hand].norm(dim=-1)
reach.sum().backward()
print(placements.grad.shape)                                 # same shape as joint_placements
```

The same pattern fits link lengths or masses to data: make `body_inertias` a leaf tensor, pass it
through `with_values`, and optimize it with any `torch.optim` optimizer or with
[`better_robot.optim`](../concepts/optimization.md). Values may also carry leading batch axes, one
set of parameters per configuration.

## Full Jacobians with `torch.func`

The state API composes with `torch.func.jacrev`. The test suite checks that the result matches
`torch.autograd.functional.jacobian` and the analytic `frame_jacobian`.

```python
def hand_position(x: torch.Tensor) -> torch.Tensor:
    return kinematics.forward_kinematics_state(model, x).frame_translation[:, hand].sum(0)

by_func = torch.func.jacrev(hand_position)(q_fixed)          # (3, 4, nq)
analytic = kinematics.frame_jacobian(model, kinematics.forward_kinematics_state(model, q_fixed), hand)
print(torch.allclose(by_func.permute(1, 0, 2), analytic[:, :3, :], atol=1e-10))
```

`jacrev` of `inverse_dynamics` with respect to `a` gives the mass matrix, one diagonal block per
batch element; `dyn.mass_matrix(model, state)` computes it directly and is cheaper.

!!! note "What is not differentiable"
    Integer topology in `model.structure` (parents, index tables, names) has no gradient by design.
    Gradients through `q` on a floating base flow through the quaternion entries; take steps with
    `model.integrate(q, dv)` so the quaternion stays on the unit sphere.

To differentiate through a whole solve rather than one function, see `solve_layer` in
[Optimization](../concepts/optimization.md).
