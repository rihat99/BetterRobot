# Dynamics

Dynamics links motion and force. Given a configuration `q`, a velocity `v` and an acceleration `a`, inverse
dynamics returns the joint torques that make that motion. Forward dynamics goes the other way, from torques to
accelerations. Every function takes the `KinematicState` of `q`, so forward kinematics runs once for all of them.

The names you will use:

| Name | What it does |
|---|---|
| `br.inverse_dynamics(model, state, v=, a=)` | the joint torques `(nv,)` for that motion (RNEA) |
| `br.mass_matrix(model, state)` | the joint-space mass matrix `(nv, nv)` (CRBA) |
| `br.forward_dynamics(model, state, v=, tau=)` | the accelerations the torques cause (ABA) |
| `dynamics.contact_wrenches(model, state, forces=, frames=)` | world forces `(C, 3)` at named frames as the `external_forces` of inverse and forward dynamics |
| `br.center_of_mass(model, state)` | the robot's centre of mass `(3,)` |
| `dynamics.centroidal_momentum(model, state, v=)` | linear and angular momentum about the centre of mass |

## Torques and the mass matrix

```python
import torch
from robot_descriptions import panda_description

import better_robot as br

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
generator = torch.Generator().manual_seed(0)
q = br.terms.random_start(model, model.q_neutral.expand(16, -1), generator=generator)
v = torch.randn(16, model.nv, generator=generator, dtype=torch.float64)
a = torch.randn(16, model.nv, generator=generator, dtype=torch.float64)
state = br.forward_kinematics(model, q)

tau = br.inverse_dynamics(model, state, v=v, a=a)
M = br.mass_matrix(model, state)
print(tau.shape, M.shape)
```
```output
torch.Size([16, 8]) torch.Size([16, 8, 8])
```

## Forward dynamics undoes inverse dynamics

```python
a_back = br.forward_dynamics(model, state, v=v, tau=tau)
print(torch.allclose(a_back, a, atol=1e-8))
```
```output
True
```

## Holding still against gravity

With zero velocity and zero acceleration, inverse dynamics gives the torques that hold the arm against gravity.

```python
still = br.forward_kinematics(model, model.q_neutral)
zero = torch.zeros(model.nv, dtype=torch.float64)
print(br.inverse_dynamics(model, still, v=zero, a=zero))
print(br.center_of_mass(model, still))
```
```output
tensor([ 8.9154e-18, -5.3059e+00,  8.9154e-18, -2.0009e+00,  4.5044e-02,
         2.1752e+00,  1.3919e-03,  0.0000e+00], dtype=torch.float64)
tensor([0.0306, 0.0061, 0.6073], dtype=torch.float64)
```

See also: [Kinematics](kinematics.md) for the state these functions read.
