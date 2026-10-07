# Rigid-body dynamics reads the kinematic state

After this page you can compute joint torques for a motion, the joint-space mass matrix, accelerations
under given torques, and the centre of mass and centroidal momentum, all batched and differentiable.
Every function lives in `better_robot.dynamics.state` and takes `(model, state, ...)`, where `state` comes
from [`forward_kinematics_state`](kinematics-and-jacobians.md).

## The equation of motion

For a tree of rigid bodies with configuration `q (B..., nq)` and velocity `v (B..., nv)`:

$$
M(q)\,\dot v + C(q, v)\,v + g(q) = \tau + J^\top f_\text{ext}
$$

| Function | Returns | Shape | Algorithm |
|----------|---------|-------|-----------|
| `inverse_dynamics(model, state, v, a, fext=None)` | $\tau$ for acceleration `a` | `(B..., nv)` | recursive Newton–Euler (RNEA) |
| `mass_matrix(model, state)` | $M(q)$, symmetric positive definite | `(B..., nv, nv)` | composite rigid body (CRBA) |
| `forward_dynamics(model, state, v, tau, fext=None)` | $\dot v$ | `(B..., nv)` | bias by RNEA, then a Cholesky solve with $M$ |
| `center_of_mass(model, state)` | whole-body COM, world frame | `(B..., 3)` | mass-weighted sum |
| `centroidal_map(model, state)` | $A_g(q)$ with $h_g = A_g v$ | `(B..., 6, nv)` | per-body momentum at the COM |
| `centroidal_momentum(model, state, v)` | $h_g$, world axes at the COM | `(B..., 6)` | $A_g v$ |

`forward_dynamics` does not run the articulated-body algorithm. It computes the bias
$C v + g - J^\top f_\text{ext}$ with one RNEA pass at zero acceleration, factors $M$ with
`torch.linalg.cholesky`, and solves with `torch.cholesky_solve`. The result matches Pinocchio's ABA
(see the parity table below).

```python
import torch
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics.state import forward_kinematics_state
from better_robot.dynamics.state import (
    inverse_dynamics, mass_matrix, forward_dynamics,
    center_of_mass, centroidal_map, centroidal_momentum,
)

builder = ModelBuilder("arm", gravity=[0.0, 0.0, -9.81])
for body in ("base", "link1", "link2"):
    builder.add_body(body, mass=1.0, com=torch.tensor([0.0, 0.0, 0.15]))
builder.add_revolute_z("j1", parent="base", child="link1")
builder.add_revolute_y("j2", parent="link1", child="link2",
                       origin=torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]))
model = build_model(builder.finalize(), dtype=torch.float64)

q = torch.randn(32, model.nq, dtype=torch.float64)
v = torch.randn(32, model.nv, dtype=torch.float64)
a = torch.randn(32, model.nv, dtype=torch.float64)
state = forward_kinematics_state(model, q)

tau = inverse_dynamics(model, state, v, a)
M = mass_matrix(model, state)
print(tau.shape, M.shape)                       # torch.Size([32, 2]) torch.Size([32, 2, 2])
```

## Forward and inverse dynamics invert each other

Feeding the torques back recovers the acceleration:

```python
a_back = forward_dynamics(model, state, v, tau)
print(torch.allclose(a_back, a, atol=1e-8))     # True
```

## Bias and gravity terms are inverse dynamics with zeros

There is no separate function for the nonlinear effects or the gravity torque:

- Bias forces $C(q, v)\,v + g(q)$: `inverse_dynamics(model, state, v, zeros)`.
- Gravity torque $g(q)$: `inverse_dynamics(model, state, zeros, zeros)`.

```python
zeros = torch.zeros_like(v)
gravity_torque = inverse_dynamics(model, state, zeros, zeros)
bias = inverse_dynamics(model, state, v, zeros)
```

Gravity is a model value, `model.gravity (6,)`, laid out as a spatial acceleration `[linear, angular]`.
Set its linear part when you build the model (`ModelBuilder(..., gravity=[gx, gy, gz])`); the default,
also used by the URDF and MJCF loaders, is `[0, 0, -9.81, 0, 0, 0]`.

## External wrenches enter per joint

`fext (B..., njoints, 6)` is one wrench `[force, torque]` per joint, expressed in that joint's own frame.
Both `inverse_dynamics` and `forward_dynamics` accept it as a keyword argument.

```python
fext = torch.zeros(32, model.njoints, 6, dtype=torch.float64)
fext[:, -1, 2] = 5.0                           # 5 N along z of the last joint's frame
tau_contact = inverse_dynamics(model, state, v, a, fext=fext)
```

## Centroidal quantities live at the centre of mass

`center_of_mass` returns the world position of the COM. `centroidal_map` returns $A_g$ whose rows are
`[linear momentum, angular momentum about the COM]` in world axes; `centroidal_momentum` multiplies it by
`v`. The linear rows divided by the total mass equal the COM velocity.

```python
com = center_of_mass(model, state)              # (32, 3)
h = centroidal_momentum(model, state, v)        # (32, 6)
print(com.shape, h.shape)
```

## The floating base is part of the tree

A floating-base model (`free_flyer=True`, see [Robot I/O](io.md)) has `nv = 6 + n_actuated`; the first six
entries of `tau` are the wrench the base would need, which is zero for a physically consistent motion.

## Parity with Pinocchio

| Quantity | Test file | Tolerance (`atol`, float64) |
|----------|-----------|-----------------------------|
| RNEA | `tests/test_pinocchio/test_rnea_matches_pinocchio.py` | `1e-10`–`1e-12` (float32: `2e-5`) |
| CRBA | `test_crba_matches_pinocchio.py` | `1e-10`–`1e-12` |
| Forward dynamics vs ABA | `test_aba_matches_pinocchio.py` | `1e-9`–`1e-12` |
| COM, centroidal map | `test_centroidal.py` | `1e-7`–`1e-12` |

API details: [dynamics reference](../reference/api/dynamics.md).
