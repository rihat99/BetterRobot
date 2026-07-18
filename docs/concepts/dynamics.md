# Dynamics

Kinematics describes motion without asking what caused it. Dynamics connects
that motion to mass and force. BetterRobot implements the standard recursive
rigid-body algorithms from Featherstone and keeps their familiar names:

- **RNEA** is inverse dynamics: given position, velocity, and acceleration,
  compute the required joint torque.
- **ABA** is forward dynamics: given position, velocity, and joint torque,
  compute joint acceleration.
- **CRBA** computes the joint-space mass matrix.
- **CCRBA** computes the centroidal momentum map and momentum.

The names are terse because they are the names used in textbooks and other
robotics libraries. Result fields use descriptive names such as
`mass_matrix`, `bias_forces`, and `centroidal_momentum`.

For a detailed derivation, see Featherstone's
[Rigid Body Dynamics Algorithms](https://royfeatherstone.org/spatial/v2/)
or Pinocchio's
[algorithm documentation](https://docs.ros.org/en/rolling/p/pinocchio/).

## The three central equations

Inverse dynamics evaluates

```{math}
\tau = M(q)\,a + b(q, v),
```

where `M(q)` is the mass matrix and `b(q, v)` contains Coriolis,
centrifugal, and gravity effects. `rnea(model, q, v, a)` returns `tau`.
External joint wrenches can be supplied with `fext=`.

Forward dynamics solves the same relationship for acceleration:

```{math}
a = M(q)^{-1}\left(\tau - b(q, v)\right).
```

`aba(model, q, v, tau)` computes it without explicitly forming and inverting
the mass matrix. Use `crba(model, q)` when the matrix itself is the desired
quantity.

`bias_forces(model, q, v)` returns `b(q, v)`, and
`compute_generalized_gravity(model, q)` returns its gravity-only part. There
is no separate public Coriolis matrix because most callers need its product
with velocity, not the matrix.

## Return values and workspaces

Public dynamics functions return their main tensor. They allocate temporary
workspace internally. Pass `data=some_data` when later work also needs the
intermediate poses, velocities, or named result fields; the exact object you
provide is populated.

Tensor shapes follow the same leading-batch rule as kinematics:

```text
q                                  (batch..., nq)
v, a, tau, bias                    (batch..., nv)
mass matrix                        (batch..., nv, nv)
external joint wrenches            (batch..., njoints, 6)
```

External wrenches are expressed in each joint's local coordinates. Check a
function's API reference before mixing quantities from different frames.

## Centroidal quantities

The centroidal momentum map `A_g(q)` relates joint velocity to the robot's
total spatial momentum about its center of mass:

```{math}
h_g = A_g(q)\,v.
```

`compute_centroidal_map` returns `A_g`, and
`compute_centroidal_momentum` returns `h_g`. `ccrba` computes both and returns
`CCRBAResult` with fields `centroidal_map` and `momentum`; the result can also
be unpacked as a two-tuple.

`center_of_mass` returns position and can populate velocity when `v` is
provided. Center-of-mass acceleration is not implemented; passing `a` reaches
an explicit guard listed in the {doc}`/reference/roadmap`.

## Differentiation

RNEA, ABA, CRBA, and CCRBA are written with differentiable PyTorch operations.
Ordinary `loss.backward()` therefore differentiates through the calculation.
The public raw variants accept `ModelStructure` and `ModelValues`, which also
lets gradients flow to model parameters.

Three convenience functions form complete Jacobians with PyTorch autograd:
`compute_rnea_derivatives`, `compute_aba_derivatives`, and
`compute_crba_derivatives`. With batched inputs, their outputs retain both the
output and input batch axes. Apply them per sample if you want only the
batch-diagonal blocks.

These helpers favor clarity over the specialized analytic derivative
recursions found in some C++ libraries. Replacing their implementation later
does not require changing their public signatures.

## Configuration integration is not simulation

`integrate_q(model, q, v, dt)` applies the manifold-aware step
`model.integrate(q, dt * v)`. It updates a configuration geometrically; it
does not calculate forces, resolve contact, or advance a simulated world.

BetterRobot deliberately stops at analysis and optimization. A simulator
must decide how contacts, actuators, constraints, and time stepping interact.
See {ref}`decision-analysis-only` for why that is a separate product.

## Numerical anchors

Parity tests compare the recursive algorithms with Pinocchio across fixed and
floating bases, several joint families, batches, external forces, and both
common floating dtypes. Autograd tests cover gradients through the public and
raw passes. Those tests, rather than a second implementation, are the
correctness anchor.

Read {doc}`kinematics_and_jacobians` for the poses and frames used inside the
recursions, or {doc}`residuals_costs_and_solvers` for using dynamics inside an
estimation problem.
