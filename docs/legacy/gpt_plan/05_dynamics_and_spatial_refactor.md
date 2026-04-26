# Dynamics And Spatial Refactor

## Current State

`spatial/` is already close to the desired style:

- `Motion`, `Force`, `Inertia`, and `Symmetric3` are thin dataclasses over tensors.
- Vector-space operators are explicit.
- Ambiguous multiplication is avoided.
- Heavy math routes through Lie functions.

`dynamics/` has a partial RNEA implementation and skeletons for CRBA, ABA, centroidal quantities, derivatives, integrators, and action models.

## Recommendation

Use the spatial package as the template for Lie classes, and use the Lie refactor to make spatial operations cleaner.

Long term:

- high-level code accepts `SE3`, `Motion`, `Force`, and `Inertia` values,
- kernels operate on packed tensors,
- algorithm outputs can be wrapped in value types for readability,
- dynamics algorithms own analytic derivatives where performance matters.

## Spatial API Adjustments

Keep:

- `Motion.data`, `Force.data`, `Inertia.data`.
- `Motion + Motion`, `Force + Force`.
- Named cross operations such as `cross_motion` and `cross_force`.
- No ambiguous `__mul__` for cross products.

Change:

- Rename `.data` to `.tensor` eventually, matching proposed `SO3` and `SE3`.
- Keep `.data` as a deprecated alias for one release if needed.
- Accept `SE3` in `se3_action`, while still accepting raw tensors internally.

Example:

```python
def se3_action(self, T: SE3 | torch.Tensor) -> "Motion":
    pose = T.tensor if isinstance(T, SE3) else T
    ...
```

## Dynamics Milestones

### D1: RNEA Hardening

RNEA should become the first fully production-grade dynamics function.

Tasks:

- Validate against Pinocchio for fixed-base, floating-base, spherical, planar, and mimic cases.
- Add batch tests.
- Add external wrench tests.
- Add gravity sign convention tests.
- Add derivative finite-difference tests.
- Document body-frame vs world-frame quantities for every cached field.

### D2: CRBA

Implement composite rigid body algorithm:

- outputs `mass_matrix`,
- reuses local placements from FK,
- fills only lower/upper then symmetrizes,
- matches Pinocchio in fp64,
- supports batch.

### D3: ABA

Implement articulated body algorithm:

- outputs `ddq`,
- supports external forces,
- validates `M @ ddq + b = tau` against CRBA/RNEA,
- becomes the default forward dynamics kernel.

### D4: Centroidal

Implement:

- center of mass,
- centroidal momentum matrix,
- centroidal momentum,
- centroidal derivatives if required for tasks.

### D5: Derivatives

Add analytic derivatives with clear ownership:

- RNEA derivatives,
- ABA derivatives,
- integrate/difference Jacobians,
- action model derivatives.

Where Torch autograd is too slow or wrong, use custom `torch.autograd.Function`.

## JointModel Hooks For Dynamics

Add before more joints land:

```python
def joint_bias_acceleration(self, q_slice, v_slice) -> torch.Tensor:
    # c_J, shape (..., 6)
    ...

def joint_transform_jacobian(self, q_slice) -> torch.Tensor:
    ...

def joint_motion_subspace_derivative(self, q_slice, v_slice) -> torch.Tensor:
    ...
```

The current RNEA assumes `c_J = 0`. That assumption should live in each simple joint model, not in the algorithm forever.

## StateMultibody

Finish `StateMultibody` early. It is the shared manifold contract for:

- dynamics action models,
- optimal control,
- trajectory optimization with velocities,
- shooting problems,
- state regularization residuals.

Minimum methods:

- `zero()`,
- `integrate(x, dx)`,
- `diff(x0, x1)`,
- `jacobian_integrate(x, dx)`,
- `jacobian_diff(x0, x1)`.

## Action Models

Keep the Crocoddyl split:

- `DifferentialActionModel`: continuous dynamics and running cost.
- `IntegratedActionModel`: integration scheme and discrete transition.
- `ActionModel`: generic interface for shooting solvers.

But adapt it for PyTorch batching:

- no per-knot Python callbacks in hot loops,
- cost stacks can be shared across knots,
- per-knot params live in tensors,
- derivatives can be dense initially and sparse later.

## Packed Tensor Compatibility

Every value type should have a packed tensor representation:

- `SO3`: `(..., 4)`.
- `SE3`: `(..., 7)`.
- `Motion`: `(..., 6)`.
- `Force`: `(..., 6)`.
- `Inertia`: `(..., 10)`.

This makes Warp kernels possible without exposing Warp or custom Python objects.

## Tests To Add

- Spatial value methods match raw tensor formulas.
- Spatial action accepts both `SE3` and tensor poses.
- RNEA matches Pinocchio across joint families.
- RNEA gradients match central finite differences for small robots.
- CRBA is symmetric positive semidefinite and matches RNEA columns.
- ABA satisfies inverse dynamics consistency.
- `StateMultibody.integrate` and `diff` roundtrip.
- Dynamics functions reject unsupported dtypes and mismatched devices clearly.
