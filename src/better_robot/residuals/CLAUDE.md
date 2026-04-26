# residuals/ — Pure Functions from State to Residual Vector

## Residual Protocol

Every residual implements:
```python
name: str                                    # unique identifier
dim: int                                     # output dimension
__call__(state: ResidualState) -> Tensor     # (B..., dim) residual vector
jacobian(state: ResidualState) -> Tensor | None  # (B..., dim, nv) or None
```

If `jacobian()` returns `None`, the solver falls back to autodiff on `__call__`. This is the `JacobianStrategy.AUTO` pattern.

Trajectory residuals can also override:
```python
apply_jac_transpose(state, vec) -> Tensor    # (B..., nv) — matrix-free J^T r
spec(state) -> ResidualSpec | property        # structural metadata for sparse solvers
```

The default `apply_jac_transpose` (in `base.py`) materialises `J = self.jacobian(state)` and returns `J.mT @ vec`; banded / sparse residuals override to skip the dense Jacobian.

## ResidualState

```python
@dataclass
class ResidualState:
    model: Model
    data: Data          # FK already computed
    variables: Tensor   # (B..., nx) flat optimization variable
```

## Registry

Use `@register_residual("name")` decorator. Lookup via `get_residual("name")`.

## Implementation Status

| Residual | Analytic Jacobian | `apply_jac_transpose` | Status |
|----------|------------------|----------------------|--------|
| `PoseResidual` (6D) | Jr_inv(log(Terr)) @ J_frame | default | Implemented |
| `PositionResidual` (3D) | top 3 rows of frame Jacobian | default | Implemented |
| `OrientationResidual` (3D) | Jr_so3(log(Rerr)) @ J_frame[3:] | default | Implemented |
| `JointPositionLimit` (2*nq) | diagonal | default | Implemented |
| `JointVelocityLimit` (2*nv) | — | default | `__call__` implemented; `.jacobian` is a stub |
| `JointAccelLimit` (2*nv) | — | default | Stub |
| `RestResidual` (nv) | weight * I | default | Implemented |
| `NullspaceResidual` | — | default | Stub |
| `ReferenceTrajectoryResidual` (T*nv) | block-diagonal | overridden (per-frame scaling) | Implemented |
| `VelocityResidual` (nv*(T-2)) | banded | overridden | Implemented |
| `AccelerationResidual` (nv*(T-2)) | tridiagonal `[+I, −2I, +I] / dt²` | overridden | Implemented |
| `JerkResidual` | — | — | Stub |
| `TimeIndexedResidual` | wraps inner residual at fixed knot | passes through | Implemented |
| `ContactConsistencyResidual` (3*K*(T-1)) | LWA frame-Jacobian linear rows | overridden | Implemented |
| `YoshikawaResidual` (1) | — | default (autodiff) | Stub |
| `SelfCollisionResidual` (n_pairs) | sparse analytic | default | Stub |
| `WorldCollisionResidual` | sparse analytic | default | Stub |

## Adding a New Residual

1. Create class implementing the protocol in a new file
2. Decorate with `@register_residual("name")`
3. Implement `__call__` (required) and `jacobian` (optional but preferred)
4. For trajectory-scale or sparse residuals, override `apply_jac_transpose` to skip the dense Jacobian
5. Optionally expose `.spec` returning a `ResidualSpec` so block-sparse solvers can pre-build masks
6. Residual must be a pure function of `ResidualState` — no side effects
