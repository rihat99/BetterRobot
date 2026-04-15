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

| Residual | Analytic Jacobian | Status |
|----------|------------------|--------|
| `PoseResidual` (6D) | Jr_inv(log(Terr)) @ J_frame | Implemented |
| `PositionResidual` (3D) | top 3 rows of frame Jacobian | Implemented |
| `OrientationResidual` (3D) | Jr_so3(log(Rerr)) @ J_frame[3:] | Implemented |
| `JointPositionLimit` (2*nq) | diagonal | Implemented |
| `RestResidual` (nv) | weight * I | Implemented |
| Smoothness (5pt) | tridiagonal | Stub |
| Manipulability | autodiff only | Stub |
| Collision | sparse analytic | Stub |

## Adding a New Residual

1. Create class implementing the protocol in a new file
2. Decorate with `@register_residual("name")`
3. Implement `__call__` (required) and `jacobian` (optional but preferred)
4. Residual must be a pure function of `ResidualState` — no side effects
