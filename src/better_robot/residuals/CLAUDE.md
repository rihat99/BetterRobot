# residuals/ — Pure Functions from State to Residual Vector

## Residual Protocol

Every residual implements:
```python
name: str                                    # unique identifier
dim: int                                     # output dimension
__call__(state: ResidualState) -> Tensor     # (B..., dim) residual vector
jacobian(state: ResidualState) -> Tensor | None  # (B..., dim, nv) or None
```

If `jacobian()` returns `None`, `JacobianStrategy.AUTO` falls back to
unbatched central finite differences. The fallback evaluates the residual
`2 * nv + 1` times; real `torch.func` strategies are scheduled for M2.

Legacy trajectory residuals can also override:
```python
apply_jac_transpose(state, vec) -> Tensor    # (B..., nv) — matrix-free J^T r
```

The default `apply_jac_transpose` (in `base.py`) materialises `J = self.jacobian(state)` and returns `J.mT @ vec`; banded residuals override it. This path is retained only for `tests/optim/test_matrix_free.py`: no production solver or task calls `LeastSquaresProblem.gradient`.

## ResidualState

```python
@dataclass
class ResidualState:
    model: Model
    data: Data          # FK already computed
    variables: Tensor   # (B..., nx) flat optimization variable
```

## Registry

Residuals are constructed explicitly and composed into a `CostStack`; there is no process-wide registry.

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
| `JointRotationPrior` (nv) | tangent AD / legacy FD | default | Implemented; exact Lie derivative, no identity approximation |
| `SwingTwistLimitResidual` (3*J) | tangent AD / legacy FD | default | Implemented; zero-twist convention at pure-pi swing |
| `NullspaceResidual` | — | default | Stub |
| `ReferenceTrajectoryResidual` (T*nv) | block-diagonal | overridden (per-frame scaling) | Implemented |
| `VelocityResidual` (nv*(T-2)) | banded | overridden | Implemented |
| `AccelerationResidual` (nv*(T-2)) | tridiagonal `[+I, −2I, +I] / dt²` | overridden | Implemented |
| `JerkResidual` | — | — | Stub |
| `TimeIndexedResidual` | wraps inner residual at fixed knot | passes through | Implemented |
| `ContactConsistencyResidual` (3*K*(T-1)) | LWA frame-Jacobian linear rows | overridden | Implemented |
| `YoshikawaResidual` (1) | — | — | Stub |
| `SelfCollisionResidual` (n_pairs) | sparse analytic | default | Stub |
| `WorldCollisionResidual` | sparse analytic | default | Stub |

## Adding a New Residual

1. Create class implementing the protocol in a new file
2. Give the class a stable `name` attribute
3. Implement `__call__` (required) and `jacobian` (optional but preferred)
4. Override `apply_jac_transpose` only when extending the retained legacy matrix-free tests; it is not a live production solver path
5. Residual must be a pure function of `ResidualState` — no side effects
