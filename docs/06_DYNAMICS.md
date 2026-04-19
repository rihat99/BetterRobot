# 06 · Dynamics — RNEA, ABA, CRBA, Centroidal, Integrators

Dynamics lands as a **skeleton** in v1: every public function is named,
signed, docstringed, and raises `NotImplementedError` until the
implementation is ready. Function names keep Pinocchio's acronyms (`rnea`,
`aba`, `crba`, `ccrba`) because they are the canonical names in the
literature; **storage names on `Data` are readable** (`mass_matrix`,
`bias_forces`, `centroidal_momentum`, …) per
[13_NAMING.md](13_NAMING.md). Implementing the bodies is a later phase —
the point of this document is to lock the shape so that `residuals/`,
`optim/`, and `tasks/` can be written against a stable dynamics API.

> The three-layer split (DifferentialActionModel → IntegratedActionModel →
> ActionModel) is retained verbatim from Crocoddyl. See §6.

## 1. Entry points

```python
# src/better_robot/dynamics/__init__.py
from .rnea        import rnea, bias_forces, compute_generalized_gravity, compute_coriolis_matrix
from .aba         import aba
from .crba        import crba, compute_minverse
from .centroidal  import center_of_mass, compute_centroidal_map, compute_centroidal_momentum, ccrba
from .derivatives import (
    compute_rnea_derivatives,
    compute_aba_derivatives,
    compute_crba_derivatives,
    compute_centroidal_dynamics_derivatives,
)
from .integrators import integrate_q, symplectic_euler, rk4, semi_implicit_euler
```

## 2. Canonical signatures

All algorithms operate on a single `Data` object with a leading batch axis.
All inputs/outputs are plain `torch.Tensor` with shape `(B..., nv)` or
`(B..., nv, nv)`.

```python
def rnea(
    model: Model,
    data: Data,
    q: Tensor,          # (B..., nq)
    v: Tensor,          # (B..., nv)
    a: Tensor,          # (B..., nv)
    *,
    fext: Tensor | None = None,   # (B..., njoints, 6) external wrenches per joint, local frame
) -> Tensor:            # (B..., nv)
    """Inverse dynamics: τ = M(q) a + b(q, v) + g(q) - J^T fext.

    Populates `data.tau`, `data.joint_pose_world`, `data.v`, `data.a`
    along the way. Uses Featherstone's RNEA in two passes: forward
    (velocities, accelerations) then backward (forces and joint torques).

    Status: NotImplementedError (planned — see 06 DYNAMICS §5).
    """
```

```python
def aba(
    model: Model,
    data: Data,
    q: Tensor,
    v: Tensor,
    tau: Tensor,
    *,
    fext: Tensor | None = None,
) -> Tensor:            # (B..., nv)
    """Forward dynamics via Articulated Body Algorithm.

    Returns joint accelerations `ddq` = M(q)^{-1}(τ - b(q, v) - g(q) + J^T fext).
    Populates `data.ddq`.

    Status: NotImplementedError.
    """
```

```python
def crba(
    model: Model,
    data: Data,
    q: Tensor,
) -> Tensor:            # (B..., nv, nv)
    """Composite Rigid Body Algorithm — joint-space inertia matrix M(q).

    Populates `data.mass_matrix`.

    Status: NotImplementedError.
    """
```

```python
def compute_minverse(
    model: Model,
    data: Data,
    q: Tensor,
) -> Tensor:            # (B..., nv, nv)
    """Direct computation of M(q)^{-1} via the ABA factorisation.

    Cheaper than `crba` + `cholesky_solve` when only the inverse is needed.

    Status: NotImplementedError.
    """
```

```python
def bias_forces(
    model: Model,
    data: Data,
    q: Tensor,
    v: Tensor,
) -> Tensor:            # (B..., nv)
    """Bias forces (aka "non-linear effects"): C(q, v) v + g(q).

    Populates `data.bias_forces`. Shorthand for `rnea(q, v, a=0)` with a
    specialised path that avoids the mass-matrix multiply.

    The short historical name `nle` is kept as a one-release deprecated
    alias; see 13_NAMING.md.

    Status: NotImplementedError.
    """
```

```python
def compute_generalized_gravity(model, data, q) -> Tensor:  # (B..., nv)
def compute_coriolis_matrix(model, data, q, v) -> Tensor:    # (B..., nv, nv)
```

All raise `NotImplementedError` for now.

## 3. Centroidal

```python
def center_of_mass(
    model: Model,
    data: Data,
    q: Tensor,
    v: Tensor | None = None,
    a: Tensor | None = None,
) -> Tensor:            # (B..., 3)
    """Whole-body center of mass and its time derivatives.

    Populates `data.com_position`, `data.com_velocity`,
    `data.com_acceleration` (each shape (B..., 3)) when the corresponding
    input is non-None.

    Status: NotImplementedError.
    """

def compute_centroidal_map(model: Model, data: Data, q: Tensor) -> Tensor:
    """Centroidal momentum matrix A_g(q) ∈ (B..., 6, nv).
    Populates `data.centroidal_momentum_matrix`."""

def compute_centroidal_momentum(model: Model, data: Data, q: Tensor, v: Tensor) -> Tensor:
    """h_g = A_g(q) v ∈ (B..., 6). Populates `data.centroidal_momentum`."""

def ccrba(model: Model, data: Data, q: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Centroidal CRBA — returns (A_g, h_g). Pinocchio's primary centroidal call."""
```

## 4. Derivatives

These are the high-value functions in Pinocchio's `algorithm/derivatives/`
directory and are what make analytic gradients of dynamics cheap enough for
DDP / iLQR. Same leading-batch convention.

```python
def compute_rnea_derivatives(
    model, data, q, v, a, fext=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (∂τ/∂q, ∂τ/∂v, ∂τ/∂a=M) each (B..., nv, nv)."""

def compute_aba_derivatives(
    model, data, q, v, tau, fext=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (∂a/∂q, ∂a/∂v, ∂a/∂τ=M^{-1}) each (B..., nv, nv)."""

def compute_crba_derivatives(model, data, q) -> Tensor:
    """Return ∂M/∂q ∈ (B..., nv, nv, nv)."""

def compute_centroidal_dynamics_derivatives(model, data, q, v, a) -> ...:
    ...
```

## 5. Implementation strategy

The skeletons land immediately; the bodies arrive in the following order.
Each milestone is independently testable. Perf budgets (latency, memory)
come from [14_PERFORMANCE.md](14_PERFORMANCE.md).

1. **Milestone D0 — skeleton (this doc).** All functions present, all raise
   `NotImplementedError`, all have docstrings and signatures. Tests assert
   shape annotations and that the symbols exist.
2. **Milestone D1 — `center_of_mass`.** Pure kinematic function, easy win.
   Inertia accumulation over topo order.
3. **Milestone D2 — `rnea`.** Two-pass Featherstone over the topology using
   `Motion`/`Force`/`Inertia` from `spatial/`. Use `better_human/smpl/dynamics.py`
   as a reference for pattern, not for code — that file is flagged as
   "poorly written" and needs a rewrite.
4. **Milestone D3 — `crba`.** Composite-rigid-body inertia pass; reuses
   Milestone D2 primitives.
5. **Milestone D4 — `aba`.** Featherstone ABA; depends on `rnea` kernels.
6. **Milestone D5 — centroidal** (`compute_centroidal_map`, `ccrba`).
7. **Milestone D6 — derivatives.** Carpentier & Mansard analytic derivative
   formulas; this is where analytic gradients of dynamics start paying off
   over autograd. RNEA/ABA backward is a **second analytic kernel**, not
   `jacrev` (see [14_PERFORMANCE.md §2.3](14_PERFORMANCE.md)).
8. **Milestone D7 — three-layer action model** (§6 below).

**Forward vs backward ownership.** Each algorithm owns both its forward
and backward kernels. `rnea.forward` calls Featherstone's two-pass
recursion; `rnea.backward` (wrapped via
`torch.autograd.Function.apply`) calls `compute_rnea_derivatives`. The
autograd integration is spec'd in
[17_CONTRACTS.md §Autograd](17_CONTRACTS.md).

At each step the autograd fallback is: *if the function is not implemented,
users can still compose kinematics and run autograd-through-FK based
optimisation.* Dynamics is optional for v1 IK / trajopt over kinematic
residuals.

## 6. Three-layer action model (Crocoddyl pattern)

After the core dynamics algorithms land, BetterRobot grows a clean
integration layer modelled after Crocoddyl's
`DifferentialActionModel` / `IntegratedActionModel` / `ActionModel` split:

```
dynamics/action/
├── differential.py   # continuous-time dynamics model
├── integrated.py     # discrete-time wrapper (Euler / RK2 / RK4 / symplectic)
└── action.py         # what the optimal-control solver sees
```

### `DifferentialActionModel`

```python
@dataclass
class DifferentialActionModel:
    """Continuous-time dynamics + per-knot cost.

    Responsibilities:
      - Define  ẋ = f(x, u)
      - Compute Fx, Fu (Jacobians of f)
      - Evaluate cost r(x, u) and its gradient / Hessian
    """
    model: Model
    state_manifold: "StateManifold"
    cost_stack: "CostStack"

    def calc(self, data: "ActionData", x: Tensor, u: Tensor) -> None: ...
    def calc_diff(self, data: "ActionData", x: Tensor, u: Tensor) -> None: ...
```

Concrete class (and the only one in v1):

```python
class DifferentialActionModelFreeFwd(DifferentialActionModel):
    """Forward dynamics with no contacts: calls aba + aba_derivatives."""
```

### `IntegratedActionModel`

```python
@dataclass
class IntegratedActionModelEuler(ActionModel):
    differential: DifferentialActionModel
    dt: float
    with_cost_residual: bool = True

    def calc(self, data, x, u) -> None:
        """x_{k+1} = state_manifold.integrate(x, dt * f(x, u))."""
    def calc_diff(self, data, x, u) -> None:
        """Compose Fx_cont * dt * Jintegrate with the state manifold jacobians."""
```

Plus `IntegratedActionModelRK4` using the standard 4-stage Butcher tableau.

### `ActionModel`

The abstract base class that the optimal-control solver (future DDP/iLQR)
sees:

```python
class ActionModel(Protocol):
    state: "StateManifold"
    nu: int
    def calc(self, data, x, u) -> None: ...
    def calc_diff(self, data, x, u) -> None: ...
```

These three layers are *skeletonised in v1* — the data classes and
signatures exist, the calc bodies raise `NotImplementedError` with a pointer
to this document.

## 7. `StateManifold` — the Crocoddyl lesson

The state of a rigid-body system is `(q, v)` where `q` lives on a manifold
(SE(3) × SO(3) × ℝ^k for a free-flyer robot). Crocoddyl handles this with a
`StateAbstractTpl` that provides `integrate`, `diff`, `Jintegrate`, `Jdiff`
on the state manifold. BetterRobot adopts the same pattern:

```python
# src/better_robot/dynamics/state_manifold.py

class StateMultibody:
    """State manifold of a rigid-body system.

    Total configuration = (q, v).
    nx  = nq + nv       (representation dim)
    ndx = 2 * nv        (tangent dim)

    Uses model.integrate / model.difference internally — which themselves
    dispatch through the per-joint JointModel implementations.
    """
    def __init__(self, model: Model): ...
    nx: int
    ndx: int

    def zero(self) -> Tensor: ...
    def integrate(self, x: Tensor, dx: Tensor) -> Tensor: ...
    def diff(self, x0: Tensor, x1: Tensor) -> Tensor: ...
    def jacobian_integrate(self, x: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]: ...
    def jacobian_diff(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]: ...
```

Again: skeleton only. Implementations defer to the per-joint
`JointModel.integrate` / `JointModel.difference` routines that are
**already required** by the data model and therefore cheap to assemble.

## 8. Integrators

```python
# src/better_robot/dynamics/integrators.py

def integrate_q(model: Model, q: Tensor, v: Tensor, dt: float) -> Tensor:
    """Retract q by dt * v via model.integrate (a.k.a. `q ⊕ dt*v`).

    Implementable today — uses only kinematic machinery.
    """

def semi_implicit_euler(model, data, q, v, tau, dt, *, fext=None): ...
def symplectic_euler   (model, data, q, v, tau, dt, *, fext=None): ...
def rk4                (model, data, q, v, tau, dt, *, fext=None): ...
```

`integrate_q` is **implementable at milestone D0** — it needs nothing from
the dynamics skeleton, only the configured `JointModel.integrate` routines
from the data model. Everything else waits for `aba`.

## 9. Relation to `better_human/smpl/dynamics.py`

The user flagged `better_human/src/better_human/smpl/dynamics.py` as
"reference kinematics and dynamics, poorly written but useful as inspiration".
Plan:

- Treat it as **pseudocode only**. Do not port file-for-file.
- Extract the useful ideas: two-pass Featherstone structure, inertia
  propagation, gravitational acceleration as a spatial bias, per-body
  velocity recursion.
- Rewrite from scratch using `spatial/motion.py` + `spatial/force.py` +
  `spatial/inertia.py` primitives rather than raw pypose calls.
- Keep the per-joint dispatch via `JointModel.joint_motion_subspace` so that
  a free-flyer or spherical joint slots in without branches.

## 10. What lands in v1

| Function | v1 status |
|----------|-----------|
| `rnea`, `aba`, `crba`, `bias_forces` | skeleton only |
| `compute_*_derivatives`         | skeleton only |
| `center_of_mass`                | **implemented** (pure kinematic) |
| `compute_centroidal_map`, `ccrba` | skeleton |
| `integrate_q`                   | **implemented** |
| `semi_implicit_euler`, `rk4`, `symplectic_euler` | skeleton |
| `DifferentialActionModel`, `IntegratedActionModel`, `StateMultibody` | skeleton |

Tests (`tests/dynamics/test_skeleton.py`) assert every symbol exists and
raises the correct error with a useful message; once a milestone lands the
test switches to a numerical check against a pinned reference (see
[16_TESTING.md §4.5](16_TESTING.md)).
