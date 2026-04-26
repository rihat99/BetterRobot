# Dynamics

The dynamics layer is where rigid-body physics meets numerical
discipline. RNEA computes inverse dynamics (`τ = M(q) a + b(q, v) +
g(q)`); ABA computes forward dynamics (`q̈` from `τ`); CRBA computes
the joint-space inertia matrix `M(q)`; CCRBA computes the centroidal
momentum matrix; the analytic derivatives — `∂τ/∂q`, `∂a/∂τ`,
`∂M/∂q`, the centroidal-dynamics derivatives — are what make
gradient-based DDP and iLQR cheap enough to be practical.

We kept Pinocchio's algorithm names verbatim (`rnea`, `aba`, `crba`,
`ccrba`) because every textbook on rigid-body dynamics uses them,
and renaming them would force readers to look up the equivalence
every time. We replaced Pinocchio's storage shorthand
(`oMi`, `oMf`, `liMi`, `nle`, `Ag`) with self-describing identifiers
(`joint_pose_world`, `frame_pose_world`, `joint_pose_local`,
`bias_forces`, `centroidal_momentum_matrix`). The names that matter
to the reader of a dynamics formula keep their math shape; the names
that matter only to the call site are spelled out.

The Featherstone passes are live: RNEA, ABA, CRBA, CCRBA, the
centroidal map and momentum, COM with derivatives, and the
autograd-derived `compute_*_derivatives` functions all work and pass
gradcheck. A handful of named symbols (`compute_minverse`,
`compute_coriolis_matrix`, the analytic Carpentier–Mansard
derivatives, the higher-order integrators, retargeting) are stubbed
with `NotImplementedError` and listed in {doc}`/reference/roadmap`.
The signatures are stable; only the bodies arrive incrementally.

## Entry points

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

All algorithms operate on a single `Data` object with a leading batch
axis. Inputs and outputs are plain `torch.Tensor` with shape
`(B..., nv)` or `(B..., nv, nv)`.

## Canonical signatures

```python
def rnea(
    model: Model,
    data: Data,
    q: Tensor,           # (B..., nq)
    v: Tensor,           # (B..., nv)
    a: Tensor,           # (B..., nv)
    *,
    fext: Tensor | None = None,   # (B..., njoints, 6) external wrenches per joint, local frame
) -> Tensor:             # (B..., nv)
    """Inverse dynamics: τ = M(q) a + b(q, v) + g(q) − Jᵀ fext.

    Two-pass Featherstone: forward (velocities, accelerations), then
    backward (forces, joint torques). Populates ``data.tau``,
    ``data.joint_pose_world``, ``data.v``, ``data.a``.
    """

def aba(
    model: Model,
    data: Data,
    q: Tensor,
    v: Tensor,
    tau: Tensor,
    *,
    fext: Tensor | None = None,
) -> Tensor:             # (B..., nv)
    """Forward dynamics via Articulated Body Algorithm.

    Returns ``q̈ = M(q)^{-1}(τ − b(q, v) − g(q) + Jᵀ fext)``.
    Populates ``data.ddq``.
    """

def crba(
    model: Model,
    data: Data,
    q: Tensor,
) -> Tensor:             # (B..., nv, nv)
    """Composite Rigid Body Algorithm — joint-space inertia M(q).
    Populates ``data.mass_matrix``."""

def bias_forces(
    model: Model,
    data: Data,
    q: Tensor,
    v: Tensor,
) -> Tensor:             # (B..., nv)
    """Bias forces (a.k.a. non-linear effects): C(q, v) v + g(q).
    Populates ``data.bias_forces``. Specialised path that avoids the
    mass-matrix multiply."""

def compute_generalized_gravity(model, data, q) -> Tensor:    # (B..., nv)
def compute_coriolis_matrix    (model, data, q, v) -> Tensor:  # (B..., nv, nv) — stub
def compute_minverse           (model, data, q) -> Tensor:     # (B..., nv, nv) — stub
```

## Centroidal

```python
def center_of_mass(
    model: Model,
    data: Data,
    q: Tensor,
    v: Tensor | None = None,
    a: Tensor | None = None,
) -> Tensor:
    """Whole-body centre of mass and its time derivatives.

    Populates ``data.com_position``, ``data.com_velocity``,
    ``data.com_acceleration`` (each shape ``(B..., 3)``) when the
    corresponding input is non-None.
    """

def compute_centroidal_map(model, data, q) -> Tensor:
    """Centroidal momentum matrix A_g(q) ∈ (B..., 6, nv).
    Populates ``data.centroidal_momentum_matrix``."""

def compute_centroidal_momentum(model, data, q, v) -> Tensor:
    """h_g = A_g(q) v ∈ (B..., 6). Populates ``data.centroidal_momentum``."""

def ccrba(model, data, q, v) -> tuple[Tensor, Tensor]:
    """Centroidal CRBA — returns (A_g, h_g)."""
```

## Derivatives

These are the high-value functions in Pinocchio's `algorithm/derivatives/`
directory and are what make analytic gradients of dynamics cheap
enough for DDP / iLQR. Same leading-batch convention.

```python
def compute_rnea_derivatives(
    model, data, q, v, a, fext=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (∂τ/∂q, ∂τ/∂v, ∂τ/∂a = M) each (B..., nv, nv)."""

def compute_aba_derivatives(
    model, data, q, v, tau, fext=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (∂a/∂q, ∂a/∂v, ∂a/∂τ = M^{-1}) each (B..., nv, nv)."""

def compute_crba_derivatives(model, data, q) -> Tensor:
    """Return ∂M/∂q ∈ (B..., nv, nv, nv)."""

def compute_centroidal_dynamics_derivatives(model, data, q, v, a) -> ...:  # stub
    ...
```

The current bodies of `compute_*_derivatives` use
`torch.autograd.functional.jacobian` through the differentiable RNEA
/ ABA / CRBA passes. This ships exact gradients to fp64 round-off
and is what `gradcheck` runs against. The analytic Carpentier–Mansard
recursion is future work; replacing the autograd implementations with
the analytic forms is a drop-in change because the call sites stay
the same.

## The `JointModel` dynamics hooks

The kinematic Protocol covered in {doc}`joints_bodies_frames` carries
two additional methods that RNEA needs:

```python
class JointModel(Protocol):
    # ... existing kinematic surface ...

    def joint_bias_acceleration(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor:                 # (B..., 6) c_J
        """Joint's spatial bias acceleration in its own frame.
        Default implementation returns zeros — correct for revolute,
        prismatic, and free-flyer joints."""

    def joint_motion_subspace_derivative(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor:                 # (B..., 6, nv_j) Ṡ_J
        """Derivative of the motion subspace. Default: zeros (correct
        for joints whose S_J is constant in q)."""
```

For revolute, prismatic, and free-flyer joints these are identically
zero and the defaults apply. For spherical, anatomical, or coupled
joints they are non-zero and the joint module overrides them. The
test `tests/dynamics/test_rnea_coupled_joint.py` exercises the
override on a synthetic coupled joint plus the standard kinds.

## The three-layer action model (Crocoddyl-style)

Above the core dynamics algorithms, BetterRobot ships a layered
integration / cost framework modelled after Crocoddyl's
`DifferentialActionModel` / `IntegratedActionModel` / `ActionModel`
split:

```
dynamics/action/
├── differential.py   # continuous-time dynamics model
├── integrated.py     # discrete-time wrapper (Euler / RK4 / symplectic)
└── action.py         # what the optimal-control solver sees
```

```python
@dataclass
class DifferentialActionModel:
    """Continuous-time dynamics + per-knot cost.

    Responsibilities:
      - Define ẋ = f(x, u)
      - Compute Fx, Fu (Jacobians of f)
      - Evaluate cost r(x, u) and its gradient / Hessian
    """
    model: Model
    state_manifold: "StateManifold"
    cost_stack: "CostStack"

    def calc(self, data: "ActionData", x: Tensor, u: Tensor) -> None: ...
    def calc_diff(self, data: "ActionData", x: Tensor, u: Tensor) -> None: ...

class DifferentialActionModelFreeFwd(DifferentialActionModel):
    """Forward dynamics with no contacts — calls aba + aba_derivatives."""

@dataclass
class IntegratedActionModelEuler(ActionModel):
    differential: DifferentialActionModel
    dt: float
    with_cost_residual: bool = True

    def calc(self, data, x, u) -> None:
        """x_{k+1} = state_manifold.integrate(x, dt * f(x, u))."""
    def calc_diff(self, data, x, u) -> None: ...
```

These data classes and signatures exist; the calc bodies are
skeletons that raise `NotImplementedError` pointing at the roadmap.
DDP / iLQR solvers that build on them are out of scope for v1.

## `StateManifold` — the Crocoddyl lesson

The state of a rigid-body system is `(q, v)` where `q` lives on a
manifold (SE(3) × SO(3) × ℝ^k for a free-flyer robot). Crocoddyl
handles this with a `StateAbstractTpl` that provides `integrate`,
`diff`, `Jintegrate`, `Jdiff` on the state manifold. We adopt the
same pattern:

```python
class StateMultibody:
    """State manifold of a rigid-body system.

    Total configuration = (q, v).
    nx  = nq + nv       (representation dim)
    ndx = 2 * nv        (tangent dim)

    Uses model.integrate / model.difference internally — which
    themselves dispatch through the per-joint JointModel implementations.
    """
    def __init__(self, model: Model): ...
    nx: int
    ndx: int

    def zero(self) -> Tensor: ...
    def integrate(self, x: Tensor, dx: Tensor) -> Tensor: ...
    def diff(self, x0: Tensor, x1: Tensor) -> Tensor: ...
    def jacobian_integrate(self, x, dx) -> tuple[Tensor, Tensor]: ...
    def jacobian_diff(self, x0, x1) -> tuple[Tensor, Tensor]: ...
```

`StateMultibody` defers to the per-joint `JointModel.integrate` /
`JointModel.difference` routines — the dispatch table that already
covers free-flyer and spherical retraction is what makes the state
manifold cheap to assemble.

## Integrators

```python
def integrate_q(model: Model, q: Tensor, v: Tensor, dt: float) -> Tensor:
    """Retract q by dt * v via model.integrate (a.k.a. q ⊕ dt*v)."""

def semi_implicit_euler(model, data, q, v, tau, dt, *, fext=None): ...   # stub
def symplectic_euler   (model, data, q, v, tau, dt, *, fext=None): ...   # stub
def rk4                (model, data, q, v, tau, dt, *, fext=None): ...   # stub
```

`integrate_q` is implementable purely with kinematic machinery (the
per-joint `JointModel.integrate`) and is live. The full physics
integrators wait for stable contact handling.

## Forward vs backward ownership

Each algorithm owns both its forward and backward kernels.
`rnea.forward` calls Featherstone's two-pass recursion; `rnea.backward`
(wrapped via `torch.autograd.Function.apply`) calls
`compute_rnea_derivatives`. The current backward path uses
`torch.autograd.functional.jacobian` through the differentiable
forward; the future analytic backward (Carpentier–Mansard) will
replace it kernel-for-kernel without changing the public surface.

## Sharp edges

- **Naming.** `Data` fields are `mass_matrix`, `bias_forces`,
  `centroidal_momentum_matrix`, `com_position` — not `M`, `nle`,
  `Ag`, `com`. The {doc}`/conventions/naming` table has the full
  rename.
- **Centroidal frame.** `compute_centroidal_map` returns a Jacobian
  expressed at the COM, not at the root joint. CCRBA pairs it with
  the centroidal momentum.
- **Stub guard.** Calling `compute_minverse`, `compute_coriolis_matrix`,
  or any of the higher-order integrators (`semi_implicit_euler`,
  `symplectic_euler`) raises `NotImplementedError` with a pointer to
  {doc}`/reference/roadmap`.
- **Floating-base RNEA.** The first 6 columns of every joint-space
  Jacobian / mass matrix correspond to the free-flyer base when
  `model.joint_models[1] = JointFreeFlyer`. There is no "stripped"
  variant for the actuated subspace — slice the array if you need it.

## Where to look next

- {doc}`kinematics` — how RNEA / ABA / CRBA consume
  `joint_pose_world` and `joint_jacobians` from `Data`.
- {doc}`/reference/roadmap` — what is currently stubbed and where.
- {doc}`/conventions/extension` §15 — adding a muscle / actuator
  that contributes to joint-space torque.
