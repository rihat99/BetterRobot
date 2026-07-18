# Dynamics

The dynamics layer is where rigid-body physics meets numerical
discipline. RNEA computes inverse dynamics (`τ = M(q) a + b(q, v) +
g(q)`); ABA computes forward dynamics (`q̈` from `τ`); CRBA computes
the joint-space inertia matrix `M(q)`; CCRBA computes the centroidal
momentum matrix; derivative helpers expose `∂τ/∂q`, `∂a/∂τ`, and
`∂M/∂q` through PyTorch autograd. Analytic rigid-body derivative recursions and
centroidal-dynamics derivatives remain future work.

We kept Pinocchio's algorithm names verbatim (`rnea`, `aba`, `crba`,
`ccrba`) because every textbook on rigid-body dynamics uses them,
and renaming them would force readers to look up the equivalence
every time. We replaced Pinocchio's storage shorthand
(`oMi`, `oMf`, `liMi`, `nle`, `Ag`) with self-describing identifiers
(`joint_pose_world`, `frame_pose_world`, `joint_pose_local`,
`bias_forces`, `centroidal_momentum_matrix`). The names that matter
to the reader of a dynamics formula keep their math shape; the names
that matter only to the call site are spelled out.

The Featherstone passes are live: RNEA, ABA, CRBA, CCRBA, the centroidal map
and momentum, COM position and velocity, and the autograd-derived
`compute_rnea_derivatives`, `compute_aba_derivatives`, and
`compute_crba_derivatives` helpers. The underlying differentiable passes have
gradient coverage. `compute_centroidal_dynamics_derivatives`, COM acceleration,
`compute_minverse`, `compute_coriolis_matrix`, and the three full-physics
integrators (`semi_implicit_euler`, `symplectic_euler`, and `rk4`) are stubs
that raise `NotImplementedError` and are listed in
{doc}`/reference/roadmap`.

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

The forward dynamics algorithms write into one caller-supplied `Data` object
and accept arbitrary leading batch axes. Configuration inputs end in `nq`;
generalized vectors end in `nv`; mass matrices end in `(nv, nv)`. The explicit
derivative helpers return higher-rank Jacobians described below.

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
    """Whole-body centre of mass and optional velocity.

    Populates ``data.com_position`` and, when ``v`` is supplied,
    ``data.com_velocity`` (each shape ``(B..., 3)``). Passing a non-None
    ``a`` currently raises ``NotImplementedError``; COM acceleration is not
    populated.
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

These signatures mirror high-value functions in Pinocchio's
`algorithm/derivatives/` directory. The three implemented helpers are
autograd-derived convenience wrappers, not the future cheap analytic recursions.

```python
def compute_rnea_derivatives(
    model, data, q, v, a, fext=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (∂τ/∂q, ∂τ/∂v, ∂τ/∂a = M).

    Unbatched shapes: (nv, nq), (nv, nv), (nv, nv).
    """

def compute_aba_derivatives(
    model, data, q, v, tau, fext=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (∂a/∂q, ∂a/∂v, ∂a/∂τ = M^{-1}).

    Unbatched shapes: (nv, nq), (nv, nv), (nv, nv).
    """

def compute_crba_derivatives(model, data, q) -> Tensor:
    """Return ∂M/∂q; unbatched shape (nv, nv, nq)."""

def compute_centroidal_dynamics_derivatives(model, data, q, v, a) -> ...:  # stub
    ...
```

The three implemented derivative helpers use
`torch.autograd.functional.jacobian` through the differentiable RNEA, ABA, and
CRBA passes. For inputs that share batch prefix `*B`, this API retains both the
output and input batch axes:

- RNEA: `∂τ/∂q` is `(*B, nv, *B, nq)`; `∂τ/∂v` and `∂τ/∂a` are
  `(*B, nv, *B, nv)`.
- ABA: `∂a/∂q` is `(*B, nv, *B, nq)`; `∂a/∂v` and `∂a/∂τ` are
  `(*B, nv, *B, nv)`.
- CRBA: `∂M/∂q` is `(*B, nv, nv, *B, nq)`.

These are full Jacobians, not collapsed batch-diagonal arrays. Call a helper per
sample when a shape such as `(*B, nv, nq)` is required. The analytic
Carpentier–Mansard recursions remain future work; replacing the autograd
implementations does not require changing their public signatures.

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

Every currently shipped built-in joint uses these zero defaults in its local
joint coordinates. The hooks remain for an extension whose local motion
subspace actually depends on configuration; no in-tree joint currently supplies
a non-zero override. Standard spherical and free-flyer RNEA behavior is checked
against Pinocchio in
`tests/test_pinocchio/test_rnea_advanced_joints.py`; there is no synthetic
coupled-joint override test.

## Optimal-control action models

The earlier Crocoddyl-style action-model skeletons were removed because they
had no internal or consumer callers and their calculation bodies were not
implemented. A future DDP/iLQR milestone should introduce this surface with
an executable contract instead of reviving the old signatures implicitly.

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

## Forward and derivative ownership

``rnea`` is an ordinary differentiable Torch recursion; it does not install a
custom ``torch.autograd.Function`` or a special ``rnea.backward`` kernel.
Autograd records the forward operations normally. The separate
``compute_rnea_derivatives`` helper uses
``torch.autograd.functional.jacobian`` around that differentiable pass. A
future analytic Carpentier–Mansard helper may replace the explicit derivative
calculation without changing the public signature, while ordinary gradients
through ``rnea`` remain owned by Torch autograd.

## Sharp edges

- **Naming.** `Data` fields are `mass_matrix`, `bias_forces`,
  `centroidal_momentum_matrix`, `com_position` — not `M`, `nle`,
  `Ag`, `com`. The {doc}`/conventions/naming` table has the full
  rename.
- **Centroidal frame.** `compute_centroidal_map` returns a Jacobian
  expressed at the COM, not at the root joint. CCRBA pairs it with
  the centroidal momentum.
- **Stub guard.** Calling `compute_minverse`, `compute_coriolis_matrix`,
  `compute_centroidal_dynamics_derivatives`, any full-physics integrator
  (`semi_implicit_euler`, `symplectic_euler`, or `rk4`), or
  `center_of_mass(..., a=...)` raises `NotImplementedError`.
- **Floating-base RNEA.** The first 6 columns of every joint-space
  Jacobian / mass matrix correspond to the free-flyer base when
  `model.joint_models[1] = JointFreeFlyer`. There is no "stripped"
  variant for the actuated subspace — slice the array if you need it.

## Where to look next

- {doc}`kinematics` — the shared model, joint-dispatch, and pose conventions
  used by the dynamics recursions.
- {doc}`/reference/roadmap` — what is currently stubbed and where.
- {doc}`/conventions/extension` §14 — the deferred muscle / actuator sketch
  that contributes to joint-space torque.
