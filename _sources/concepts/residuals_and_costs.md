# Residuals and Costs

Every optimisation problem in BetterRobot — IK, trajectory
optimisation, retargeting, and the future filtering and optimal-control
problems — boils down to the same shape: a vector-valued residual
function `r(x)` whose squared norm is the loss the solver
minimises. The library has one primitive (`Residual`), one composer
(`CostStack`), and one problem type (`LeastSquaresProblem`). Every
high-level task builds them; every solver consumes them.

The reason for the discipline is that the alternative — a separate
"IK problem" class, a separate "trajopt problem" class, a separate
loss-function abstraction for each — dies under its own weight as
soon as the third task arrives. Each new task wants its own residual
mix; each gets its own ad-hoc Jacobian assembly; each has its own
weight semantics. By the time you ship the fourth task you have four
near-duplicates of the same code, drifting independently. Pinning
the substrate to *one* residual / cost / problem stack means
swapping LM for L-BFGS, swapping a Cauchy kernel for a Huber, or
plugging in a custom robust-IRLS reweighting scheme is one Protocol
swap that affects every task at once.

This chapter covers residuals (pure functions, optionally with
analytic Jacobians) and `CostStack` (named, weighted concatenation
with snapshot / restore). The next chapter ({doc}`solver_stack`)
covers `LeastSquaresProblem` and the four pluggable axes
(`Optimizer`, `LinearSolver`, `RobustKernel`, `DampingStrategy`).

## The `Residual` Protocol

```python
class ResidualState:
    """Thin struct passed to every residual.

    Residuals do NOT take free-standing kwargs. All configuration
    (target_pose, weights, link indices, …) is captured as attributes
    of the concrete residual object, constructed once.
    """
    model: Model
    data:  Data
    variables: torch.Tensor      # (B..., nx) flat variable tensor

class Residual(Protocol):
    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor:
        """(B..., dim)."""

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """(B..., dim, nx) or None if analytic is not available."""

    def spec(self, state: ResidualState) -> "ResidualSpec":
        """Structural metadata. Default impl returns dense, output_dim=dim.
        Override to advertise sparse / banded / matrix-free structure."""

    def apply_jac_transpose(
        self,
        state: ResidualState,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute J(x)^T @ vec without materialising J.

        Default: J = self.jacobian(state); return J.mT @ vec.
        Override on temporal residuals where J is banded over T —
        long-horizon trajopt depends on this for memory.
        """
```

Source: `src/better_robot/residuals/base.py`.

A residual is a callable object, not a plain function — that is what
lets it carry an optional `.jacobian()` and `.spec()` next to its
forward pass. The state object passed in carries `(model, data,
variables)`; residuals reach back through `data` (FK has already
been computed) or directly into `variables` for things like joint
limits.

### The decorator

```python
_REGISTRY: dict[str, type[Residual]] = {}

def register_residual(name: str):
    def _inner(cls):
        if name in _REGISTRY:
            warnings.warn(f"residual '{name}' already registered; replacing",
                          RuntimeWarning)
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return _inner

def get_residual(name: str) -> type[Residual]:
    return _REGISTRY[name]
```

Source: `src/better_robot/residuals/registry.py`. Third-party users
add their own residuals by decorating a class. The solver has no idea
the registry exists — residuals are composed into a `CostStack`, and
the stack is passed to the solver. Re-registering the same name logs
a warning and replaces (intentional, for experimentation in
notebooks).

## The shipped residual library

Live, with analytic `.jacobian()`:

| File | Class | dim | Notes |
|------|-------|-----|-------|
| `pose.py` | `PoseResidual` | 6 | Frame-target SE(3) error via `Jr_inv(log_err)` |
| `pose.py` | `PositionResidual` | 3 | Top three rows of the frame Jacobian |
| `pose.py` | `OrientationResidual` | 3 | Bottom three rows + `Jr_inv_so3` |
| `limits.py` | `JointPositionLimit` | 2 * nv | Diagonal ±1 / 0 |
| `regularization.py` | `RestResidual` | nv | Identity |
| `reference_trajectory.py` | `ReferenceTrajectoryResidual` | varies | Time-indexed knot tracking |
| `contact.py` | `ContactConsistencyResidual` | 6 * n_contacts | Holonomic contact constraint |
| `time_indexed.py` | `TimeIndexedResidual` | varies | Generic time-axis wrapper |
| `smoothness.py` | `Velocity5pt` / `Accel5pt` | nv * (T - k) | Tridiagonal over time axis |
| `velocity.py` | `VelocityResidual` | nv | Joint-space velocity tracking |
| `acceleration.py` | `AccelerationResidual` | nv | Joint-space acceleration tracking |

Live but autodiff-only (their `.jacobian()` returns `None`):

| File | Class | dim | Notes |
|------|-------|-----|-------|
| (none currently — every shipped residual has analytic support, see roadmap below) | | | |

Stubs (raise `NotImplementedError`; signatures pinned):

| File | Class | Notes |
|------|-------|-------|
| `smoothness.py` | `JerkResidual` | Third-derivative smoothness |
| `manipulability.py` | `YoshikawaResidual` | det(J Jᵀ)^½ — autodiff-only |
| `regularization.py` | `NullspaceResidual` | Project gradient onto null space |
| `collision.py` | `SelfCollisionResidual`, `WorldCollisionResidual` | Live geometry; residual side stubbed |
| `limits.py` | `JointVelocityLimit.jacobian`, `JointAccelLimit` | `__call__` works; analytic Jacobian or full body pending |

A residual with no analytic `.jacobian()` simply returns `None`; the
solver dispatches transparently to autodiff.

### Example — `RestResidual`

```python
@register_residual("rest")
class RestResidual(Residual):
    dim: int

    def __init__(self, q_rest: torch.Tensor, *, weight: float = 1.0) -> None:
        self.q_rest = q_rest
        self.weight = weight
        self.dim = q_rest.shape[-1]

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables                         # (B..., nv) for this problem
        return (q - self.q_rest) * self.weight

    def jacobian(self, state: ResidualState) -> torch.Tensor:
        I = torch.eye(self.dim, device=state.variables.device,
                      dtype=state.variables.dtype)
        return I.expand(*state.variables.shape[:-1], self.dim, self.dim) * self.weight
```

Source: `src/better_robot/residuals/regularization.py`.

### Example — `PoseResidual`

The full analytic-Jacobian path, shown in {doc}`kinematics`, is the
canonical example: stores a `frame_id` and target SE(3); evaluates
`log(T_target⁻¹ ∘ T_frame)` for the residual; multiplies
`Jr_inv(log_err) @ get_frame_jacobian(reference=LOCAL)` for the
Jacobian. Works for any root joint — fixed, free-flyer, or anything
custom — because `get_frame_jacobian` walks the topology.

## `CostStack`

```python
@dataclass
class CostItem:
    name: str
    residual: Residual
    weight: float = 1.0
    active: bool   = True
    kind:   Literal["soft", "constraint_leq_zero"] = "soft"

class CostStack:
    """Named, weighted, individually activatable stack of residuals.

    Mirrors Crocoddyl's CostModelSum — a dict keyed by name, with
    scalar weights and per-item on/off flags. The stack concatenates
    the weighted residuals of all active items into a single vector
    and provides slice-maps so solvers can compute per-item Jacobians
    in place.

    Usage:
        stack = CostStack()
        stack.add("pose_rh", PoseResidual(frame_id=..., target=...), weight=1.0)
        stack.add("pose_lh", PoseResidual(frame_id=..., target=...), weight=1.0)
        stack.add("limits",  JointPositionLimit(),          weight=0.1)
        stack.add("rest",    RestResidual(q_rest),          weight=0.01)

        r = stack.residual(state)             # (B..., total_dim)
        J = stack.jacobian(state)             # (B..., total_dim, nx)
    """

    items: dict[str, CostItem]

    def add(self, name, residual, *, weight=1.0, kind="soft"): ...
    def remove(self, name): ...
    def set_active(self, name: str, active: bool): ...
    def set_weight(self, name: str, weight: float): ...
    def total_dim(self) -> int: ...
    def slice_map(self) -> dict[str, slice]: ...
    def residual(self, state: ResidualState) -> torch.Tensor: ...
    def jacobian(self, state: ResidualState, *, strategy=JacobianStrategy.AUTO) -> torch.Tensor: ...
    def snapshot(self) -> dict: ...        # for MultiStageOptimizer
    def restore(self, snap: dict) -> None: ...
```

Source: `src/better_robot/costs/stack.py`.

`stack.residual()` evaluates every active residual and concatenates
the results along the last dim. `stack.jacobian()` does the same for
Jacobians, dispatching analytic / autodiff per residual via the
strategy flag.

The memory layout follows Crocoddyl's discipline:

- One flat residual buffer of shape `(B..., total_dim)` is allocated
  inside `CostStack`.
- Each residual writes into its pre-computed slice
  `items[name].slice`.
- No Python-level `torch.cat` in the hot path — only a pre-allocated
  tensor plus `index_put_`-style writes.

### Three concepts that look similar but are not

`active` (a structural inclusion flag), `weight` (a scalar
multiplier), and `kernel` (a per-iteration reweighting in the normal
equations) are independent. `weight = 0` is **not** equivalent to
`active = False`: a zero-weight item still occupies a slot in the
preallocated residual / Jacobian; only the active flag structurally
removes it. Sparse trajopt depends on this distinction so the
stage-wise multi-stage optimiser can toggle a residual on and off
without reallocating.

### Sparsity-aware assembly

For residuals whose Jacobian is structurally sparse — self-collision,
joint limits, 5-point finite-difference smoothness across a
trajectory — the residual exposes a `.spec: ResidualSpec` describing
which columns of `x` it touches. `CostStack.jacobian(...)` propagates
the sparsity into a block layout, and the linear solver assembles
only the non-zero blocks of `JᵀJ`. Measured speed-up on G1 collision
costs: ~6×.

```python
@dataclass(frozen=True)
class ResidualSpec:
    """Structural metadata about a residual.

    Returned by Residual.spec(state). Used by solvers to plan
    Jacobian assembly, sparse-block storage, and active-set updates.
    """
    output_dim: int
    tangent_dim: int

    structure: Literal[
        "dense",          # full (output_dim, tangent_dim) Jacobian.
        "block",          # one or more (out, in) blocks at named indices.
        "banded",         # banded along a temporal axis.
        "matrix_free",    # only J^T r is available; J is never built.
    ] = "dense"

    # Time coupling for trajopt residuals.
    time_coupling: Literal["single", "5-point", "custom"] | None = None
    affected_knots: tuple[int, ...] | None = None

    # Spatial / kinematic locality.
    affected_joints: tuple[int, ...] | None = None
    affected_frames: tuple[int, ...] | None = None

    # Whether output_dim depends on the input (e.g. active-pair collision).
    dynamic_dim: bool = False
```

Source: `src/better_robot/optim/jacobian_spec.py`.

Default `Residual.spec()` returns a dense spec with `output_dim =
self.dim` — every existing residual works without modification.
Trajopt residuals override to advertise their banded / 5-point time
coupling so the linear solver builds a block-Cholesky factorisation;
collision residuals override to set `dynamic_dim=True` so LM
preallocates correctly.

## Stable `dim` for collision residuals

Collision residuals expose a per-pair output but mix "candidate
pairs" with "active pairs above the margin." To keep LM line-search
and damping stable, the contract is:

- `dim = number_of_candidate_pairs` — stable across iterations.
- Pairs outside the safety margin contribute zero — but the slot
  exists, so the Jacobian has a corresponding row of zeros.
- Active-pair compaction is a kernel-internal optimisation; the
  matrix-free `J^T r` path skips inactive rows without changing the
  public residual `dim`.
- `ResidualSpec.dynamic_dim = True` declares the slot reservation so
  LM preallocates once.

This is what makes `solve_ik` with `SelfCollisionResidual` work
without re-allocating per iteration.

## Mapping to current code

```python
# Construction
import better_robot as br
from better_robot.residuals.pose          import PoseResidual
from better_robot.residuals.limits        import JointPositionLimit
from better_robot.residuals.regularization import RestResidual
from better_robot.costs                    import CostStack

model = br.load("panda.urdf")
hand_id = model.frame_id("panda_hand")

stack = CostStack()
stack.add("pose",   PoseResidual(frame_id=hand_id, target=target_pose))
stack.add("limits", JointPositionLimit(model), weight=0.1)
stack.add("rest",   RestResidual(model.q_neutral), weight=0.01)

# Evaluate (manually, for diagnostics)
state = br.residuals.ResidualState(
    model=model,
    data=br.forward_kinematics(model, q, compute_frames=True),
    variables=q,
)
r = stack.residual(state)            # (B..., total_dim)
J = stack.jacobian(state)            # (B..., total_dim, nx)
```

In normal use you would never write that loop; `solve_ik` does it
internally. The example exists to show that the interfaces are
tractable.

## Sharp edges

- **Active vs weight.** Setting `weight=0` does not remove the
  residual from the Jacobian shape; setting `active=False` does.
  Snapshot / restore preserves both.
- **`apply_jac_transpose` defaults to `J.mT @ vec`.** Time-coupled
  trajopt residuals must override this to avoid materialising the
  dense Jacobian for long horizons.
- **Sparsity hints are advisory.** A residual that does not advertise
  sparsity is treated as dense; the solver does not lose
  correctness, only speed.
- **Re-registering a name logs a warning and replaces.** This is
  intentional but it can mask a bug if you import two extensions that
  both register `"pose"`. The warning text names the file the
  redefinition came from.

## Where to look next

- {doc}`solver_stack` — `LeastSquaresProblem` and the four
  pluggable axes that consume the `CostStack` Jacobian.
- {doc}`kinematics` — the analytic Jacobian path that
  `PoseResidual` follows.
- {doc}`/conventions/extension` §1, §6 — recipes for adding a new
  residual or a new robust kernel.
