# Residuals and Costs

BetterRobot currently has two explicit optimisation lanes. Legacy flat-vector
callers compose `Residual` objects in a `CostStack` and pass the stack to a
`LeastSquaresProblem`. Current named-block tasks compose `ResidualItem` objects
in a `Problem`; `solve_trajopt` is a compatibility bridge that adapts active
soft `CostStack` items into those named residual items. The lanes share
residual concepts, but they do not otherwise share a problem type.

This chapter documents the legacy residual protocol and the canonical
optimizer-owned `CostStack`: named, weighted concatenation of active
residuals. The next chapter ({doc}`solver_stack`)
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

# Optional extension, not a requirement of the Residual protocol:
def apply_jac_transpose(
    self,
    state: ResidualState,
    vec: torch.Tensor,
) -> torch.Tensor: ...
```

Source: `src/better_robot/residuals/base.py`.

A residual is a callable object, not a plain function — that is what
lets it carry `.jacobian()` and optional structural helpers next to its
forward pass. The state object passed in carries `(model, data,
variables)`; residuals reach back through `data` (FK has already
been computed) or directly into `variables` for things like joint
limits.

### Explicit composition

Residual instances are constructed directly and added to a `CostStack`.
There is no global name-to-class registry or import-time registration side
effect. Custom residuals implement the `Residual` protocol, including a stable
`name`, and use the same explicit composition path as built-ins.

## The shipped residual library

Live, with analytic `.jacobian()`:

| File | Class | dim | Notes |
|------|-------|-----|-------|
| `pose.py` | `PoseResidual` | 6 | Frame-target SE(3) error via `Jr_inv(log_err)` |
| `pose.py` | `PositionResidual` | 3 | Top three rows of the frame Jacobian |
| `pose.py` | `OrientationResidual` | 3 | Bottom three rows + `Jr_inv_so3` |
| `limits.py` | `JointPositionLimit` | 2 * nq | Diagonal ±1 / 0 |
| `regularization.py` | `RestResidual` | nv | Manifold difference; approximate identity Jacobian |
| `reference_trajectory.py` | `ReferenceTrajectoryResidual` | varies | Time-indexed knot tracking |
| `contact.py` | `ContactConsistencyResidual` | 6 * n_contacts | Holonomic contact constraint |
| `time_indexed.py` | `TimeIndexedResidual` | varies | Generic time-axis wrapper |
| `smoothness.py` | `Velocity5pt` / `Accel5pt` | nv * (T - k) | Tridiagonal over time axis |
| `velocity.py` | `VelocityResidual` | nv | Joint-space velocity tracking |
| `acceleration.py` | `AccelerationResidual` | nv | Joint-space acceleration tracking |

Live without an analytic Jacobian (their `.jacobian()` returns `None` and
named-block `Problem` uses tangent-space `torch.func` AD; the legacy
`CostStack` lane uses unbatched central finite differences:

| File | Class | dim | Notes |
|------|-------|-----|-------|
| `human.py` | `SwingTwistLimitResidual` | 3 * selected joints | Piecewise swing/twist limits; decomposition is singular at pure-pi swing |
| `regularization.py` | `JointRotationPrior` | nv | Exact per-joint weighted manifold difference; no small-angle identity Jacobian claim |

Stubs (raise `NotImplementedError`; signatures pinned):

| File | Class | Notes |
|------|-------|-------|
| `smoothness.py` | `JerkResidual` | Third-derivative smoothness |
| `manipulability.py` | `YoshikawaResidual` | det(J Jᵀ)^½ |
| `regularization.py` | `NullspaceResidual` | Project gradient onto null space |
| `collision.py` | `SelfCollisionResidual`, `WorldCollisionResidual` | Live geometry; residual side stubbed |
| `limits.py` | `JointVelocityLimit.jacobian`, `JointAccelLimit` | `__call__` works; analytic Jacobian or full body pending |

A residual with no analytic `.jacobian()` simply returns `None`; the
legacy stack dispatches to unbatched central finite differences at a cost of
`2 * nv + 1` residual evaluations. Named-block `Problem` instead differentiates
through `RobotConfig.retract` with `torch.func.jacrev` or `jacfwd`.

### Temporal structure declarations

`reads` answers which named blocks a residual depends on. For a variable
declared with `VarSpec(..., time_axis=0)`, an optional
`temporal_structure(variable_name)` hook can refine that dependency with an
optimizer-independent value:

```python
TemporalPattern(
    rows=T - 2,
    row_width=model.nv,
    row_origin=1,
    offsets=(-1, 0, 1),
)
```

For row group `r`, offset `o` names knot `r + row_origin + o`. Rows and widths
are positive, offsets are sorted/unique/non-empty, every knot is in range, and
`rows * row_width == dim`. The optional numeric hook
`temporal_jacobian_blocks(ctx, variable_name)` returns one tensor per offset
with shape `(B..., rows, row_width, reduced_width_per_knot)`. Those tensors do
not include `ResidualItem.weight` or robust IRLS row scaling; `Problem`
applies both once.

Velocity declares offsets `(-1, +1)`, acceleration `(-1, 0, +1)`,
time-indexed and reference-trajectory terms `(0,)`, and contact consistency
`(0, +1)`. Missing declarations keep automatic LM dense. A declaration
without numeric blocks is eligible only for the explicit normal-operator
fallback. Structure is static: a zero weight never makes an undeclared item
eligible, and numerical zeros are never inspected.

### Human spherical-joint limits and priors

`SwingTwistLimitResidual` selects spherical joints and returns three one-sided
rows per joint: swing above its cone limit, twist below its lower limit, and
twist above its upper limit. The twist axis is joint-local, quaternion sign is
folded across the double cover, and limits are non-wrapping intervals inside
`(-pi, pi)`. Swing/twist decomposition cannot assign a unique twist to a pure
180-degree swing; the implementation chooses zero twist in a tiny numerical
neighbourhood and deliberately does not advertise a globally analytic
Jacobian.

`JointRotationPrior` evaluates `model.difference(q_mean, q)` and scales either
each joint tangent slice or each individual tangent coordinate. It differs
from `RestResidual` by declining the latter's small-step identity-Jacobian
approximation: current named-block tasks receive the exact tangent AD block.

### Vision and padded point-cloud residuals

The M4 vision pack is named-block-native. It uses fixed event shapes and
arbitrary leading execution batches:

- `ProjectionResidual(model, point_ids, K, extrinsics, target_px, ...)`
  projects body, marker, or site frame-table rows through a camera-thin
  world-to-camera transform. Confidence and validity are per point. Its
  complete analytic `q` block composes the pinhole derivative, camera rotation,
  and frame Jacobian. Put `GemanMcClure` on the surrounding `ResidualItem` with
  `group_size=2`; robust weighting is not embedded in the residual.
- `MaskedChamferResidual` consumes padded source/target point tensors and bool
  validity masks. It emits source-to-target and, by default,
  target-to-source nearest distances. Correspondence indices are detached;
  gradients flow through the selected distances only.
- `SceneSDFProvider` performs one chunked detached nearest-neighbour pass for a
  padded query cloud and oriented scene cloud. Its `SceneSDFResult` feeds
  `ScenePenetrationResidual`, `SceneAttractionResidual`, and
  `SceneClearanceResidual`, so a `Problem` containing all three does not repeat
  the nearest-neighbour work.

The ragged-data convention is deliberately small: pad to a fixed count and
carry a same-prefix boolean validity mask. Empty frames yield finite zero rows.
There is no camera class, visibility policy, keypoint mapping, contact-label
heuristic, or rendering abstraction in this pack.

### Example — `RestResidual`

```python
class RestResidual:
    name = "rest"
    reads = ("q",)

    def __init__(self, model, q_rest, *, weight=1.0, name="rest",
                 target_name=None):
        self.model = model
        self.q_rest = q_rest
        self.target_name = target_name
        self.reads = ("q", target_name) if target_name is not None else ("q",)
        self.weight = weight
        self.dim = model.nv

    def __call__(self, value: ResidualState | Mapping[str, Any]) -> torch.Tensor:
        model, q = _residual_model_q(value, model=self.model)
        q_rest = self.q_rest
        if self.target_name is not None and not isinstance(value, ResidualState):
            q_rest = value[self.target_name]
        q_rest = q_rest.to(device=q.device, dtype=q.dtype)
        if q.ndim > 1 and q_rest.ndim == 1:
            q_rest = q_rest.expand_as(q)
        return model.difference(q_rest, q) * self.weight

    def jacobian(self, value: ResidualState | Mapping[str, Any]) -> torch.Tensor:
        _, q = _residual_model_q(value, model=self.model)
        I = torch.eye(self.dim, device=q.device, dtype=q.dtype)
        return I.expand(*q.shape[:-1], self.dim, self.dim) * self.weight
```

Source: `src/better_robot/residuals/regularization.py`.

The residual is `model.difference(q_rest, q)`, not raw configuration-space
subtraction: free-flyer and spherical coordinates therefore produce the
correct `nv`-dimensional tangent displacement. The identity Jacobian is the
documented small-step approximation. With `target_name=None`, both legacy and
named-block calls use the captured `q_rest`. With a `target_name`, named-block
evaluation declares that name in `reads` and obtains the live target from the
context; `solve_ik` uses this form with an explicit `Problem` parameter.

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

    A dict keyed by name, with scalar weights and per-item on/off flags.
    The stack concatenates the weighted residuals of all active items
    into a single vector. ``slice_map()`` reports their current slices
    in insertion order.

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
    def gradient(self, state: ResidualState) -> torch.Tensor: ...
```

Source: `src/better_robot/optim/cost_stack.py`.

The canonical definitions and qualified imports are
`better_robot.optim.CostStack`, `better_robot.optim.CostItem`, and
`better_robot.optim.CostKind`. The root `better_robot.CostStack` export and
`better_robot.costs.stack` forward to the same class object for compatibility;
they are not separate implementations.

`stack.residual()` evaluates every active residual, applies its scalar weight,
and uses `torch.cat(..., dim=-1)` to build the result. `stack.jacobian()`
dispatches analytic or central-finite-difference evaluation per residual,
applies the same weights, and concatenates along the residual-row dimension.
Both outputs are assembled anew on each call. `slice_map()` likewise computes
the active layout on demand; the stack does not own a persistent flat buffer.

`CostStack` has no snapshot/restore API. Named-block staged solves use
`better_robot.optim.Phase` and `run_phases()`, which build phase-specific
`Problem` values rather than mutating and restoring this legacy stack.

### Three concepts that look similar but are not

`active` (a structural inclusion flag), `weight` (a scalar
multiplier), and a solver's robust `kernel` are independent. `weight = 0` is
**not** equivalent to `active = False`: a zero-weight item still contributes
rows to the concatenated residual and Jacobian, while an inactive item is
omitted and changes `total_dim()` and `slice_map()`. The `kind` field is
currently metadata; `CostStack` evaluation does not enforce constraints.

### Gradient path

`CostStack.gradient()` sums each active item's contribution to
`Jᵀ r`. If a residual implements `apply_jac_transpose(state, vec)`, the
stack uses that hook; otherwise it materialises that residual's Jacobian and
multiplies. The item weight is squared, matching the gradient of
`0.5 * ||stack.residual(state)||²`.

The legacy `CostStack` composer carries no symbolic sparsity metadata and
`CostStack.jacobian()` always returns a dense concatenated tensor. Temporal
metadata lives on the residual objects and is consumed only after a named
`Problem` has validated the block/time layout; `Problem.structured_normal()`
assembles the lower block bands and JVP/VJP operations.

## Stable `dim` for collision residuals

Collision residuals expose a per-pair output but mix "candidate
pairs" with "active pairs above the margin." To keep LM line-search
and damping stable, the contract is:

- `dim = number_of_candidate_pairs` — stable across iterations.
- Pairs outside the safety margin contribute zero — but the slot
  exists, so the Jacobian has a corresponding row of zeros.
- A future residual may compact work internally, but when used with the
  current stack it must still return the declared public shape.
- The active subset may vary internally, but the output dimension remains
  stable and legacy LM assembly remains dense.

This is the reserved shape contract for a future
`SelfCollisionResidual`. That residual still raises `NotImplementedError`;
collision-aware `solve_ik` integration is roadmap work.

## Mapping to current code

```python
# Construction
import better_robot as br
from robot_descriptions import panda_description
from better_robot.residuals.pose          import PoseResidual
from better_robot.residuals.limits        import JointPositionLimit
from better_robot.residuals.regularization import RestResidual
from better_robot.optim                    import CostStack

model = br.load(panda_description.URDF_PATH)
hand_id = model.frame_id("body_panda_hand")

stack = CostStack()
stack.add("pose",   PoseResidual(frame_id=hand_id, target=target_pose))
stack.add("limits", JointPositionLimit(model), weight=0.1)
stack.add("rest",   RestResidual(model, model.q_neutral), weight=0.01)

# Evaluate (manually, for diagnostics)
state = br.residuals.ResidualState(
    model=model,
    data=br.forward_kinematics(model, q, compute_frames=True),
    variables=q,
)
r = stack.residual(state)            # (B..., total_dim)
J = stack.jacobian(state)            # (B..., total_dim, nx)
```

This is the legacy flat evaluation path for direct callers. `solve_ik`
constructs named-block `ResidualItem`s directly. `solve_trajopt` accepts the
same `CostStack` composition surface but adapts only its active soft items into
a named temporal `Problem`; constraint-kind items and legacy optimizer objects
are rejected rather than silently reinterpreted.

## Sharp edges

- **Active vs weight.** Setting `weight=0` does not remove the
  residual from the Jacobian shape; setting `active=False` does.
  Mutations remain until callers explicitly reset them.
- **`apply_jac_transpose` is optional.** Without that method,
  `CostStack.gradient()` materialises the per-residual Jacobian and computes
  `J.mT @ r`.
- **`CostStack.jacobian()` stays dense.** Temporal structure is a residual
  declaration consumed by named `Problem`, not a different legacy stack
  return type. Undeclared items make automatic temporal LM fall back to dense.
- **Duplicate names are rejected.** Calling `add()` with an existing name
  raises `ValueError`; remove or update the existing item explicitly.

## Where to look next

- {doc}`solver_stack` — `LeastSquaresProblem` and the four
  pluggable axes that consume the `CostStack` Jacobian.
- {doc}`kinematics` — the analytic Jacobian path that
  `PoseResidual` follows.
- {doc}`/conventions/extension` §1, §6 — recipes for adding a new
  residual or a new robust kernel.
