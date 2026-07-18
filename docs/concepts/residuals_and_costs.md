# Residuals and Costs

BetterRobot composes structural residuals as `ResidualItem` values in a
named-block `Problem`. Each item carries its weight and robust grouping; the
problem owns variable blocks, providers, and evaluation. `solve_ik` and
`solve_trajopt` use the same composition surface as direct callers.

This chapter documents the residual library, explicit composition, robust
groups, and temporal declarations. The next chapter ({doc}`solver_stack`)
covers evaluation and the Adam/LM/GN solver lifecycle.

## The `Residual` Protocol

```python
class Residual(Protocol):
    name: str
    reads: tuple[str, ...]
    dim: int

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        """(B..., dim)."""
```

Source: `src/better_robot/optim/blocks/problem.py`.

A residual is a callable object with a stable name, static output dimension,
and declared context reads. It may carry optional analytic and temporal
helpers beside its forward pass. Expensive shared work such as FK is requested
through a provider output in the context rather than recomputed by every item.

### Explicit composition

Residual instances are constructed directly and wrapped in `ResidualItem`.
There is no global name-to-class registry or import-time registration side
effect. Custom residuals implement the `Residual` protocol, including a stable
`name`, and use the same explicit `Problem` composition path as built-ins.

## The shipped residual library

Live, with analytic `.jacobian()`:

| File | Class | dim | Notes |
|------|-------|-----|-------|
| `pose.py` | `PoseResidual` | 6 | Frame-target SE(3) error via `Jr_inv(log_err)` |
| `pose.py` | `PositionResidual` | 3 | Top three rows of the frame Jacobian |
| `pose.py` | `OrientationResidual` | 3 | Bottom three rows + `Jr_inv_so3` |
| `limits.py` | `JointPositionLimit` | 2 * nq | Diagonal ±1 / 0 |
| `regularization.py` | `RestResidual` | nv | Manifold difference; approximate identity Jacobian |
| `regularization.py` | `ReferenceTrajectoryResidual` | T * nv | Per-knot manifold reference tracking |
| `contact.py` | `ContactConsistencyResidual` | 3 * K * (T - 1) | Linear velocity of K contact frames |
| `temporal.py` | `TimeIndexedResidual` | inner dim | Generic single-knot wrapper |
| `smoothness.py` | `VelocityResidual` | nv * (T - 2) | Central tangent velocity over time |
| `smoothness.py` | `AccelerationResidual` | nv * (T - 2) | Three-knot tangent acceleration |

Live without an analytic Jacobian (`Problem` uses tangent-space `torch.func`
AD):

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
| `collision.py` | `SelfCollisionResidual`, `WorldCollisionResidual` | Collision primitives and residual evaluation are currently stubbed |
| `limits.py` | `JointVelocityLimit.jacobian`, `JointAccelLimit` | `__call__` works; analytic Jacobian or full body pending |

A residual without an analytic block is differentiated through
`RobotConfig.retract` with `torch.func.jacrev` or `jacfwd`; finite differences
remain an explicit debugging strategy.

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
approximation: current tasks receive the exact tangent AD block.

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
documented small-step approximation. With `target_name=None`, evaluation uses
the captured `q_rest`. With a `target_name`, named-block
evaluation declares that name in `reads` and obtains the live target from the
context; `solve_ik` uses this form with an explicit `Problem` parameter.

### Example — `PoseResidual`

The full analytic-Jacobian path, shown in {doc}`kinematics`, is the
canonical example: stores a `frame_id` and target SE(3); evaluates
`log(T_target⁻¹ ∘ T_frame)` for the residual; multiplies
`Jr_inv(log_err) @ get_frame_jacobian(reference=LOCAL)` for the
Jacobian. Works for any root joint — fixed, free-flyer, or anything
custom — because `get_frame_jacobian` walks the topology.

## Explicit problem composition

`ResidualItem` stores a residual with its scalar or tensor weight, optional
robust kernel, and robust `group_size`. Item order defines residual row order;
variable order defines Jacobian column order. A Python-zero weight skips the
item during evaluation, while omitting an item removes it from the static
problem structure.

`Problem.residual()` returns the weighted raw vector. `Problem.objective()`
applies grouped robust losses, `Problem.gradient()` differentiates that same
objective in reduced tangent coordinates, and `Problem.dense_jacobian()`
assembles the explicit residual Jacobian. Temporal metadata is consumed only
after `Problem` validates the block/time layout;
`Problem.structured_normal()` assembles lower block bands and JVP/VJP
operations.

## Stable `dim` for collision residuals

Collision residuals expose a per-pair output but mix "candidate
pairs" with "active pairs above the margin." To keep LM line-search
and damping stable, the contract is:

- `dim = number_of_candidate_pairs` — stable across iterations.
- Pairs outside the safety margin contribute zero — but the slot
  exists, so the Jacobian has a corresponding row of zeros.
- A future residual may compact work internally, but it must still return the
  declared public shape.
- The active subset may vary internally, but the output dimension remains
  stable and explicit LM assembly remains dense.

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
from better_robot.optim import (
    Problem, ResidualItem, RobotConfig, RobotStateProvider, VarSpec,
)

model = br.load(panda_description.URDF_PATH)
hand_id = model.frame_id("body_panda_hand")

pose = PoseResidual(
    frame_id=hand_id,
    target=target_pose,
    model=model,
    name="pose",
)
limits = JointPositionLimit(model, name="limits")
rest = RestResidual(model, model.q_neutral, name="rest")

problem = Problem(
    vars=(VarSpec("q", (model.nq,), manifold=RobotConfig(model)),),
    residuals=(
        ResidualItem("pose", pose),
        ResidualItem("limits", limits, weight=0.1),
        ResidualItem("rest", rest, weight=0.01),
    ),
    providers=(RobotStateProvider(model),),
)

# Evaluate (manually, for diagnostics)
r = problem.residual({"q": q})
J = problem.dense_jacobian({"q": q})
```

`solve_ik` constructs the same item and problem types internally.
`solve_trajopt` accepts a sequence of `ResidualItem` values and adapts their
residuals to one named temporal variable.

## Sharp edges

- **Names are structural.** Every item name is unique and must match its
  residual's declared name.
- **Weights and robust kernels are separate.** The weight scales residual
  rows; the kernel groups and robustifies them for objective/solver use.
- **Temporal structure is explicit.** Undeclared items make automatic
  temporal LM fall back to dense rather than inferring sparsity from zeros.
- **Selection is explicit.** Omit a residual item when it should not be part of
  a solve; task facades do not maintain active/kind flags.

## Where to look next

- {doc}`solver_stack` — named-block evaluation and the Adam/LM/GN lifecycle.
- {doc}`kinematics` — the analytic Jacobian path that
  `PoseResidual` follows.
- {doc}`/conventions/extension` §1, §6 — recipes for adding a new
  residual or a new robust kernel.
