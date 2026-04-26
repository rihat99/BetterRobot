# Tasks — IK, Trajectory Optimisation, Retargeting

`tasks/` is the thinnest layer in the library, and that is on
purpose. Every entry point here is a short facade that does the
same five things in order:

1. Pick frames / links / targets the user gave by name.
2. Build a `CostStack` from residuals in `residuals/`.
3. Wrap it in a `LeastSquaresProblem`.
4. Call an `Optimizer`.
5. Return a clean result type.

There is no Jacobian code here, no solver loop, no fixed-vs-floating
branch. All of that belongs one layer down. What lives in `tasks/`
is the convenience translation between "give me an IK solution to
this pose" and the substrate that already knows how to do it. If a
user wants finer control, they drop down to `optim/` directly; the
tasks layer is pure ergonomics on top of a fully-public solver
stack.

## Inverse kinematics

```python
def solve_ik(
    model: Model,
    targets: dict[str, torch.Tensor],         # {frame_name: (7,) SE3 pose}
    *,
    initial_q: torch.Tensor | None = None,
    cost_cfg: IKCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    robot_collision: "RobotCollision | None" = None,
) -> IKResult:
    """Whole-body inverse kinematics for one or more frame targets.

    There is one code path. Floating-base robots are those whose
    model.joint_models[1] is JointFreeFlyer — the solver does not need
    to know. All 7 quaternion-xyz values of the free-flyer live inside q.
    """
```

Source: `src/better_robot/tasks/ik.py`.

### `IKCostConfig`

```python
@dataclass(frozen=True)
class IKCostConfig:
    pos_weight:       float = 1.0
    ori_weight:       float = 1.0
    pose_weight:      float = 1.0
    limit_weight:     float = 0.1
    rest_weight:      float = 0.01
    collision_margin: float = 0.02
    collision_weight: float = 1.0
    q_rest: torch.Tensor | None = None        # default: model.q_neutral
```

### Implementation sketch

```python
def solve_ik(model, targets, *, initial_q=None, cost_cfg=None,
             optimizer_cfg=None, robot_collision=None) -> IKResult:
    cost_cfg      = cost_cfg or IKCostConfig()
    optimizer_cfg = optimizer_cfg or OptimizerConfig()
    q0            = initial_q if initial_q is not None else model.q_neutral.clone()

    stack = CostStack()
    for name, target in targets.items():
        fid = model.frame_id(name)
        stack.add(f"pose_{name}",
                  PoseResidual(frame_id=fid, target=target,
                               pos_weight=cost_cfg.pos_weight,
                               ori_weight=cost_cfg.ori_weight),
                  weight=cost_cfg.pose_weight)
    stack.add("limits", JointPositionLimit(model), weight=cost_cfg.limit_weight)
    stack.add("rest",   RestResidual(cost_cfg.q_rest or model.q_neutral),
              weight=cost_cfg.rest_weight)
    if robot_collision is not None:
        stack.add("self_collision",
                  SelfCollisionResidual(model, robot_collision,
                                        margin=cost_cfg.collision_margin),
                  weight=cost_cfg.collision_weight)

    problem = LeastSquaresProblem(
        cost_stack=stack,
        state_factory=lambda x: ResidualState(
            model=model,
            data=forward_kinematics(model, x, compute_frames=True),
            variables=x,
        ),
        x0=q0,
        lower=model.lower_pos_limit,
        upper=model.upper_pos_limit,
        jacobian_strategy=optimizer_cfg.jacobian_strategy,
    )
    optimizer = _build_optimizer(optimizer_cfg)
    result = optimizer.minimize(problem, max_iter=optimizer_cfg.max_iter, ...)
    return IKResult(q=result.x, residual=result.residual, iters=result.iters,
                    converged=result.converged, model=model)
```

That is the entire task implementation: build a `CostStack`, wrap it
in `LeastSquaresProblem`, hand it to an `Optimizer`, return an
`IKResult`. Every line of solver math lives one layer down in
`optim/`; every line of Jacobian assembly lives one layer further
down in `kinematics/`.

### `IKResult`

```python
@dataclass
class IKResult:
    q: torch.Tensor                            # (B..., nq)
    residual: torch.Tensor
    iters: int
    converged: bool
    model: Model

    def fk(self) -> Data:
        return forward_kinematics(self.model, self.q, compute_frames=True)

    def frame_pose(self, name: str) -> torch.Tensor:
        return self.fk().frame_pose_world[..., self.model.frame_id(name), :]

    def q_only(self) -> torch.Tensor:
        return self.q
```

`q` always has shape `(B..., nq)`. There is no separate "fixed-base
returns `(n,)` / floating-base returns `(7,) + (n,)`" rule. If the
user wants the free-flyer part separately, they slice
`q[..., :7]`.

### Two-stage solver (cuRobo-style)

High-DOF humanoids fail single-stage LM when the cost includes
collisions: the pose-only Hessian is well-conditioned but the
collision terms add near-zero gradients far from contact, stalling
the solver. The fix is **seed with LM (pose-only), refine with L-BFGS
(full cost)**:

```python
cfg = OptimizerConfig(optimizer="lm_then_lbfgs",
                      max_iter=30,          # LM seed
                      refine_max_iter=30)   # L-BFGS refine
```

`_build_optimizer(cfg)` returns a composite optimiser that:

1. **Stage 1.** Runs LM with
   `CostStack.set_active("self_collision", False)` and
   `set_active("world_collision", False)`. ~30 iterations, analytic
   pose Jacobian, fast.
2. **Stage 2.** Re-enables the full stack and runs L-BFGS for
   `refine_max_iter` iterations starting from stage-1's solution.

Convergence is measured on the full cost from the user's
perspective, but the heavy lifting happens on the simpler
sub-problem.

### Warm-started batched IK

All shapes are batched:

```python
q0 = torch.randn(128, model.nq)           # (128, nq)
targets = {"tool0": target_batch}         # (128, 7)
res = solve_ik(model, targets, initial_q=q0)
res.q                                     # (128, nq)
```

Residuals, cost stack, and solvers all accept the leading batch
dim; the FK traverses `(B, njoints, 7)`.

## `Trajectory`

```python
@dataclass(frozen=True)
class Trajectory:
    """A discrete-time motion plan, with optional batching.

    Two shape conventions are accepted:

    - **Unbatched** — q.shape == (T, nq), t.shape == (T,).
    - **Batched** — q.shape == (*B, T, nq), t.shape == (*B, T).
      B may be any prefix shape.
    """
    t:        torch.Tensor                # (*B, T) timestamps
    q:        torch.Tensor                # (*B, T, nq)
    v:        torch.Tensor | None = None  # (*B, T, nv)
    a:        torch.Tensor | None = None
    tau:      torch.Tensor | None = None  # (*B, T, nv) controls
    extras:   dict
    metadata: dict

    def __post_init__(self) -> None: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def num_knots(self) -> int: ...
    def with_batch_dims(self, ndim: int) -> "Trajectory": ...
    @property
    def duration(self) -> torch.Tensor: ...
    def slice(self, start_idx: int, end_idx: int) -> "Trajectory": ...
    def resample(self, t_new: torch.Tensor, *, kind: str = "linear") -> "Trajectory":
        """Manifold-aware: kind ∈ {'linear', 'cubic', 'sclerp'}.
        q interpolates SE(3) / SO(3) blocks via sclerp; v/a follow
        the chain rule. Raw quaternion lerp is wrong and is not exposed."""
    def downsample(self, factor: int) -> "Trajectory": ...
    def to_data(self, model: "Model", knot_idx: int | slice | None = None) -> "Data": ...
    @staticmethod
    def stack(*trajectories: "Trajectory") -> "Trajectory": ...
```

Source: `src/better_robot/tasks/trajectory.py`.

The two key behaviours are:

- **Manifold-aware resampling.** SO(3) blocks of `q` interpolate via
  spherical linear interpolation (sclerp); raw quaternion lerp would
  produce non-unit quaternions and is not exposed.
- **Optional batching.** Algorithms that need a concrete batch axis
  call `traj.with_batch_dims(1)` or read `traj.batch_shape` to
  normalise. `B = 1` is *not* forced for unbatched input — other
  public APIs already accept arbitrary leading shapes including the
  empty prefix.

## Trajectory optimisation

```python
def solve_trajopt(
    model: Model,
    *,
    horizon: int,
    dt: float,
    keyframes: dict[int, dict[str, torch.Tensor]] | None = None,
    initial_traj: Trajectory | None = None,
    cost_cfg: TrajOptCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    robot_collision: "RobotCollision | None" = None,
    parameterization: Literal["knots", "bspline"] = "bspline",
    n_control_points: int = 8,
) -> TrajOptResult:
    """Kinematic trajectory optimisation.

    Variable shape depends on parameterization:
      - "knots"   → (B, T, nq)               one variable per knot
      - "bspline" → (B, n_control_points, nq) cuRobo-style; smoothness implicit

    Residual stack:
        - one PoseResidual per keyframe target
        - JointPositionLimit at every evaluated knot
        - Velocity5pt / Accel5pt smoothness residuals (knots only)
        - SelfCollision at every knot (sparse Jacobian)
    """
```

Source: `src/better_robot/tasks/trajopt.py`.

### B-spline parameterisation (the default)

`solve_trajopt(parameterization="bspline" | "knots", ...)` picks the
optimisation variable. cuRobo's default — B-spline control points —
gives three wins:

1. **Fewer variables.** Optimise `n_control_points × nq` instead of
   `T × nq`; the LM Jacobian shrinks by ~4×.
2. **Implicit smoothness.** The B-spline basis is `C²`; no explicit
   velocity / acceleration / jerk residuals needed.
3. **Time-parameterisable.** The spline defines a function from
   normalised time `s ∈ [0, 1]` to `q(s)`; resampling to a different
   `dt` is closed-form.

The knot-based path is retained as an opt-in for cases where
smoothness residuals carry problem-specific weights (e.g. physical
limits on higher derivatives).

### `TrajectoryParameterization` Protocol

`Trajectory` is the sample-at-knots representation. For optimisation,
the variable that the solver sees is often *not* the per-knot tensor
— it can be a B-spline control-point grid, a basis-coefficient
vector, etc. The Protocol that owns this mapping:

```python
class TrajectoryParameterization(Protocol):
    """Map a flat optimisation variable z to a Trajectory.

    The solver works in z-space; residuals consume the unpacked
    Trajectory. Differentiable: chain rule from residual.jacobian
    wrt q composes back to z via the parameterisation Jacobian.
    """
    @property
    def tangent_dim(self) -> int: ...
    def unpack(self, z: torch.Tensor) -> Trajectory: ...
    def retract(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor: ...
    def pack_initial(self, traj: Trajectory) -> torch.Tensor: ...

class KnotTrajectory(TrajectoryParameterization):
    """Identity parameterisation: z is the per-knot Trajectory after
    a reshape. Right for dense exact-per-timestep constraints."""

class BSplineTrajectory(TrajectoryParameterization):
    """B-spline control points. Variable is (B, K, nq); the trajectory
    is reconstructed via a fixed basis at given knot times."""
```

Source: `src/better_robot/tasks/parameterization.py`.

The parameterisation handles its own manifold retraction so SO(3)
blocks of a control point retract correctly; the LM linear solver
works in `dz`-space. Residuals continue to read from `Trajectory` —
they do not see the parameterisation. Cross-references the
`ResidualSpec.time_coupling` field from {doc}`residuals_and_costs`.

### Sparsity and matrix-free

- Trajectory is a single variable with a leading time axis, not a
  list of per-knot `Data` objects.
- Smoothness residuals use 5-point finite differences vectorised
  along `T`.
- Per-knot limits broadcast across `T`; the solver sees a single
  flat variable `(B, T * nv)` (knots) or `(B, n_control_points * nv)`
  (B-spline).
- Each collision residual touches only the knots and chains it
  observes; `ResidualSpec` carries the sparsity hints.
- Long-horizon trajopt does **not** materialise a dense Jacobian.
  Adam and L-BFGS read `LeastSquaresProblem.gradient(x)` (matrix-free).
  Temporal residuals override `apply_jac_transpose` for banded `J^T r`
  in `O(T·nv)` memory.

## Retargeting

```python
def retarget(
    source_model: Model,
    target_model: Model,
    source_trajectory: Trajectory,
    *,
    frame_map: dict[str, str],            # {source_frame: target_frame}
    cost_cfg: RetargetCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
) -> TrajOptResult:
    """Motion retargeting: find a target-model trajectory that tracks
    source-model frame poses over time.

    Status: stub in v1. Reduces to solve_trajopt with keyframes
    built from the source trajectory's per-step frame poses.
    """
```

Source: `src/better_robot/tasks/retarget.py`. Currently raises
`NotImplementedError` — listed in {doc}`/reference/roadmap`.
Retargeting is a thin reduction to trajopt, not a separate algorithm.

## Examples

The shipped examples under `BetterRobot/examples/` are imported from
`tests/examples/test_examples.py` so they cannot bit-rot:

| File | What it demonstrates |
|------|----------------------|
| `01_basic_ik.py` | Single-arm IK, fixed base, analytic Jacobian, viser visualiser, draggable target gizmo |
| `02_g1_ik.py` | Humanoid whole-body IK with free-flyer root; proves the same `solve_ik` call handles it |
| `04_smpl_like_body.py` | Builder DSL constructs an SMPL-like body and runs FK |
| `05_panda_trajopt.py` | Reach-and-hold trajectory optimisation |

Each script has a `main()` that the example tests can call headlessly
(`viser` skipped on CI).

## API stability

The public surface of `tasks/` is stable from v1:

- **Top-level**: `solve_ik`, `solve_trajopt`, `retarget`,
  `Trajectory`.
- **Submodule-public** (reachable from `from better_robot.tasks.ik
  import …`): `IKResult`, `IKCostConfig`, `OptimizerConfig`.
- **Submodule-public** (`from better_robot.tasks.trajopt import …`):
  `TrajOptResult`, `TrajOptCostConfig`.

`solve_trajopt` and `retarget` are marked **experimental** in
{doc}`/conventions/contracts` §7.3 — the signatures will not wander
without a release note, but the internals may iterate.

## Sharp edges

- **Initial `q` is not clamped.** `solve_ik` projects `x_new` to
  `[lower, upper]` after each accepted step but does not project the
  initial guess. Callers must ensure `q0` is feasible if limits
  matter — `model.q_neutral` is *not* automatically inside bounds for
  every URDF (Panda's joint 4 upper limit is `-0.07` rad, while
  `q_neutral[3] = 0`; use `q_neutral.clamp(lower, upper)` as the
  starting point).
- **Free-flyer Jacobian shape.** For G1 (`nv = 42`), a single-frame
  Jacobian is `(B..., 6, 42)`. The first 6 columns are the base
  block. Slice if you need only the actuated subspace.
- **`solve_trajopt(parameterization="bspline")` smoothness is
  implicit.** Adding a `Velocity5pt` residual on top of a B-spline
  parameterisation double-counts smoothness; the knot path is the
  right choice if you want explicit smoothness residuals.
- **`Trajectory.resample` uses sclerp for SO(3).** Raw quaternion
  lerp would produce non-unit quaternions and is intentionally not
  exposed.

## Where to look next

- {doc}`solver_stack` — `LeastSquaresProblem` and the four
  pluggable axes that `solve_ik` builds.
- {doc}`residuals_and_costs` — the residual library that
  `solve_ik` and `solve_trajopt` compose.
- {doc}`viewer` — interactive IK with a draggable target gizmo
  (`add_ik_targets`).
