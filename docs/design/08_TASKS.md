# Tasks — IK, Trajectory Optimization, Retargeting

`tasks/` is the **thinnest** layer. Every entry point here is a short facade
that:

1. Picks frames / links / targets the user gave by name,
2. Builds a `CostStack` out of residuals from `residuals/`,
3. Wraps it in a `LeastSquaresProblem`,
4. Calls an `Optimizer`,
5. Returns a clean result type.

No Jacobian code, no solver loops, no fixed-vs-floating-base branch live
here — all of that belongs one layer down.

## 1. Inverse kinematics

```python
# src/better_robot/tasks/ik.py

def solve_ik(
    model: Model,
    targets: dict[str, torch.Tensor],              # {frame_name: (7,) SE3}
    *,
    initial_q: torch.Tensor | None = None,
    cost_cfg: IKCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    robot_collision: "RobotCollision | None" = None,
) -> IKResult:
    """Whole-body inverse kinematics for one or more frame targets.

    There is only **one** code path. Floating-base robots are those whose
    `model.joint_models[1]` is `JointFreeFlyer` — the solver does not need
    to know. All 7 quaternion-xyz values of the free-flyer live inside `q`.
    """
```

### `IKCostConfig`

```python
@dataclass
class IKCostConfig:
    pos_weight:      float = 1.0
    ori_weight:      float = 1.0
    pose_weight:     float = 1.0
    limit_weight:    float = 0.1
    rest_weight:     float = 0.01
    collision_margin: float = 0.02
    collision_weight: float = 1.0
    q_rest: torch.Tensor | None = None            # default: model.q_neutral
```

### `OptimizerConfig`

```python
@dataclass
class OptimizerConfig:
    optimizer: Literal["lm", "gn", "adam", "lbfgs", "lm_then_lbfgs"] = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO
    linear_solver: Literal["cholesky", "lstsq", "cg"] = "cholesky"
    kernel: Literal["l2", "huber", "cauchy", "tukey"] = "l2"
    damping: Literal["constant", "adaptive", "trust_region"] = "adaptive"
    tol: float = 1e-7

    # Two-stage refinement (cuRobo pattern): first LM on the pose-only cost
    # for fast seed convergence; then L-BFGS on the full cost for collision
    # refinement. Enabled by `optimizer="lm_then_lbfgs"`.
    refine_max_iter:    int = 30           # iterations in the second stage
    refine_cost_mask:   tuple[str, ...] = ("pose_*", "limits", "rest",
                                            "self_collision", "world_collision")
```

### `solve_ik` body (sketch)

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

### `IKResult`

```python
@dataclass
class IKResult:
    q: torch.Tensor                           # (B..., nq)
    residual: torch.Tensor
    iters: int
    converged: bool
    model: Model                              # for convenient FK / frame lookup

    def fk(self) -> Data:
        return forward_kinematics(self.model, self.q, compute_frames=True)

    def frame_pose(self, name: str) -> torch.Tensor:
        return self.fk().frame_pose_world[..., self.model.frame_id(name), :]

    def q_only(self) -> torch.Tensor:
        return self.q
```

No separate "fixed-base returns `(n,)` / floating-base returns `(7,) + (n,)`"
rule. `q` always has shape `(B..., nq)`. If the user wants the free-flyer
part separately, they slice: `q[..., :7]`.

### Two-stage solver (cuRobo-style)

High-DoF humanoids fail single-stage LM when the cost includes
collisions: the pose-only Hessian is well-conditioned but the collision
terms add near-zero gradients far from contact, stalling the solver. The
fix is **seed with LM (pose-only), refine with L-BFGS (full cost)**:

```python
cfg = OptimizerConfig(optimizer="lm_then_lbfgs",
                      max_iter=30,          # LM seed
                      refine_max_iter=30)   # L-BFGS refine
```

Under the hood, `_build_optimizer(cfg)` returns a composite optimiser
that:
1. **Stage 1** — runs LM with `CostStack.set_active("self_collision", False)`
   and `set_active("world_collision", False)`. ~30 iterations, analytic
   pose Jacobian, fast.
2. **Stage 2** — re-enables the full stack and runs L-BFGS for
   `refine_max_iter` iterations starting from stage-1's solution.

Convergence is measured on the full cost from the user's perspective,
but the heavy lifting happens on the simpler sub-problem. This is the
cuRobo MPPI-+-LBFGS split adapted to gradient-based IK.

### Warm-started batched IK

All shapes are batched:

```python
q0 = torch.randn(128, model.nq)          # (128, nq)
targets = {
    "tool0": target_batch,               # (128, 7)
}
res = solve_ik(model, targets, initial_q=q0)
res.q                                    # (128, nq)
```

The residuals, cost stack, and solvers all accept the leading batch dim;
the FK traverses `(B, njoints, 7)`.

## 2. `Trajectory`

```python
# src/better_robot/tasks/trajectory.py

@dataclass(frozen=True)
class Trajectory:
    """A discrete-time motion plan, with optional batching.

    Two shape conventions are accepted:

    - **Unbatched** — ``q.shape == (T, nq)``, ``t.shape == (T,)``.
    - **Batched** — ``q.shape == (*B, T, nq)``, ``t.shape == (*B, T)``.
      ``B`` may be any prefix shape, matching the ``(B, [T,] ..., D)``
      convention from
      `10_BATCHING_AND_BACKENDS.md §1 <10_BATCHING_AND_BACKENDS.md>`_.

    Algorithms that need a concrete batch axis call
    ``traj.with_batch_dims(1)`` (or read ``traj.batch_shape``) to
    normalise. We do **not** force ``B = 1`` for unbatched input — other
    public APIs already accept arbitrary leading shapes including the
    empty prefix.
    """
    t:        torch.Tensor                # (*B, T) timestamps
    q:        torch.Tensor                # (*B, T, nq)
    v:        torch.Tensor | None = None  # (*B, T, nv)
    a:        torch.Tensor | None = None
    tau:      torch.Tensor | None = None  # (*B, T, nv) controls
    extras:   dict = field(default_factory=dict)   # per-trajectory aux (optional)
    metadata: dict = field(default_factory=dict)   # provenance

    def __post_init__(self) -> None:
        # validate consistency: t.shape[:-1] == q.shape[:-2];
        # T axes match; v/a/tau (if present) share leading shape;
        # all tensors on same device + dtype. Raises ShapeError.
        ...

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading dims of q excluding T and nq. ``()`` if unbatched."""
        return tuple(self.q.shape[:-2])

    @property
    def num_knots(self) -> int:
        return int(self.q.shape[-2])

    def with_batch_dims(self, ndim: int) -> "Trajectory":
        """Normalise to exactly ``ndim`` leading batch dims."""

    @property
    def duration(self) -> torch.Tensor:    # (B,)
        return self.t[..., -1] - self.t[..., 0]

    def slice(self, start_idx: int, end_idx: int) -> "Trajectory": ...
    def resample(self, t_new: torch.Tensor, *, kind: str = "linear") -> "Trajectory":
        """Manifold-aware: ``kind ∈ {'linear', 'cubic', 'sclerp'}``.
        ``q`` interpolates SE(3) / SO(3) blocks via sclerp; v/a follow
        the chain rule. Raw quaternion lerp is wrong and is not exposed."""
    def downsample(self, factor: int) -> "Trajectory": ...
    def to_data(self, model: "Model", knot_idx: int | slice | None = None) -> "Data": ...
    @staticmethod
    def stack(*trajectories: "Trajectory") -> "Trajectory": ...
```

`solve_trajopt` returns `TrajOptResult(trajectory=...)` — the typed result
mirrors `IKResult`. `TrajOptResult` is **not** in top-level `__all__`;
import via `from better_robot.tasks.trajopt import TrajOptResult`.

## 3. Trajectory optimization

```python
def solve_trajopt(
    model: Model,
    *,
    horizon: int,
    dt: float,
    keyframes: dict[int, dict[str, torch.Tensor]] | None = None,
    # {t_index: {frame_name: target_pose}}
    initial_traj: Trajectory | None = None,
    cost_cfg: TrajOptCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    robot_collision: "RobotCollision | None" = None,
    parameterization: Literal["knots", "bspline"] = "bspline",
    n_control_points: int = 8,
) -> TrajOptResult:
    """Kinematic trajectory optimisation.

    Variable shape depends on `parameterization`:
      - "knots"   → (B, T, nq)        one optimisation variable per knot
      - "bspline" → (B, n_control_points, nq)  cuRobo-style; knots are the
                    spline evaluation, smoothness is implicit

    Residual stack:
        - one PoseResidual per keyframe target
        - JointPositionLimit at every evaluated knot
        - Velocity5pt / Accel5pt / Jerk5pt smoothness residuals (only if
          parameterization == "knots"; B-spline carries smoothness in the
          basis itself)
        - SelfCollision at every knot (sparse Jacobian — see 09)
        - Swept-volume collision (coarse: sphere sweep; optional in v1)

    Status: skeleton in v1. Implementation arrives once `Trajectory`
    threads cleanly through `ResidualState` and `CostStack`.
    """
```

### B-spline parameterisation (default)

`solve_trajopt(parameterization="bspline" | "knots", ...)` picks the
optimisation variable. cuRobo's default — **B-spline control points** —
gives three wins:

1. **Fewer variables** — optimise `n_control_points × nq` instead of
   `T × nq`; the LM Jacobian shrinks by ~4×.
2. **Implicit smoothness** — the B-spline basis is `C²`; no explicit
   velocity/acceleration/jerk residuals needed.
3. **Time-parameterisable** — the spline defines a function from
   normalised time `s ∈ [0,1]` to `q(s)`; resampling to a different
   `dt` is closed-form.

The knot-based path is retained as an opt-in for cases where
smoothness residuals carry problem-specific weights (e.g. physical
limits on higher derivatives).

### 3.1 `TrajectoryParameterization` Protocol

`Trajectory` is the **sample-at-knots** representation. For optimisation,
the variable that the solver sees is often *not* the per-knot tensor —
it can be a B-spline control-point grid, a basis-function coefficient
vector, etc. The Protocol that owns this mapping:

```python
# src/better_robot/tasks/parameterization.py
from typing import Protocol

class TrajectoryParameterization(Protocol):
    """Map a flat optimization variable z to a Trajectory.

    The solver works in z-space; residuals consume the unpacked
    Trajectory. Differentiable: chain rule from residual.jacobian wrt q
    composes back to z via the parameterization Jacobian.
    """
    @property
    def tangent_dim(self) -> int: ...
    def unpack(self, z: torch.Tensor) -> Trajectory: ...
    def retract(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor: ...
    def pack_initial(self, traj: Trajectory) -> torch.Tensor: ...

class KnotTrajectory(TrajectoryParameterization):
    """Identity parameterization: z is the per-knot Trajectory after a
    reshape. Right for dense exact-per-timestep constraints."""

class BSplineTrajectory(TrajectoryParameterization):
    """B-spline control points. Variable is (B, K, nq); the trajectory
    is reconstructed via a fixed basis at given knot times. Smaller
    optimisation variable for the same horizon."""
```

`solve_trajopt(parameterization=BSplineTrajectory(...))` is the default
for smooth motion generation. The parameterisation handles its own
manifold retraction so SO(3) blocks of a control point retract correctly;
the LM linear solver works in `dz`-space.

Residuals continue to read from `Trajectory` — they don't see the
parameterisation. Cross-references
[07_RESIDUALS_COSTS_SOLVERS.md §7](07_RESIDUALS_COSTS_SOLVERS.md) for the
`ResidualSpec` time-coupling field.

Key constraints:

- Trajectory is **a single variable with a leading time axis**, not a list
  of per-knot `Data` objects.
- Smoothness residuals use 5-point finite differences vectorised along `T`.
- Per-knot limits broadcast across `T`; the solver sees a single flat
  variable `(B, T * nv)` (knots) or `(B, n_control_points * nv)` (bspline).
- Sparse Jacobian blocks are the rule — each collision residual touches
  only the knots and chains it observes. `ResidualSpec` (see
  [07_RESIDUALS_COSTS_SOLVERS.md §7](07_RESIDUALS_COSTS_SOLVERS.md))
  carries the sparsity hints.
- Long-horizon trajopt does **not** materialise a dense Jacobian; Adam
  and L-BFGS read `LeastSquaresProblem.gradient(x)` (matrix-free) per
  [07 §4](07_RESIDUALS_COSTS_SOLVERS.md). Temporal residuals override
  `apply_jac_transpose` for banded `J^T r` in `O(T·nv)` memory.

## 4. Retargeting

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

    Status: skeleton in v1. Reduces to `solve_trajopt` with `keyframes`
    built from the source trajectory's per-step frame poses.
    """
```

Retargeting is a thin reduction to trajopt, not a separate algorithm.

## 5. Already retired (reference)

The pre-skeleton task layer had the following structure; it has been
removed. Listed here so readers tracing old code can find replacements:

- `tasks/ik/solver.py` (585 lines of fixed/floating autodiff/analytic
  code) → `tasks/ik.py` (~120 lines of facade).
- `tasks/ik/variable.py` (`IKVariable` flat packing) → deleted. `q`
  *is* the flat variable; free-flyer and joint DOFs coexist in it.
- `tasks/ik/config.py` (`IKConfig`) → split into `IKCostConfig` and
  `OptimizerConfig`.
- `tasks/trajopt/` stub → `tasks/trajopt.py` skeleton.
- `tasks/retarget/` stub → `tasks/retarget.py` skeleton.

## 6. Examples we ship

Both examples live under `examples/` and are imported from
`tests/test_examples.py` (PyRoki pattern).

| File | What it demonstrates |
|------|----------------------|
| `examples/01_panda_ik.py` | Single-arm IK, fixed base, analytic jacobian, viser visualiser |
| `examples/02_g1_floating_ik.py` | Humanoid whole-body IK with free-flyer root; proves that the same `solve_ik` call handles it |
| `examples/03_batched_ik.py` | `(B=128, nq)` parallel IK on GPU |
| `examples/04_smpl_like_body.py` | Builder DSL constructs an SMPL-like body, runs marker-fitting IK |
| `examples/05_trajopt_reach.py` | (post-skeleton) simple reach-and-hold trajopt |
| `examples/06_collision_ik.py` | IK with `SelfCollisionResidual` active |

## 7. API stability

After v1, the public surface of `tasks/` is frozen: `solve_ik`,
`solve_trajopt`, `retarget`, `Trajectory`, `IKResult`, `TrajOptResult`,
`IKCostConfig`, `TrajOptCostConfig`, `OptimizerConfig`. Everything else is
private and may be reshaped without a deprecation. Users who want
fine-grained control drop down to `optim/` directly — the tasks layer is
pure convenience.
