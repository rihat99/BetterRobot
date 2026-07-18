# Tasks — IK and Trajectory Optimisation

`tasks/` is the thin user-facing translation from a robot request to the
optimization substrate. There is no Jacobian code or fixed-vs-floating branch
here. `solve_ik` and knot-based `solve_trajopt` are named-block `Problem`
presets; `solve_trajopt` automatically uses banded temporal assembly when all
active residuals declare the required structure.

## Inverse kinematics

```python
def solve_ik(
    model: Model,
    targets: dict[str, torch.Tensor],         # {frame_name: (B..., 7) SE3 pose}
    *,
    initial_q: torch.Tensor | None = None,
    cost_cfg: IKCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
) -> IKResult:
    """Solve one or an arbitrary leading batch of frame-target problems."""
```

Source: `src/better_robot/tasks/ik.py`.

### `IKCostConfig`

```python
@dataclass
class IKCostConfig:
    pos_weight:       float = 1.0
    ori_weight:       float = 1.0
    pose_weight:      float = 1.0
    limit_weight:     float = 0.1
    rest_weight:      float = 0.01
    q_rest: torch.Tensor | None = None        # default: model.q_neutral
```

### Named-block preset

```python
def solve_ik(model, targets, *, initial_q=None, cost_cfg=None,
             optimizer_cfg=None) -> IKResult:
    cost_cfg      = cost_cfg or IKCostConfig()
    optimizer_cfg = optimizer_cfg or OptimizerConfig()
    seed = initial_q if initial_q is not None else model.q_neutral
    q_rest = cost_cfg.q_rest if cost_cfg.q_rest is not None else model.q_neutral
    active_q_rest = q_rest if cost_cfg.rest_weight > 0.0 else None
    q0 = broadcast_and_project(seed, targets, q_rest=active_q_rest)
    parameters = {f"target_pose_{i}": target
                  for i, target in enumerate(targets.values())}
    differentiable_parameters = list(parameters)
    residuals = pose_items(targets, parameters, cost_cfg)
    residuals += optional_limit_items(model, cost_cfg)
    if cost_cfg.rest_weight > 0.0:
        parameters["target_rest"] = q_rest
        differentiable_parameters.append("target_rest")
        residuals.append(ResidualItem(
            "rest",
            RestResidual(model, q_rest, target_name="target_rest"),
            weight=cost_cfg.rest_weight,
        ))
    robot = RobotConfig(model)
    problem = Problem(
        parameters=parameters,
        differentiable_parameters=tuple(differentiable_parameters),
    )
    problem.add_variable("q", manifold=robot, bounds=robot.joint_bounds())
    for item in residuals:
        problem.add_residual(
            item.residual,
            weight=item.weight,
            kernel=item.kernel,
            name=item.name,
        )
    values, state = named_block_solver(optimizer_cfg).run({"q": q0}, problem)
    return IKResult(q=values["q"], residual=problem.residual(values), ...)
```

Pose targets and the enabled rest target are declared `Problem.parameters` and
are read by name from the residual context. This keeps their differentiation
role explicit for the generic LM/GN implicit-solve boundary; no tensor-identity
inference is used. The `solve_ik` facade itself remains detached. `RestResidual`
reads the latter through `target_name="target_rest"`. Built-in pose, limit, and
rest residuals all use the named-context protocol used by direct problems.

### `IKResult`

```python
@dataclass
class IKResult:
    q: torch.Tensor                            # (B..., nq)
    residual: torch.Tensor
    iters: int | torch.Tensor
    converged: bool | torch.Tensor
    model: Model

    def fk(self) -> Data:
        return forward_kinematics(self.model, self.q, compute_frames=True)

    def frame_pose(self, name: str) -> torch.Tensor:
        return self.fk().frame_pose_world[..., self.model.frame_id(name), :]
```

Unbatched calls keep Python `int`/`bool` diagnostics. Batched calls return
per-element tensors with the common leading batch shape. There is no separate
fixed/floating return rule; a free-flyer remains part of `q`.

### Two-stage solver

`lm_then_adam` seeds with named-block LM and refines through the `torch.optim`
adapter:

```python
cfg = OptimizerConfig(optimizer="lm_then_adam",
                      max_iter=60)
```

The iteration budget is split evenly between the stages. Entries named in
`refine_disabled_items` receive a zero weight in the refinement problem. Named-block
L-BFGS is deliberately deferred because batching requires per-element history,
line search, and curvature-reset semantics; `"lbfgs"` and
`"lm_then_lbfgs"` fail with an actionable error. Collision is not wired into
`solve_ik`; its collision residuals remain stubbed and task integration is
explicitly deferred.

### Batched IK

`initial_q`, every target, and an enabled optional `q_rest` use ordinary
right-aligned Torch broadcasting over leading axes. When `rest_weight <= 0`,
the rest residual and parameter are absent, so `q_rest` does not participate in
broadcasting. One call solves the resulting common batch with independent
damping, acceptance, convergence, and iteration state. The committed
acceptance protocol compares one B=128 Panda call with 128 B=1 calls of this
same facade.

## `Trajectory`

```python
@dataclass
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
    extras:   dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_id: int = -1                   # legacy viewer alias

    def __post_init__(self) -> None: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def num_knots(self) -> int: ...
    def with_batch_dims(self, n: int = 1) -> "Trajectory": ...
    @property
    def horizon(self) -> int: ...
    @property
    def batch_size(self) -> int: ...
    def slice(self, t_start: float, t_end: float) -> "Trajectory": ...
    def resample(
        self,
        new_t: torch.Tensor,
        *,
        kind: Literal["linear", "sclerp"] = "linear",
    ) -> "Trajectory": ...
    def downsample(self, factor: int) -> "Trajectory": ...
    def to_data(self, model: "Model") -> "Data": ...
```

Source: `src/better_robot/tasks/trajectory.py`.

The key behaviours are:

- **Two explicit resampling modes.** `kind="linear"` interpolates every
  coordinate. `kind="sclerp"` assumes that `q[..., :7]` is one SE(3)-style
  block `[xyz, qx, qy, qz, qw]`: translation and all remaining coordinates
  stay linear, while only `q[..., 3:7]` uses quaternion SLERP. Optional
  `v`, `a`, and `tau` are linearly interpolated in both modes; no chain-rule
  reconstruction is performed. This mode does not discover additional SO(3)
  blocks from a `Model`.
- **Time-based slicing.** `slice(t_start, t_end)` selects the inclusive time
  interval, not array indices. Slicing and resampling use the first batch
  element as the reference time grid.
- **Optional batching.** Algorithms that need a concrete batch axis
  call `traj.with_batch_dims(1)` or read `traj.batch_shape` to
  normalise. `B = 1` is *not* forced for unbatched input — other
  public APIs already accept arbitrary leading shapes including the
  empty prefix.
- **Vectorised materialisation.** `to_data(model)` flattens all batch and time
  samples into one FK batch and computes frame placements.
- **Pose smoothing.** `smooth_trajectory(traj, kernel, kind="auto"|"so3"|"se3")`
  applies an odd-length box, Gaussian, or other non-negative kernel using
  iterative SLERP / ScLERP means. Batch and time windows are vectorised;
  quaternion signs are hemisphere-aligned before smoothing. `"auto"` selects
  SO(3) for a 4-vector and SE(3) for a 7-vector. Only `q` changes.

## Inverse contact-force fitting

`solve_contact_forces` fits world-frame point forces for a frozen
floating-base trajectory using the named-block LM solver:

```python
result = br.solve_contact_forces(
    model,
    q_traj,                         # (*B, T, nq)
    contact_joint_ids=[left, right],
    active_mask=contact_mask,       # broadcastable to (*B, T, C)
    dt=1.0 / 30.0,
    gravity=gravity,
    weights=ContactForceWeights(
        base_wrench=1.0,
        force_magnitude=1e-4,
        force_smooth=1e-3,
        torque_smooth=1e-3,
    ),
)
```

Forces are one Euclidean variable block with event shape `(T, C, 3)`. A
provider rotates them from world to joint-local coordinates, scatters
`[force, torque=0]` rows into `fext`, and evaluates RNEA once per optimization
context. The four weights control base-wrench balance, force magnitude,
temporal force smoothness, and actuated-torque smoothness. A `(..., 3)` or
spatial `(..., 6)` gravity tensor can be supplied per clip without replacing
the model.

`ContactForceResult` returns fitted world forces, local external wrenches,
generalized forces, final residual/cost, and per-batch solver status. This task
uses the Torch dynamics lane; Warp dynamics kernels remain unimplemented.

## Trajectory optimisation

```python
def solve_trajopt(
    model: Model,
    *,
    horizon: int,
    dt: float,
    initial_q_traj: torch.Tensor,
    residuals: Sequence[ResidualItem],
    optimizer: LevenbergMarquardt | None = None,
    max_iter: int = 50,
    jacobian_strategy: Literal[
        "auto", "analytic", "jacrev", "jacfwd", "finite_difference"
    ] = "auto",
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    parameterization: KnotTrajectory | None = None,
) -> TrajOptResult:
    """Kinematic trajectory optimisation.

    KnotTrajectory is the only supported robot parameterisation. Residual
    items are adapted to a named temporal q block. None constructs route-aware
    LM; pass named-block LM with linearization="dense" for the dense oracle.
    """
```

Source: `src/better_robot/tasks/trajopt.py`.

`initial_q_traj` may have arbitrary leading batch axes. Unbatched calls retain
the historical `(1, T, nq)` returned trajectory shape; batched calls preserve
their leading axes. `TrajOptResult` returns per-element `iters`, `converged`,
and `status`, plus `linearization_requested`, `linearization_used`,
`linearization_reason`, and `linearization_detail`. Automatic mode reports
`"banded"` only after every potentially active temporal residual supplies a
valid declaration and direct numeric blocks; otherwise it reports the dense
fallback and its stable reason.

### B-spline numerical utility

`BSplineTrajectory` builds and evaluates a Euclidean cubic basis, so it is
useful for numerical compression experiments. It is **not** accepted by robot
`solve_trajopt`. Linear interpolation of configuration coordinates does not
preserve quaternion manifolds; the former path also discarded bounds and was
not compatible with multi-stage problem replacement. A correct
spline-on-manifold trajectory path—including matching retraction/Jacobian and
feasible bounds—requires a separate reviewed design. Use `KnotTrajectory`
today.

### `TrajectoryParameterization` Protocol

`Trajectory` is the sample-at-knots representation. For optimisation,
the variable that the solver sees is often *not* the per-knot tensor
— it can be a B-spline control-point grid, a basis-coefficient
vector, etc. The Protocol that owns this mapping:

```python
class TrajectoryParameterization(Protocol):
    """Map an optimisation variable z to a sampled trajectory."""
    def init(self, q_traj_seed: torch.Tensor) -> torch.Tensor: ...
    def expand(self, z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor: ...
    def tangent_dim_per_step(self) -> int: ...

class KnotTrajectory(TrajectoryParameterization):
    """Identity parameterisation: z is the per-knot configuration tensor."""

class BSplineTrajectory(TrajectoryParameterization):
    """Euclidean B-spline basis utility; not safe robot trajopt."""
```

Source: `src/better_robot/tasks/parameterization.py`.

The structural Protocol describes the numerical mapping only; it does not
promise manifold or bound semantics. M5 deliberately did not reinterpret that
Euclidean utility as a robot-manifold map, so the robot task facade accepts
only `KnotTrajectory` pending a separate reviewed design.

### Dense and banded routes

- The trajectory is one `VarSpec("q", (T, nq), time_axis=0)`, not one Python
  variable per knot. Solver vectors remain flat knot-major tangents.
- Central velocity and acceleration, reference-trajectory, time-indexed, and
  contact-consistency residuals declare exact `TemporalPattern` support and
  provide reduced per-knot numeric blocks.
- `linearization="auto"` uses `BandedCholesky` when the whole active problem is
  directly eligible and otherwise uses the dense correctness path. Forced
  `"structured"` rejects an ineligible problem rather than falling back.
- Mixed temporal/shared optimized variables remain dense because Schur
  elimination has no second production caller. Collision residuals without a
  temporal declaration likewise keep the complete problem dense.

## Examples

The shipped examples under `examples/` demonstrate the main task facades:

| File | What it demonstrates |
|------|----------------------|
| `01_basic_ik.py` | Single-arm IK, fixed base, analytic Jacobian, viser visualiser, draggable target gizmo |
| `02_g1_ik.py` | Humanoid whole-body IK with free-flyer root; proves the same `solve_ik` call handles it |
| `04_smpl_like_body.py` | Builder DSL constructs an SMPL-like body and runs FK |
| `05_panda_trajopt.py` | Reach-and-hold trajectory optimisation |

Each script has a `main()` entry point. The documentation front-page example is
mechanically executed; the interactive scripts require the `demos` and
`viewer` extras.

## API stability

The public surface of `tasks/` is stable from v1:

- **Top-level**: `solve_ik`, `solve_trajopt`, `solve_contact_forces`,
  `Trajectory`.
- **Tasks-package public**: `smooth_trajectory`, `ContactForceResult`, and
  `ContactForceWeights`.
- **Submodule-public** (reachable from `from better_robot.tasks.ik
  import …`): `IKResult`, `IKCostConfig`, `OptimizerConfig`.
- **Submodule-public** (`from better_robot.tasks.trajopt import …`):
  `TrajOptResult`.

`solve_trajopt` is marked **experimental** in
{doc}`/conventions/contracts` §7.3 — the signatures will not wander
without a release note, but the internals may iterate.

## Sharp edges

- **The IK preset projects its seed.** Public `VarSpec` validation rejects an
  infeasible start, so `solve_ik` first projects `initial_q` (or
  `model.q_neutral`) to supported state bounds. This intentionally handles
  URDFs such as Panda whose neutral joint 4 lies outside its declared box.
  Direct named-block callers remain responsible for supplying a feasible
  initial `Values` mapping.
- **Free-flyer Jacobian shape.** For the locked G1 fixture (`nq = 36`,
  `nv = 35`), a single-frame Jacobian is `(B..., 6, 35)`. The first 6 columns
  are the base block. Slice if you need only the actuated subspace.
- **Non-knot robot trajopt is rejected.** `BSplineTrajectory` is a Euclidean
  numerical basis, not a manifold-aware robot parameterisation. Use
  `KnotTrajectory` until a reviewed spline-on-manifold mapping supplies the
  required retraction, Jacobian, and bound semantics.
- **Trajectory residual selection is explicit.** `solve_trajopt` evaluates the
  `ResidualItem` values the caller passes; omit a term to disable it. Results
  report requested/used linearization, reason/detail, and per-batch LM status.
- **`Trajectory.resample(..., kind="sclerp")` has one fixed pose layout.**
  It applies quaternion SLERP only to indices `3:7`; other coordinates and
  optional derivative/control arrays use linear interpolation. Use
  `kind="linear"` only when coordinate-wise interpolation is valid.

## Where to look next

- {doc}`solver_stack` — LM/GN, `torch.optim`, and automatic temporal routing.
- {doc}`residuals_and_costs` — the residual library that
  `solve_ik` and `solve_trajopt` compose.
- {doc}`viewer` — interactive IK with a draggable target gizmo
  (`add_ik_targets`).
