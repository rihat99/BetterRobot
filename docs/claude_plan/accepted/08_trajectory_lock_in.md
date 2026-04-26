# 08 · Lock the `Trajectory` shape before trajopt body lands

★★ **Structural.** Pre-condition for
[06_DYNAMICS.md §6 Three-layer action model](../../design/06_DYNAMICS.md)
and
[18_ROADMAP.md §2 Trajectory optimisation](../../status/18_ROADMAP.md).

## Problem

`Trajectory` is currently a stub: a dataclass shell, with `.slice()`,
`.resample()`, and `.to_data()` raising `NotImplementedError`. The
type is exported in `better_robot.__all__`, which means **its
interface is committed to v1 stability** under
[17 §7.1 SemVer](../../conventions/17_CONTRACTS.md). When the body
lands, every binding decision becomes a SemVer matter.

That is the wrong order. We should pin the type's shape and
contract *now*, while it is still a stub, so trajopt and retargeting
have a fixed target to write against. Otherwise we will end up
inventing ad-hoc trajectory shapes inside `tasks/trajopt.py` and
`tasks/retarget.py`, then retrofitting the public type to whatever
they happened to need.

The reference projects all converged on the same shape:

| Library | Trajectory representation |
|---------|---------------------------|
| Crocoddyl | `xs`, `us` lists keyed by `T` knots; per-knot `ActionData` |
| PyRoki | implicit `(T,)` along the time axis on the variable tensor |
| mjlab | sliced state; `(B, T, ...)` |
| Drake | `Trajectory<T>` polymorphic; spline / piecewise polynomial |

The pattern that fits BetterRobot's batched-by-default convention
is mjlab's: a typed dataclass with `(B, T, ...)` tensors and
explicit time axis.

## The proposal

### 8.A The `Trajectory` dataclass

```python
# src/better_robot/tasks/trajectory.py
from __future__ import annotations
from dataclasses import dataclass
import torch
from .._typing import ConfigTensor, VelocityTensor

@dataclass(frozen=True)
class Trajectory:
    """A discrete-time motion plan, with optional batching.

    Layout
    ------
    Two shape conventions are accepted:

    - **Unbatched** — ``q.shape == (T, nq)``, ``t.shape == (T,)``.
      A single trajectory written by hand or returned from a
      single-instance solve.
    - **Batched** — ``q.shape == (*B, T, nq)``, ``t.shape == (*B, T)``.
      ``B`` may be any prefix shape, matching the
      ``(B, [T,] ..., D)`` convention from
      [10_BATCHING_AND_BACKENDS.md §1](../../design/10_BATCHING_AND_BACKENDS.md).

    Algorithms that need a concrete batch axis call
    ``traj.with_batch_dims(1)`` (or read ``traj.batch_shape``) to
    normalise. We do **not** force ``B = 1`` for unbatched input
    — the gpt-plan review correctly flagged that as ceremony
    without benefit, since BetterRobot's other public APIs already
    accept arbitrary leading shapes including the empty prefix.

    Attributes
    ----------
    t        : Tensor[*B, T]              # timestamps in seconds
    q        : Tensor[*B, T, nq]          # configuration (manifold-valued)
    v        : Tensor[*B, T, nv] | None   # joint velocities (tangent-valued)
    a        : Tensor[*B, T, nv] | None   # joint accelerations
    tau      : Tensor[*B, T, nv] | None   # control torques (= u in OC)
    extras   : dict[str, Tensor]          # per-trajectory aux (optional)
    metadata : dict[str, Any]             # provenance: model name, solver, seed, …

    Invariants
    ----------
    - ``t`` is monotonically non-decreasing along the T axis.
    - ``t.shape[:-1] == q.shape[:-2]`` (consistent leading shape).
    - All tensors live on the same device and dtype.
    """
    t:       torch.Tensor
    q:       torch.Tensor
    v:       torch.Tensor | None = None
    a:       torch.Tensor | None = None
    tau:     torch.Tensor | None = None
    extras:  dict = ...        # default factory dict
    metadata: dict = ...        # default factory dict

    def __post_init__(self) -> None:
        ...                    # validate consistency at construction

    # ───────── shape ─────────

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading dims of ``q`` excluding T and nq. ``()`` if unbatched."""
        return tuple(self.q.shape[:-2])

    @property
    def num_knots(self) -> int:
        return int(self.q.shape[-2])

    def with_batch_dims(self, ndim: int) -> "Trajectory":
        """Normalise to exactly ``ndim`` leading batch dims.

        Useful for algorithms that need a concrete batch axis. If the
        current batch shape is shorter, prepends size-1 axes; if
        longer, raises ``ShapeError`` (no implicit reshape — the user
        must collapse explicitly).
        """

    @property
    def duration(self) -> torch.Tensor:        # (B,)
        return self.t[..., -1] - self.t[..., 0]

    # ───────── sampling ─────────

    def slice(self, start_idx: int, end_idx: int) -> "Trajectory":
        """Return a sub-trajectory ``[start_idx, end_idx)`` along the T axis."""

    def resample(self, t_new: torch.Tensor, *, kind: str = "linear") -> "Trajectory":
        """Resample at given timestamps. ``kind ∈ {'linear', 'cubic', 'sclerp'}``.

        ``q`` is manifold-aware (``sclerp`` for SE(3) / SO(3) blocks); ``v`` and
        ``a`` are interpolated linearly (or via the chain rule of the cubic
        spline) to remain consistent with ``q``.
        """

    def downsample(self, factor: int) -> "Trajectory":
        """Take every ``factor``-th knot."""

    # ───────── conversion ─────────

    def to_data(self, model, knot_idx: int | slice | None = None) -> Data:
        """Materialise a ``Data`` for a single knot (or a slice).

        Useful when a residual or visualizer wants per-knot FK output.
        """

    def stack(*trajectories: "Trajectory") -> "Trajectory":
        """Concatenate along the batch axis. Knot count must match;
        timestamps are assumed shared."""
```

### 8.B Conventions

- **Time axis is `T`, second-to-last batch dim**. This matches the
  shape convention from
  [10_BATCHING_AND_BACKENDS.md §1](../../design/10_BATCHING_AND_BACKENDS.md):
  `(B, [T,] ..., D)`.
- **Unbatched and batched are both first-class.** `q.shape == (T, nq)`
  and `q.shape == (*B, T, nq)` both validate.
- **`q` is manifold-valued**; `Trajectory.resample(...)` knows how
  to interpolate SE(3) / SO(3) blocks via `sclerp` / `slerp`. Linear
  interpolation on raw quaternions is wrong and the type must not
  enable it accidentally.
- **`v`, `a`, `tau` may be `None`**. Trajectory optimisation
  problems with cost only on `q` (kinematic IK over time) can leave
  them None. Dynamics-level trajopt fills them.
- **`extras`** is the escape hatch — contact phases, constraint
  multipliers, custom diagnostic — without forcing the dataclass
  to grow new typed fields for every research need.
- **`metadata`** is for user-side annotations (problem id, solver
  config dict, …). Not used by the library.

### 8.C What `solve_trajopt` returns

```python
@dataclass(frozen=True)
class TrajOptResult:
    trajectory: Trajectory
    residual: torch.Tensor             # final residual vector
    iters: int
    converged: bool
    status: str                        # "success" | "max_iter" | "stalled" | ...
    history: list[dict] | None = None  # per-iter (loss, lam, ...)
```

Same shape as `IKResult`; the discipline is consistent.

### 8.D The temporal residuals know the type

`residuals/smoothness.py:VelocityResidual`,
`AccelerationResidual`, `JerkResidual` already use 5-point finite
differences over a `T` axis. Their `state.variables` is a flat
`(B, T, nv)` tensor; the residual indexes it. They do **not**
construct a `Trajectory` — that is the user-facing type, returned
*from* `solve_trajopt`, not consumed inside the residual.

The contract: residuals operate on flat tensors; tasks (`solve_*`)
construct typed results.

### 8.E Compatibility with `compute_joint_jacobians` and friends

A `Trajectory` of shape `(B, T, nq)` can be folded into an FK call
in two ways:

```python
# (A) flatten time into batch
B, T, nq = traj.q.shape[:-1] + (traj.q.shape[-1],)
data = forward_kinematics(model, traj.q.reshape(B * T, nq))
# data.joint_pose_world: (B*T, njoints, 7) — reshape back if needed

# (B) FK supports the (B, T, ...) leading shape natively
data = forward_kinematics(model, traj.q)
# data.joint_pose_world: (B, T, njoints, 7)
```

(B) is preferred and already implied by the
`(B, [T,] ..., D)` convention. The trajopt path uses (B); the
single-knot debug path uses (A) via `Trajectory.to_data(...,
knot_idx=k)`.

### 8.F Time-coupling sparsity

[07 §7](../../design/07_RESIDUALS_COSTS_SOLVERS.md) defines
`ResidualSpec.time_coupling: Literal["single", "5-point", "custom"]`.
Temporal residuals' `.spec` declares their coupling so that the
sparse-Cholesky linear solver can build the right block structure.

This is already in the spec; the proposal here is just to *use* it
when trajopt lands, not to invent something new.

### 8.G Trajectory parameterizations live one layer up

`Trajectory` is the **sample-at-knots** representation. For
optimisation, the variable that the solver sees is often *not* the
per-knot tensor: it can be a B-spline control-point grid, a Lin/
Cosine basis, a per-segment polynomial, etc. That family is owned
by [Proposal 16 §16.E](16_optim_wiring_and_matrix_free.md), which
introduces a `TrajectoryParameterization` Protocol with concrete
`KnotTrajectory` and `BSplineTrajectory` subclasses.

The contract: parameterizations *produce* a `Trajectory` (via
`unpack(z) -> Trajectory`), and the solver works in `z`-space.
Residuals continue to read from `Trajectory` — they don't see the
parameterization. This proposal does **not** introduce the
parameterization Protocol; it just guarantees the parameterization
returns the type we are pinning here.

## Doc 08 amendments

[08_TASKS.md](../../design/08_TASKS.md) gets one new section pinning the
above. Until that doc is filled in, this proposal *is* the contract.

## Files that change

```
tasks/trajectory.py                  full implementation of the spec above
tasks/trajopt.py                     stub returns TrajOptResult instead of NotImplementedError once D2
tasks/retarget.py                    same shape; returns TrajOptResult
docs/design/08_TASKS.md              new section §1.5 "Trajectory dataclass"
tests/tasks/test_trajectory.py       new — slice, resample, batch_shape, validation
tests/tasks/test_trajopt_stub.py     stays — exercises the type even before the body lands
__init__.py                          Trajectory in __all__ (already there)
                                     TrajOptResult: not in __all__ (only IKResult is — see Proposal 06)
```

## Tradeoffs

| For | Against |
|-----|---------|
| Pinning the type means trajopt and retarget have a stable target. | We commit to a shape now without having implemented trajopt. Mitigation: the shape is conservative — `(B, T, ...)`, optional fields, escape hatches — and matches every reference library's convergence point. |
| `Trajectory.resample` does the right thing on manifold-valued `q`. | `kind="cubic"` for manifold-valued `q` is non-trivial. v1 ships `linear` and `sclerp`; `cubic` lands later. |
| `extras` and `metadata` give research code room without API churn. | A common research footgun is to put load-bearing data in `extras` and forget to test it. Mitigation: docstring is explicit; `extras` is for non-load-bearing aux data. |

## Acceptance criteria

- `Trajectory(t=..., q=...)` constructs cleanly for both
  ``q.shape == (T, nq)`` and ``q.shape == (*B, T, nq)``;
  `__post_init__` validates shape consistency and raises
  `ShapeError` on mismatch.
- `traj.batch_shape == ()` for unbatched input;
  `traj.with_batch_dims(1)` returns a normalised view.
- `traj.slice(0, 10).num_knots == 10`.
- `traj.resample(t_new, kind='sclerp')` interpolates SE(3) blocks
  geodesically; manifold-correctness asserted by
  `assert_close_manifold` from `utils/testing.py`.
- `traj.to_data(model, knot_idx=5)` returns a `Data` whose
  `q.shape` matches the trajectory's batch shape.
- `forward_kinematics(model, traj.q)` returns
  `data.joint_pose_world` whose leading shape matches
  `(*B, T, njoints, 7)` (or `(T, njoints, 7)` for unbatched).
- `tests/tasks/test_trajectory.py` covers every method, including
  the unbatched case as a first-class fixture.
- `docs/design/08_TASKS.md` references this proposal as the source
  of truth until the trajopt body lands.

## Cross-references

- [10_BATCHING_AND_BACKENDS.md §1](../../design/10_BATCHING_AND_BACKENDS.md) —
  `(B, T, ...)` convention.
- [07 §7](../../design/07_RESIDUALS_COSTS_SOLVERS.md) — `ResidualSpec`.
- [Proposal 06](06_public_api_audit.md) — `Trajectory` stays in
  `__all__`.
- [Proposal 14](14_dynamics_milestone_plan.md) — once `aba` lands,
  `solve_trajopt` over dynamics consumes this type.
