# 00 · Vision

> BetterRobot is a **PyTorch-first, GPU-native, batched-by-default** library for
> robot and human-body **kinematics, dynamics, and optimization**, modelled on
> Pinocchio's mathematical discipline and PyRoki's architectural economy, and
> designed from day one for a future Warp backend.

## What we're building

A ~5–8k LOC core that delivers:

1. **One coherent data model.** A pinocchio-style `Model` (frozen topology) +
   `Data` (per-query workspace) separation, with a *universal* joint/body/frame
   taxonomy — no "static vs floating base" dichotomy, no format-specific shortcuts.
2. **Batched, temporal tensors everywhere.** Every kinematic/dynamic routine
   accepts `(B, [T,] ..., D)` tensors. Single poses are just `B=1, T=1`.
3. **Elegant Jacobians.** Forward kinematics, pose residuals, and dynamics
   gradients all have a single Jacobian strategy API that transparently dispatches
   to analytic or autodiff without duplicating code paths.
4. **One residual / cost / solver stack** that serves IK, trajectory optimization,
   filtering, and (eventually) optimal control.
5. **Lego-like modularity.** Costs, residuals, kernels, strategies, and solvers
   are small, decoupled, individually replaceable.
6. **Readable names, canonical algorithms.** We steal algorithm names from
   Pinocchio (`rnea`, `aba`, `crba`, `forward_kinematics`, `compute_joint_jacobians`,
   `SE3`, `Motion`, `Force`, `Inertia`) because they are universal in the
   literature. We **do not** steal Pinocchio's cryptic storage names
   (`oMi`, `oMf`, `liMi`, `nle`, `Ycrb`) — those become readable identifiers
   (`joint_pose_world`, `frame_pose_world`, `joint_pose_local`, `bias_forces`,
   `composite_inertia`). See [13_NAMING.md](13_NAMING.md) for the full table
   and rationale.
7. **A parser layer that doesn't bleed.** URDF, MJCF, and programmatic builders
   all produce the same internal representation; parsers live at the boundary,
   never inside `Model`.
8. **Future-proof backend.** Today: PyPose for Lie math and torch-native kernels
   for everything else. Tomorrow: Warp kernels swapped in behind the same
   `torch.Tensor`-only API.

## Goals (measurable)

| # | Goal | Measurement |
|---|------|-------------|
| G1 | Universal joint system | Fixed, revolute (R{X,Y,Z}, unaligned), prismatic (P{X,Y,Z}, unaligned), spherical (SO3), planar, free-flyer (SE3), helical, translation, composite, mimic — all round-trippable through URDF and MJCF |
| G2 | No static/floating dichotomy | Floating base = a `JointFreeFlyer` root; no `base_pose` special case anywhere in the solver/task layer |
| G3 | Batched FK | `forward_kinematics(model, q: (B, T, nq)) → (B, T, n_link, 7)` in a single call, no Python loops over batch or time |
| G4 | One Jacobian API | `jacobian(model, data, frame_idx, strategy=JacStrategy.ANALYTIC | .AUTODIFF | .FUNCTIONAL)` returns the same shape regardless of strategy |
| G5 | GPU-ready | `model.to("cuda")` moves all tensor buffers; `forward_kinematics` runs on GPU with no Python branching on tensor values |
| G6 | Canonical algorithm names + readable storage | `rnea`, `aba`, `crba`, `compute_joint_jacobians`, `update_frame_placements`, `center_of_mass`, `compute_centroidal_map`; `Data` fields use `joint_pose_world` / `frame_pose_world` / `mass_matrix` / `bias_forces` per [13_NAMING.md](13_NAMING.md) |
| G7 | Small public API | `<= 25` exports at `better_robot.__init__` |
| G8 | Skeleton-first | Every algorithm in the public API has a stub with the correct signature on day one; implementations land incrementally |

## Non-goals (for v1)

- Own physics engine — contact/constraint integrators are placeholders; interop
  with mjwarp or similar comes later.
- Muscle dynamics or SMPL parameterisation — but the data model **must** be
  expressive enough that a user can construct an SMPL-like body by adding
  bodies/joints/frames with fixed shape parameters.
- Optimal-control solvers (DDP/iLQR) — room is reserved in the architecture
  (section 06/07) but implementations are future work.
- A Systems framework à la Drake. Controllers, filters and observers are out of
  scope for v1.
- A custom config metaclass system (e.g. IsaacLab `@configclass`). Plain
  `@dataclass` + `__post_init__` validation only.

## Performance targets (v1)

Concrete budgets live in [14_PERFORMANCE.md](14_PERFORMANCE.md). The
headline numbers: **Panda FK ≤ 150 µs, solve_ik ≤ 8 ms (B=1, 30 iter,
CUDA)**; **G1 whole-body IK ≤ 25 ms (60 iter)**. Every benchmark is
enforced on CI with a 20% regression gate.

## Unique contributions

No reference project simultaneously delivers:

- **PyTorch-first** (autograd composes end-to-end, no mode flag, no `AutoDiffXd`)
- **Pinocchio-grade API** (Model/Data/Motion/Force/Inertia, canonical algorithm names)
- **Batched + temporal as default** (`(B, [T,] ..., D)` from day one)
- **One residual/cost/solver stack** that serves IK, trajopt, filtering, OC
- **Elegant analytic + autodiff Jacobians** through a single strategy flag
- **A Warp backend roadmap** that does not leak into the public API

Those five are the reason to rewrite rather than patch.

## Guiding principles

1. **Skeleton first, bodies later.** Lay down every public symbol on day one
   with the correct signature, docstring, and `NotImplementedError` — or a
   minimal correct implementation when trivial. Refuse half-drawn abstractions
   that block downstream work.
2. **Steal algorithms, not jargon.** Pinocchio for dynamics (`rnea`, `aba`,
   `crba`), PyRoki for residual factories, Crocoddyl for the
   differential/integrated/action split, mjlab for parser patterns and the
   WarpBridge, cuRobo for `Protocol`-pluggable optimisers, B-spline
   trajectory parameterisation and adaptive kernel dispatch, PyPose for
   pluggable solver components. **Readable storage names** override the
   source library's conventions where those conventions impede reading.
3. **Batch axis, period.** If a function doesn't accept a leading batch axis,
   it is wrong and blocks the release.
4. **No backend leakage.** `pypose` imports are restricted to `math/`. `warp`
   imports (when added) are restricted to `backends/warp/`. Users see
   `torch.Tensor` and nothing else.
5. **Delete fearlessly.** The current split of fixed-base vs floating-base IK,
   the duplicated autodiff/analytic floating-base solvers, and the
   `_solve_floating_autodiff`/`_solve_floating_analytic` twins must all go.
   A single solver driven by joint-system topology replaces them.
6. **No speculative abstractions.** Skeletons are OK; speculative generality
   (e.g. a `Backend` ABC before there is a second backend) is not.

## Success criteria for v1

- `solve_ik(model, targets)` works for Panda (fixed) and G1 (free-flyer) through
  **the same code path** — the only difference is that G1's `model.joints[0]` is
  `JointFreeFlyer`.
- `forward_kinematics(model, q)` runs on CPU and CUDA unchanged.
- `compute_joint_jacobians(model, data)` returns results that round-trip via
  `jacrev` to within `1e-6` (analytic == autodiff proof).
- `rnea`, `aba`, `crba` exist as named functions with the correct signatures and
  raise `NotImplementedError("see docs/06_DYNAMICS.md")` when called.
- Everything in `src/better_robot/` respects the layering in
  [ARCHITECTURE.md](01_ARCHITECTURE.md).
- Parsing a URDF never imports anything outside `better_robot.io.parsers`.
