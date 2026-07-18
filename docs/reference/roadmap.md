# Roadmap

BetterRobot ships a kinematics + optimisation stack with a number of
deliberately unfinished extension points and fail-fast capability guards.
The inventory below is the canonical list of Python source files that contain
an explicit `raise NotImplementedError`. It is machine-checked against the
source tree; the later tables provide human-oriented detail for the main
planned capabilities.

The inventory is file-level because one file can contain several related
raise sites. Absence from it means only that a file has no explicit
`NotImplementedError`, not that every conceivable capability is implemented.

## Complete explicit-raise inventory

Paths are relative to the repository root. Keep this block sorted; the
contract test reports both missing source files and stale documentation
entries.

<!-- not-implemented-inventory:start -->
- `src/better_robot/collision/closest_pts.py`
- `src/better_robot/collision/geometry.py`
- `src/better_robot/collision/pairs.py`
- `src/better_robot/collision/robot_collision.py`
- `src/better_robot/dynamics/centroidal.py`
- `src/better_robot/dynamics/crba.py`
- `src/better_robot/dynamics/derivatives.py`
- `src/better_robot/dynamics/integrators.py`
- `src/better_robot/dynamics/rnea.py`
- `src/better_robot/io/build_model.py`
- `src/better_robot/residuals/collision.py`
- `src/better_robot/residuals/contact.py`
- `src/better_robot/residuals/limits.py`
- `src/better_robot/residuals/manipulability.py`
- `src/better_robot/residuals/regularization.py`
- `src/better_robot/residuals/smoothness.py`
- `src/better_robot/spatial/force.py`
- `src/better_robot/tasks/ik.py`
- `src/better_robot/tasks/trajopt.py`
<!-- not-implemented-inventory:end -->

## Dynamics

The recursive Featherstone passes are live: `rnea`, `aba`, `crba`,
`ccrba`, `compute_centroidal_map`, `compute_centroidal_momentum`, and
`center_of_mass`. The autograd-derived `compute_rnea_derivatives`,
`compute_aba_derivatives`, and `compute_crba_derivatives` are implemented.
The underlying RNEA/ABA passes have gradcheck coverage, while derivative
identity tests compare `∂τ/∂a` with CRBA and `∂a/∂τ` with the inverse mass
matrix. The remaining pieces:

| Symbol | File | What it needs |
|---|---|---|
| `compute_minverse` | `dynamics/crba.py` | Direct ABA-factorisation path that skips the explicit `crba` + `cholesky_solve`. |
| `compute_coriolis_matrix` | `dynamics/rnea.py` | Standalone world-frame recursion for `C(q, v)`. |
| `compute_centroidal_dynamics_derivatives` | `dynamics/derivatives.py` | Analytic recursion. The autograd path is documented as a workaround. |
| Analytic Carpentier–Mansard derivatives | `dynamics/derivatives.py` | Replace the autograd bodies of the RNEA, ABA, and CRBA derivative helpers with the analytic forms. |
| `semi_implicit_euler`, `symplectic_euler`, `rk4` | `dynamics/integrators.py` | Bodies. `integrate_q` is live. |

## Residuals

The full residual library is live except:

| Symbol | File |
|---|---|
| `JerkResidual` | `residuals/smoothness.py` |
| `YoshikawaResidual` | `residuals/manipulability.py` |
| `SelfCollisionResidual`, `WorldCollisionResidual` | `residuals/collision.py` |
| `JointVelocityLimit.jacobian` | `residuals/limits.py` (the `__call__` works; the analytic Jacobian is missing) |
| `JointAccelLimit` | `residuals/limits.py` |
| `NullspaceResidual` | `residuals/regularization.py` |

## Collision

The exported collision dataclasses are usable only as containers.
`distance`, the closest-point helpers, `colldist_from_sdf`, every
`RobotCollision` constructor/query, and both collision residual evaluations
raise `NotImplementedError`. No collision task integration or performance
claim ships today; see {doc}`/concepts/collision_and_geometry`.

## Tasks

`solve_ik` and knot-based `solve_trajopt` are live on named blocks. Temporal
residuals declare `TemporalPattern` support; automatic LM uses block-banded
assembly when directly eligible and otherwise records a dense fallback.
`BSplineTrajectory` remains a Euclidean numerical basis utility: M5 did not
make it a robot-manifold map, and robot use stays rejected pending a separate
reviewed interpolation/retraction and bound contract. Schur elimination for a
temporal block plus shared variables is likewise deferred until a second
production caller exists. The former retargeting placeholder was removed;
build retargeting explicitly from trajectory residuals and `solve_trajopt`.

## Viewer

The live viewer surface consists of `Visualizer`, `Scene`,
`SkeletonMode`, `URDFMeshMode`, the grid, frame-axes, target, and force-vector
overlays, `PrimitiveHandle`, `ViserBackend`, `MockBackend`,
`build_joint_panel`, and integer-frame `TrajectoryPlayer` playback. The
viewer currently has no explicit-raise roadmap entries: unsupported surfaces
are omitted instead of shipping as importable placeholders.

## Compute lanes

The direct Torch raw passes are live and use `ModelStructure` plus
`ModelValues`. A fused Warp FK pass is live behind the explicit
``use_warp=True`` selector. It has CUDA forward/VJP parity coverage,
forward-only capture coverage, and forward-only benchmark evidence, but remains opt-in pending owner review of
the Torch-recompute backward cost. Warp is not a public Protocol or
process-wide selector.

| Work item | Status |
|---|---|
| Fused Warp FK and Torch-recompute autograd bridge | CUDA-validated, opt-in; default review pending |
| Warp Jacobian / residual / dynamics kernels and adjoints | Not implemented |
| Eligibility and explicit lane choice | Implemented at the FK boundary; repeat locally for each future pass |
| Forward/backward parity | Implemented for FK; required independently for every future pass |

## Performance

The hot-path lint, contract suite, and benchmark harness are in place.
Caller-side full-graph compilation is validated for raw FK. Automatic
compilation and broader boundaries remain roadmap work:

| Symbol | File |
|---|---|
| Automatic `@torch.compile(fullgraph=True)` on public FK / Jacobian / `Problem` evaluation | not applied; raw FK supports explicit caller-side compilation |
| `@cache_kernel` adaptive dispatch | not yet wired |
| `BR_PROFILE=1` env hook | not yet wired |

No public or internal capture driver, context manager, or captured solver mode
ships today. Public solver ``run`` remains eager, and an end-to-end IK capture
lifecycle and benchmark are still open.

## How to close an entry

1. Read the relevant chapter in [`concepts/`](../concepts/index.md) and
   the matching extension recipe in
   [`conventions/extension.md`](../conventions/extension.md).
2. Implement the body. Keep the existing public signature.
3. Add tests under the matching folder in `tests/`. The contract
   tier in [`conventions/testing.md`](../conventions/testing.md)
   tells you which tests must accompany a public symbol.
4. Move the entry to `CHANGELOG.md` under the landing release.
