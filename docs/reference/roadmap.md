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
- `src/better_robot/optim/state.py`
- `src/better_robot/residuals/collision.py`
- `src/better_robot/residuals/contact.py`
- `src/better_robot/residuals/limits.py`
- `src/better_robot/residuals/manipulability.py`
- `src/better_robot/residuals/regularization.py`
- `src/better_robot/residuals/smoothness.py`
- `src/better_robot/spatial/force.py`
- `src/better_robot/tasks/ik.py`
- `src/better_robot/viewer/camera.py`
- `src/better_robot/viewer/interaction.py`
- `src/better_robot/viewer/overlays/com.py`
- `src/better_robot/viewer/overlays/path_trace.py`
- `src/better_robot/viewer/overlays/residual_plot.py`
- `src/better_robot/viewer/recorder.py`
- `src/better_robot/viewer/render_modes/collision.py`
- `src/better_robot/viewer/renderers/offscreen_backend.py`
- `src/better_robot/viewer/renderers/viser_backend.py`
- `src/better_robot/viewer/trajectory_player.py`
- `src/better_robot/viewer/visualizer.py`
<!-- not-implemented-inventory:end -->

## Dynamics

The recursive Featherstone passes are live: `rnea`, `aba`, `crba`,
`ccrba`, `compute_centroidal_map`, `compute_centroidal_momentum`, and
`center_of_mass`. The autograd-derived `compute_rnea_derivatives`,
`compute_aba_derivatives`, and `compute_crba_derivatives` work and pass
gradcheck. The remaining pieces:

| Symbol | File | What it needs |
|---|---|---|
| `compute_minverse` | `dynamics/crba.py` | Direct ABA-factorisation path that skips the explicit `crba` + `cholesky_solve`. |
| `compute_coriolis_matrix` | `dynamics/rnea.py` | Standalone world-frame recursion for `C(q, v)`. |
| `compute_centroidal_dynamics_derivatives` | `dynamics/derivatives.py` | Analytic recursion. The autograd path is documented as a workaround. |
| Analytic Carpentier–Mansard derivatives | `dynamics/derivatives.py` | Replace the autograd bodies of `compute_*_derivatives` with the analytic forms. |
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

## Tasks

`solve_ik` and `solve_trajopt` (with knot and B-spline parameterisations)
are live. The former retargeting placeholder was removed; see
{doc}`m1_removed_symbols`.

## Viewer

V1 ships interactive `Visualizer`, `Scene`, `SkeletonMode`,
`URDFMeshMode`, `GridOverlay`, `FrameAxesOverlay`, `TargetsOverlay`,
`ForceVectorsOverlay`, `ViserBackend`, `MockBackend`, `build_joint_panel`,
and a minimal `TrajectoryPlayer` with `show_frame` and `play`. The
``Backend`` suffix in this section names a scene renderer only; it is
unrelated to algorithm compute lanes. The
remaining pieces sit behind named placeholders so user code and tests
have a target to reach for:

| Symbol | File |
|---|---|
| `CollisionMode` | `viewer/render_modes/collision.py` |
| `ComOverlay`, `PathTraceOverlay`, `ResidualPlotOverlay` | `viewer/overlays/{com,path_trace,residual_plot}.py` |
| `VideoRecorder`, `render_trajectory` | `viewer/recorder.py` |
| `OffscreenBackend` | `viewer/renderers/offscreen_backend.py` |
| `TrajectoryPlayer.seek` / `.step` / `.pause` / `.set_speed` / `.set_loop` / `.set_ghost` / `.set_trace` / `.set_batch_index` | `viewer/trajectory_player.py` |
| `CameraPath.orbit`, `CameraPath.follow_frame` | `viewer/camera.py` |

## Compute lanes

The direct Torch raw passes are live and use `ModelStructure` plus
`ModelValues`. Warp is planned as an opt-in whole-pass optimisation, not as a
public Protocol or process-wide selector.

| Work item | Location |
|---|---|
| Warp FK / Jacobian / RNEA kernels and their adjoints | Beside the corresponding Torch pass |
| Eligibility and explicit lane choice | The owning whole-pass integration boundary |
| Forward and backward parity | Shared Torch-oracle fixtures |

## Performance

The hot-path lint, contract suite, and benchmark harness are in place.
Compilation and capture remain explicit roadmap work:

| Symbol | File |
|---|---|
| `@torch.compile(fullgraph=True)` on FK / Jacobian / `CostStack` | not yet applied |
| `@cache_kernel` adaptive dispatch | not yet wired |
| `BR_PROFILE=1` env hook | not yet wired |

No capture decorator or context manager ships today. A future capture path
must use fixed storage and record forward and backward together; its
lifecycle belongs to the M2b/M6 solver and kernel work.

## How to close an entry

1. Read the relevant chapter in [`concepts/`](../concepts/index.md) and
   the matching extension recipe in
   [`conventions/extension.md`](../conventions/extension.md).
2. Implement the body. Keep the existing public signature.
3. Add tests under the matching folder in `tests/`. The contract
   tier in [`conventions/testing.md`](../conventions/testing.md)
   tells you which tests must accompany a public symbol.
4. Move the entry to `CHANGELOG.md` under the landing release.
