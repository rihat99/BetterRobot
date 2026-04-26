# Roadmap

BetterRobot ships a complete kinematics + optimisation stack today. A
small set of named symbols are deliberately stubbed — they have the
correct signature, raise `NotImplementedError`, and point here. This
page is the single canonical list.

If a symbol is *not* on this page, it is implemented and tested.

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
are live. Retargeting is the remaining stub:

| Symbol | File |
|---|---|
| `retarget` | `tasks/retarget.py` |

## Viewer

V1 ships interactive `Visualizer`, `Scene`, `SkeletonMode`,
`URDFMeshMode`, `GridOverlay`, `FrameAxesOverlay`, `TargetsOverlay`,
`ForceVectorsOverlay`, `ViserBackend`, `MockBackend`, `build_joint_panel`,
and a minimal `TrajectoryPlayer` with `show_frame` and `play`. The
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

## Backends

The `Backend` Protocol and the default `torch_native` backend are live.
Warp is the experimental second backend:

| Symbol | File |
|---|---|
| `WarpBridge.to_warp` / `.to_torch` | `backends/warp/bridge.py` |
| Warp FK / Jacobian / RNEA / ABA / CRBA kernels | `backends/warp/kernels/` |
| `enable_warp_backend`, `disable_warp_backend` | `backends/warp/__init__.py` |

## Performance

The hot-path lint, contract suite, and benchmark harness are in place.
The compile / capture hooks are wired but await dispatcher bodies:

| Symbol | File |
|---|---|
| `@torch.compile(fullgraph=True)` on FK / Jacobian / `CostStack` | not yet applied |
| `@cache_kernel` adaptive dispatch | not yet wired |
| `BR_PROFILE=1` env hook | not yet wired |

`@graph_capture` is a no-op under the torch-native backend; the real
CUDA-graph capture path lands with the Warp backend.

## How to close an entry

1. Read the relevant chapter in [`concepts/`](../concepts/index.md) and
   the matching extension recipe in
   [`conventions/extension.md`](../conventions/extension.md).
2. Implement the body. Keep the existing public signature.
3. Add tests under the matching folder in `tests/`. The contract
   tier in [`conventions/testing.md`](../conventions/testing.md)
   tells you which tests must accompany a public symbol.
4. Move the entry to `CHANGELOG.md` under the landing release.
