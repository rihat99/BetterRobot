# 12 · Viewer

`viewer/` is the **topmost** layer in the DAG (see 01 ARCHITECTURE §Dependency
rule). Nothing imports from it. Its job is to turn `Model`, `Data`, and
`Trajectory` into something a user can see in a browser — without dragging
any visualisation dependency into the core.

The V1 scope is deliberately small: **interactive viser-backed rendering
of one robot, using two render modes (skeleton, URDF mesh) plus a grid
and frame-axes overlay, draggable IK-target gizmos for interactive IK,
and straight-through-the-frames trajectory playback.** Everything else
(video recording, offscreen rendering, collision visualisation, COM
marker, path traces, residual plots, camera paths, multi-robot
sessions) is explicitly **future work** and lands as a placeholder —
see §10.

The design is layered so the cuts above are additive: once V1 is solid,
each future item can be dropped in without restructuring.

## 1. Goals for V1

1. **Interactive browser visualisation only.** `Visualizer(model).show()`
   opens a viser session and draws the robot. That's the one happy path.
2. **Two render modes: `SkeletonMode` and `URDFMeshMode`.** Skeleton mode
   works on any `Model`. URDF mesh mode is used automatically when the
   robot was loaded from a URDF/MJCF (i.e. `model.meta["ir"]` is populated).
3. **`Scene` composes render modes.** Users (or `Scene.default`) attach
   zero-or-more modes; the scene routes kinematics updates to each one.
   This is the extensibility point for future modes.
4. **Draggable IK target gizmos (`TargetsOverlay`).** Lives inside the
   same `Scene.add_mode` lifecycle as any other render mode. Interactive
   backends expose each target as a draggable SE(3) gizmo; on drag an
   `on_change` callback fires with the updated target dict, so callers
   can re-solve IK in-the-loop and push the new `q` to the viewer — see
   §7.1 and `examples/01_basic_ik.py`, `examples/02_g1_ik.py`.
5. **Straight-through sequence playback.** A `TrajectoryPlayer` knows how
   to push a given frame of a `(B, T, nq)` `Trajectory` to the scene.
   There is no scrub bar, speed control, loop toggle, ghosting, trace
   overlay, or batch picker in V1 — **you can just play the sequence.**
6. **Backend-agnostic render modes.** Render modes speak only to the
   `RendererBackend` protocol. V1 ships one real backend (`ViserBackend`)
   plus `MockBackend` for tests. Additional backends (offscreen pyrender,
   playwright capture) land later as drop-ins — see §10.
7. **No viser import at package import time.** `import better_robot`
   and `import better_robot.viewer` work on machines without viser
   installed; the import only fires when `Visualizer.show()` (or direct
   `ViserBackend(...)`) is first called.
8. **The 25-symbol public API ceiling is preserved.** `Visualizer` lives
   at `better_robot.viewer.Visualizer`, not in `better_robot.__all__`.

## 2. Non-goals (V1)

Each of these is a *placeholder* in the source tree (the class or
function exists so future work can fill it in) but raises
`NotImplementedError` pointing at §10 when called:

- Video recording / mp4 output (`VideoRecorder`, `render_trajectory`).
- Offscreen / headless rendering (`OffscreenBackend`, pyrender, EGL).
- Trajectory playback controls beyond "show frame *k*": no speed, loop,
  seek bar, ghosting, trace, manifold interpolation between keyframes.
  Playback steps through integer frame indices only.
- Collision capsule visualisation (`CollisionMode` remains a stub).
- COM marker, residual plot, path trace overlays.
- Multi-robot sessions (`Visualizer.add_robot`).
- Batch-axis picker (a `(B, T, nq)` trajectory renders `b=0` only).
- Camera paths (`CameraPath.orbit`, `CameraPath.follow_frame`).
- Playwright/Chromium browser-capture path.

All of the above have reserved file locations and export names, so when
one is eventually implemented it slots in without refactoring the
surrounding code.

## 3. Directory layout

```
src/better_robot/viewer/
├── __init__.py                # Visualizer + small surface (§9)
├── visualizer.py              # Visualizer — thin facade
├── scene.py                   # Scene — render-mode composition
├── trajectory_player.py       # TrajectoryPlayer — show_frame / play
├── camera.py                  # Camera dataclass (+ CameraPath stubs, §10)
├── themes.py                  # Theme dataclass + DEFAULT_THEME
├── helpers.py                 # xyzw ↔ wxyz quaternion conversions
├── panels.py                  # build_joint_panel (viser-only sliders)
├── interaction.py             # GizmoHandle / FramePicker — stubs (§10)
├── recorder.py                # VideoRecorder + render_trajectory — stubs (§10)
│
├── renderers/
│   ├── __init__.py
│   ├── base.py                # RendererBackend protocol
│   ├── viser_backend.py       # V1 — real implementation (lazy viser import)
│   ├── offscreen_backend.py   # stub (§10)
│   └── testing.py             # MockBackend — records every call, for tests
│
├── render_modes/
│   ├── __init__.py            # MODE_REGISTRY + register_mode
│   ├── base.py                # RenderMode protocol + RenderContext
│   ├── skeleton.py            # V1 — always available
│   ├── urdf_mesh.py           # V1 — trimesh-backed, available iff URDF-loaded
│   └── collision.py           # stub (§10)
│
└── overlays/
    ├── __init__.py
    ├── grid.py                # V1 — ground grid + world triad
    ├── frame_axes.py          # V1 — coordinate triads on named frames
    ├── targets.py             # stub (§10)
    ├── path_trace.py          # stub (§10)
    ├── com.py                 # stub (§10)
    └── residual_plot.py       # stub (§10)
```

## 4. Render modes — the layer system

The central abstraction is `RenderMode`: a small object that knows how to
attach itself to a backend for a given `Model`, how to update its nodes
from a new `Data`, and how to tear itself down. A `Scene` holds any
number of render modes — typically one "primary" mode plus zero or more
overlays — and routes each update to exactly the visible modes.

### 4.1. `RenderMode` protocol

```python
# src/better_robot/viewer/render_modes/base.py

@dataclass
class RenderContext:
    backend: Any                    # RendererBackend instance
    namespace: str                  # e.g. "/robot/skeleton"
    batch_index: int = 0            # always 0 in V1
    theme: Any = None               # Theme instance


@runtime_checkable
class RenderMode(Protocol):
    name: ClassVar[str]             # e.g. "Skeleton"
    description: ClassVar[str]

    @classmethod
    def is_available(cls, model: Model, data: Data) -> bool: ...

    def attach(self, context: RenderContext, model: Model, data: Data) -> None: ...
    def update(self, data: Data) -> None: ...
    def set_visible(self, visible: bool) -> None: ...
    def detach(self) -> None: ...
```

`is_available` is what makes the UI show a toggle for a mode *iff* the
loaded robot can support it. `Scene` queries it once per registered mode
at attach time and quietly skips modes that cannot render the given robot.

### 4.2. `SkeletonMode`

Always available. For every articulated joint `j` (nv > 0), draws a
sphere at `data.joint_pose_world[j, :3]`. For every non-root joint `j`
with parent `p`, draws a cylinder between
`data.joint_pose_world[p, :3]` and `data.joint_pose_world[j, :3]`.
Degenerate (zero-length) cylinders are skipped. This is the only mode
that works for programmatically-built robots (SMPL-like bodies, Python
DSL humanoids) that never go through a URDF.

### 4.3. `URDFMeshMode`

Loads `IRBody.visual_geoms` off the `io.IRModel` captured at load time
(still accessible as `model.meta["ir"]`). Mesh URIs are resolved through
the **`AssetResolver`** stored on the model: `model.meta["asset_resolver"]`
(set by the URDF/MJCF parser at parse time, see
[04_PARSERS.md §11](04_PARSERS.md)). The viewer never re-implements URDF
path logic; it delegates to the resolver:

```python
from better_robot.io.assets import AssetResolver

class URDFMeshMode(RenderMode):
    def __init__(self, *, resolver: AssetResolver | None = None) -> None:
        self._resolver = resolver  # if None, read model.meta["asset_resolver"]
```

Each geom is tessellated through `trimesh` and pushed to the backend via
`add_mesh`. Analytical primitives (`<box>`, `<cylinder>`, `<sphere>`,
`<capsule>`) are handled by `trimesh.creation.*`. Loading failures skip
the geom silently.

### 4.4. Mode registry

```python
MODE_REGISTRY: dict[str, type[RenderMode]] = {}

def register_mode(cls: type[RenderMode]) -> type[RenderMode]:
    MODE_REGISTRY[cls.__name__] = cls
    return cls
```

Built-in modes register themselves at import time. Third-party modes can
use the same decorator, and `Scene` iterates `MODE_REGISTRY.values()` at
attach time so future custom modes show up automatically.

## 5. Overlays

Overlays are `RenderMode`s with a different mental category: they paint
*on top of* a primary mode, they are small, and they compose freely.
V1 ships two real overlays and four stubs:

| Overlay | Status | Pulls from |
|---------|--------|------------|
| `GridOverlay` | **V1** | static — ground plane + world-frame triad |
| `FrameAxesOverlay` | **V1** | `data.frame_pose_world[frame_id]` — triads on named frames |
| `TargetsOverlay` | **V1** | `dict[str, SE3]` — draggable IK gizmos (§7.1) |
| `PathTraceOverlay` | stub (§10) | trajectory + frame_id |
| `ComOverlay` | stub (§10) | `dynamics.center_of_mass` |
| `ResidualPlotOverlay` | stub (§10) | `optim.history` |

## 6. `Scene` — layer composition

```python
# src/better_robot/viewer/scene.py

class Scene:
    def __init__(
        self,
        model: Model,
        *,
        backend: RendererBackend,
        namespace: str = "/robot",
        theme: Theme | None = None,
    ) -> None: ...

    # ─── mode management ───────────────────────────────────────────
    def add_mode(self, mode: RenderMode) -> None: ...
    def remove_mode(self, mode_name: str) -> None: ...
    def available_modes(self) -> list[str]: ...
    def set_mode_visible(self, mode_name: str, visible: bool) -> None: ...

    # ─── state updates ─────────────────────────────────────────────
    def update(self, data: Data) -> None: ...
    def update_from_q(self, q: torch.Tensor) -> None:
        """Run FK (+ update_frame_placements) and push to all modes."""

    @classmethod
    def default(
        cls,
        model: Model,
        *,
        backend: RendererBackend,
        theme: Theme | None = None,
    ) -> "Scene":
        """Attach URDFMeshMode if available else SkeletonMode, plus
        GridOverlay and FrameAxesOverlay."""
```

`Scene.default(...)` is what `Visualizer` uses. Power users who want a
non-default composition construct a `Scene` by hand and `add_mode` their
own.

## 7. `Visualizer` — top-level facade

```python
# src/better_robot/viewer/visualizer.py

class Visualizer:
    """Top-level viser-backed visualiser.

    Example
    -------
    >>> viewer = Visualizer(model, port=8080)
    >>> viewer.show()                   # opens browser, blocks for lifetime
    >>> viewer.update(q)                # single-pose update
    >>> viewer.add_trajectory(traj)     # play through the sequence
    """

    def __init__(
        self,
        model: Model,
        *,
        port: int = 8080,
        theme: Theme | None = None,
    ) -> None: ...

    # ─── lifecycle ────────────────────────────────────────────────
    def show(self, *, block: bool = True) -> None: ...
    def close(self) -> None: ...

    # ─── state ────────────────────────────────────────────────────
    def update(self, q_or_data: torch.Tensor | Data) -> None: ...

    @property
    def last_q(self) -> torch.Tensor: ...

    # ─── trajectory playback (§8) ─────────────────────────────────
    def add_trajectory(self, trajectory: Trajectory) -> "TrajectoryPlayer": ...

    # ─── convenience: draw an IK solution configuration ──────────
    def add_ik_result(self, result: IKResult) -> None: ...

    # ─── interactive IK (§7.1) ────────────────────────────────────
    def add_ik_targets(
        self,
        targets: dict[str, torch.Tensor],
        *,
        on_change: Callable[[dict[str, torch.Tensor]], None] | None = None,
        scale: float = 0.15,
    ) -> "TargetsOverlay": ...
```

`Visualizer` is intentionally thin: it owns one `ViserBackend`, one
`Scene.default`, and forwards `update` to the scene. `ViserBackend` is
constructed lazily inside `_ensure_server()` — the first call to
`show()` / `update()` / `add_trajectory()` triggers the viser import.

`record`, `add_robot`, and `set_batch_index` live as stubs that raise
`NotImplementedError` pointing at §10.

### 7.1. Interactive IK via `add_ik_targets`

`add_ik_targets` drops a `TargetsOverlay` onto the active scene through
the same `Scene.add_mode` lifecycle as any other render mode. Each
target renders as a frame triad (visible everywhere) plus a draggable
SE(3) transform control (on interactive backends only). On every drag
the overlay updates its internal targets dict and fires the caller's
`on_change` hook with the full updated dict — the caller re-solves IK
and pushes the new configuration via `Visualizer.update`.

```python
viewer = Visualizer(model)
viewer.update(q0)

def on_move(new_targets):
    r = br.solve_ik(model, new_targets, initial_q=viewer.last_q)
    viewer.update(r.q)

viewer.add_ik_targets({ee_frame: T_target}, on_change=on_move)
viewer.show()
```

`examples/01_basic_ik.py` is the single-target Panda version and
`examples/02_g1_ik.py` is the floating-base G1 version with four
simultaneous whole-body targets. Both warm-start each IK solve from
the previous configuration so interactive dragging converges in
milliseconds.

On non-interactive backends (`MockBackend`, future `OffscreenBackend`)
the overlay still draws the frame triads but no gizmos — useful for
static rendering of an IK target dataset. Callbacks never fire on
those backends.

## 8. Trajectory playback

```python
# src/better_robot/viewer/trajectory_player.py

class TrajectoryPlayer:
    """Drives a Scene through a (B, T, nq) Trajectory frame-by-frame.

    V1 is deliberately tiny: show a given frame index, or loop through
    all frames at a fixed fps. No scrub bar, no speed, no loop toggle,
    no ghost/trace, no manifold interpolation between keyframes.
    """

    def __init__(self, scene: Scene, trajectory: Trajectory) -> None: ...

    @property
    def horizon(self) -> int: ...

    def show_frame(self, k: int) -> None:
        """Push the k-th keyframe of the trajectory to the scene."""

    def play(self, *, fps: float = 30.0) -> None:
        """Blocking loop: show_frame(0), …, show_frame(T-1).

        Sleeps 1/fps between frames. Returns when the sequence is done.
        Ctrl-C interrupts cleanly.
        """
```

That's it — no seek, no step, no speed, no loop. Users who want looping
wrap `play()` in a `while True:`. Users who want scrubbing get it back
once §10's transport-control milestone lands.

Because V1 plays only integer frame indices, there is no need for
`model.integrate`/`model.difference`-based interpolation between
keyframes — that correctness requirement returns when seek-to-arbitrary-
cursor comes back with the transport controls.

## 9. `RendererBackend` protocol

```python
# src/better_robot/viewer/renderers/base.py

@runtime_checkable
class RendererBackend(Protocol):
    is_interactive: bool              # True for viser
    supports_gui: bool                # True iff GUI widgets are available

    # ─── scene graph ───────────────────────────────────────────────
    def add_mesh(self, name, vertices, faces, *, rgba, parent=None) -> None: ...
    def add_sphere(self, name, *, radius, rgba, parent=None) -> None: ...
    def add_cylinder(self, name, *, radius, length, rgba, parent=None) -> None: ...
    def add_capsule(self, name, *, radius, length, rgba, parent=None) -> None: ...
    def add_frame(self, name, *, axes_length=0.1) -> None: ...

    def remove(self, name) -> None: ...
    def set_transform(self, name, pose) -> None:  # pose is (7,) xyzw
        ...
    def set_visible(self, name, visible) -> None: ...

    # ─── optional GUI hooks (viser only) ──────────────────────────
    def add_gui_button(self, label, callback) -> None: ...
    def add_gui_slider(self, label, *, min, max, step, value, callback) -> None: ...
    def add_gui_checkbox(self, label, *, value, callback) -> None: ...
```

The protocol is deliberately **small**: mesh, sphere, cylinder, capsule,
frame, transform/visibility updates, plus three GUI primitives. This is
the full surface V1's render modes and overlays need.

`capture_frame` and `set_camera` are *not* on the V1 protocol — they
come back with §10's offscreen backend, and `ViserBackend` will gain a
capture path via playwright at the same time.

### 9.1. `ViserBackend`

The one real V1 backend. Lazily imports `viser` on construction and
maps each backend primitive onto the corresponding `server.scene.*`
call. `wxyz`/`xyzw` quaternion conversions go through
`viewer/helpers.py`. GUI hooks (`add_gui_button`, slider, checkbox) map
onto `server.gui.*`.

### 9.2. `MockBackend`

Lives at `renderers/testing.py`. Pure-Python implementation of the
protocol — every call is recorded into a list for test assertions.
Render-mode unit tests attach a mode to a `MockBackend`, call
`update(data)`, and check the recorded calls. No viser, no pyrender,
no ffmpeg needed.

## 10. Future expansion

Everything below is a placeholder in the source tree: the file, class,
or function exists so the surrounding code has a named target, but the
body raises `NotImplementedError("see docs/design/12_VIEWER.md §10.*")`.

### 10.1. Video recording

`viewer/recorder.py` defines `VideoRecorder` and `render_trajectory(...)`
as placeholders. The future implementation routes frames through the
`RendererBackend` capture path (§10.2) and encodes via
`imageio-ffmpeg`. The top-level one-liner would be
`render_trajectory(model, traj, "out.mp4")`.

The `test_only_recorder_imports_imageio` layer rule in
`tests/test_layer_dependencies.py` reserves `imageio` imports for
`recorder.py`, so this future work lands without leaking dependencies.

### 10.2. Offscreen (headless) renderer

`viewer/renderers/offscreen_backend.py` defines `OffscreenBackend` as a
placeholder. The future implementation uses pyrender with an EGL
(Linux) or OSMesa (pure-CPU) context. When it lands, `RendererBackend`
gains a `capture_frame() → ndarray` method and a `set_camera(camera)`
method, and the linter rule `test_only_offscreen_backend_imports_pyrender`
enforces that pyrender stays in that one file.

### 10.3. Sequence playback transport controls

`TrajectoryPlayer` gains `play/pause/seek/seek_frame/step/set_speed/
set_loop`, plus ghost (faded skeleton every N frames) and trace (swept
frame path). The cursor becomes a normalised float `t ∈ [0, 1]` with
`model.integrate`/`model.difference`-based interpolation between
keyframes so free-flyer bases and ball joints animate along geodesics.
A viser timeline widget (`panels.py → build_playback_panel`) drives it
from the GUI.

### 10.4. Collision capsules

`render_modes/collision.py → CollisionMode` becomes a real mode backed
by `RobotCollision.world_capsules(data)` (see 09 §5). Active pairs
(under the margin) are drawn in a warning colour.

### 10.5. Secondary overlays

`overlays/com.py`, `overlays/path_trace.py`, `overlays/residual_plot.py`
all become real overlays. COM requires `dynamics.center_of_mass` (from
milestone D1). Residual plot requires `optim.history`. Path trace is a
`frame_id + Trajectory` + a series of sphere markers on the world-frame
sweep.

### 10.6. Camera paths

`viewer/camera.py → CameraPath.orbit / follow_frame / static` become
real, and `RendererBackend.set_camera` comes back. These chain
naturally off §10.1–§10.2 since the primary use case is cinematic video
rendering.

### 10.7. Multi-robot sessions

`Visualizer.add_robot(model, *, name, namespace)` gets a real body,
with one `Scene` per robot and independent UI groups. Requires basic
scene-namespace routing that the V1 `Visualizer` does not need.

### 10.8. Batch-axis picker

`Scene.set_batch_index(idx)` and a `(B, T, nq)` picker in the UI panel.
V1 always renders `batch_index=0`.

## 11. Testing story

V1 tests live under `tests/viewer/`:

```
tests/viewer/
├── test_helpers_quat_conversions.py   — xyzw ↔ wxyz roundtrip
├── test_scene_mode_toggling.py        — add/remove/visible on MockBackend
├── test_skeleton_mode_math.py         — sphere/cylinder transforms match data.joint_pose_world
├── test_urdf_mode_availability.py     — availability contract on URDF vs builder
├── test_collision_mode.py             — stub behaviour (NotImplementedError)
├── test_targets_overlay.py            — TargetsOverlay attach/callback/detach
├── test_trajectory_player_basic.py    — show_frame + play() non-crashing
└── test_backend_mock.py               — SkeletonMode through MockBackend
```

Strategy:

1. **Backend-agnostic via `MockBackend`.** Render modes write into a
   pure-Python `RendererBackend` implementation that records every
   call. All render-mode math can be checked without viser/pyrender.
2. **Mode availability is a contract.** Every real mode has a test for
   `is_available` on (a) a URDF robot and (b) a programmatic builder.
3. **Interactive smoke is optional.** A future `test_smoke_viser.py`
   spins a real viser server, attaches a `SkeletonMode`, pushes one
   update, and tears down. Skipped unless `viser` is importable.

Future milestones re-enable (and grow) these test files:

- §10.1 → `test_render_trajectory_headless.py`,
  `test_render_trajectory_no_ffmpeg.py`, `test_write_frame.py`
- §10.2 → `test_backend_matrix.py` (Mock ↔ Offscreen)
- §10.3 → `test_trajectory_player_manifold.py` (manifold interpolation)
- §10.5 → `test_path_trace_overlay.py`, `test_com_overlay.py`,
  `test_residual_plot_overlay.py`
- §10.6 → `test_camera_path.py`

## 12. Dependency hygiene

| Import | Allowed in | Why |
|--------|------------|-----|
| `viser` | `viewer/renderers/viser_backend.py` only | V1 backend implementation |
| `pyrender` | `viewer/renderers/offscreen_backend.py` only (future) | §10.2 |
| `trimesh` | `viewer/render_modes/urdf_mesh.py` only | Mesh I/O, one module |
| `imageio` / `imageio-ffmpeg` | `viewer/recorder.py` only (future) | §10.1 |
| `playwright` | `viewer/renderers/viser_backend.py` (future, lazy) | Browser capture (§10.1) |
| `numpy` | anywhere in `viewer/*` | Conversion layer |
| `torch` | anywhere in `viewer/*` | Canonical type |

**Render modes** in `viewer/render_modes/*` may import **neither**
`viser` nor `pyrender` — they only talk to the `RendererBackend`
protocol. This is what keeps the same mode code compatible with every
future backend.

`tests/test_layer_dependencies.py` enforces these rules with the same
AST walker as the existing `test_only_lie_imports_pypose` check:

- `test_only_viser_backend_imports_viser`
- `test_only_offscreen_backend_imports_pyrender`
- `test_only_urdf_mesh_imports_trimesh`
- `test_only_recorder_imports_imageio`

All four are green in V1 — the future-work files simply don't import
their forbidden library yet.

## 13. Public API

The viewer exports live under `better_robot.viewer.*`, **not** in
`better_robot.__all__`. The 25-symbol public API ceiling from
01 ARCHITECTURE is non-negotiable.

```python
# src/better_robot/viewer/__init__.py

from . import helpers
from .camera import Camera
from .render_modes.base import RenderContext, RenderMode
from .render_modes.skeleton import SkeletonMode
from .render_modes.urdf_mesh import URDFMeshMode
from .renderers.base import RendererBackend
from .renderers.viser_backend import ViserBackend
from .scene import Scene
from .trajectory_player import TrajectoryPlayer
from .visualizer import Visualizer

__all__ = [
    # Interactive facade
    "Visualizer",
    "Scene",
    # Render modes (V1)
    "RenderMode",
    "RenderContext",
    "SkeletonMode",
    "URDFMeshMode",
    # Renderer backend (V1)
    "RendererBackend",
    "ViserBackend",
    # Playback + misc
    "TrajectoryPlayer",
    "Camera",
    "helpers",
]
```

Future milestones extend the export list without rearranging: §10.1
brings back `render_trajectory` and `VideoRecorder`; §10.2 brings back
`OffscreenBackend`; §10.4 brings back `CollisionMode`; §10.6 brings
back `CameraPath`. Each is already a placeholder class so downstream
callers can `from better_robot.viewer import CollisionMode` today and
get a `NotImplementedError` at call time rather than an `ImportError`.

## 14. Definition of done (V1)

V1 is done when:

1. Every public symbol in §13 is implemented and has a docstring.
2. `tests/viewer/` as listed in §11 is green.
3. `examples/01_basic_ik.py` can be updated to call
   `Visualizer(model).show()` and display the Panda.
4. Programmatic / URDF-less robots (e.g. SMPL-like body) render as a
   skeleton through `SkeletonMode`.
5. The layer-dependency linter (§12) is green.
6. `docs/CHANGELOG.md` has an entry pointing at this file and naming
   V1 explicitly — so the "what's left" list in §10 is discoverable.
