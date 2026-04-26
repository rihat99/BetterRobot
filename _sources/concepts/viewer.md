# Viewer

The viewer is the topmost layer in the DAG: nothing imports from it.
Its job is to turn `Model`, `Data`, and `Trajectory` into something a
user can see in a browser — without dragging any visualisation
dependency into the core. A user who never calls
`Visualizer.show()` should not pay for `viser`, `trimesh`, or any of
the rendering machinery on `import better_robot`.

The deliberate choice is to ship a small, opinionated V1: an
interactive viser-backed renderer with two render modes (skeleton,
URDF mesh), a small overlay set (grid, frame axes, force vectors,
draggable IK target gizmos), and a minimal trajectory player that
can `show_frame(k)` and `play(fps=30)`. Everything more ambitious —
video recording, offscreen rendering, scrub-bar transport controls,
COM / path-trace / residual-plot overlays, multi-robot sessions,
camera paths — is reserved as a placeholder. The placeholders raise
`NotImplementedError` with a pointer to {doc}`/reference/roadmap`,
so user code that imports them keeps working at static-analysis time
even before the body lands.

This chapter covers what V1 does and the seams that make later
expansion additive rather than disruptive.

## V1 scope

- **Interactive browser visualisation only.** `Visualizer(model).show()`
  opens a viser session and draws the robot. That is the one happy
  path.
- **Two render modes.** `SkeletonMode` works on any `Model`.
  `URDFMeshMode` is selected automatically when the robot was loaded
  from a URDF / MJCF (i.e. `model.meta["ir"]` is populated).
- **`Scene` composes render modes.** Users (or `Scene.default`) attach
  zero-or-more modes; the scene routes kinematics updates to each
  one. This is the extensibility point for future modes.
- **Draggable IK target gizmos.** `add_ik_targets` drops a
  `TargetsOverlay` that renders each target as a frame triad plus a
  draggable SE(3) transform control. On every drag, an `on_change`
  callback fires with the updated targets dict, so callers can
  re-solve IK in the loop.
- **Straight-through trajectory playback.** A `TrajectoryPlayer`
  pushes a given frame of a `(B, T, nq)` `Trajectory` to the scene.
  No scrub bar, no speed control, no loop toggle — just `show_frame`
  and `play`. (Transport controls are reserved for a later
  milestone.)
- **Backend-agnostic render modes.** Modes speak only to the
  `RendererBackend` Protocol. V1 ships one real backend
  (`ViserBackend`) plus `MockBackend` for tests. Additional backends
  (offscreen pyrender, playwright capture) land later as drop-ins.
- **No viser import at package import time.** `import better_robot`
  and `import better_robot.viewer` work on machines without viser
  installed; the import only fires when `Visualizer.show()` (or
  direct `ViserBackend(...)`) is first called.
- **Public API ceiling preserved.** `Visualizer` lives at
  `better_robot.viewer.Visualizer`, not in `better_robot.__all__`.

## Directory layout

```
src/better_robot/viewer/
├── visualizer.py              # Visualizer — thin facade
├── scene.py                   # Scene — render-mode composition
├── trajectory_player.py       # TrajectoryPlayer — show_frame / play
├── camera.py                  # Camera dataclass (+ CameraPath stubs)
├── themes.py                  # Theme dataclass + DEFAULT_THEME
├── helpers.py                 # xyzw ↔ wxyz quaternion conversions
├── panels.py                  # build_joint_panel (viser-only sliders)
├── interaction.py             # GizmoHandle / FramePicker
├── recorder.py                # VideoRecorder + render_trajectory (stub)
│
├── renderers/
│   ├── base.py                # RendererBackend protocol
│   ├── viser_backend.py       # V1 — real implementation (lazy viser import)
│   ├── offscreen_backend.py   # stub
│   └── testing.py             # MockBackend — records every call
│
├── render_modes/
│   ├── base.py                # RenderMode protocol + RenderContext
│   ├── skeleton.py            # V1 — always available
│   ├── urdf_mesh.py           # V1 — trimesh-backed, available iff URDF-loaded
│   └── collision.py           # stub
│
└── overlays/
    ├── grid.py                # V1 — ground grid + world triad
    ├── frame_axes.py          # V1 — coordinate triads on named frames
    ├── targets.py             # V1 — draggable IK gizmos
    ├── force_vectors.py       # V1 — force-vector arrows
    ├── path_trace.py          # stub
    ├── com.py                 # stub
    └── residual_plot.py       # stub
```

## Render modes — the layer system

The central abstraction is `RenderMode`: a small object that knows
how to attach itself to a backend for a given `Model`, how to update
its nodes from a new `Data`, and how to tear itself down. A `Scene`
holds any number of render modes — typically one "primary" mode plus
zero or more overlays — and routes each update to exactly the
visible modes.

### `RenderMode` Protocol

```python
@dataclass
class RenderContext:
    backend: Any                    # RendererBackend instance
    namespace: str                  # e.g. "/robot/skeleton"
    batch_index: int = 0
    theme: Any = None

@runtime_checkable
class RenderMode(Protocol):
    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    def is_available(cls, model: Model, data: Data) -> bool: ...

    def attach(self, context: RenderContext, model: Model, data: Data) -> None: ...
    def update(self, data: Data) -> None: ...
    def set_visible(self, visible: bool) -> None: ...
    def detach(self) -> None: ...
```

Source: `src/better_robot/viewer/render_modes/base.py`. `is_available`
is what makes the UI show a toggle for a mode *iff* the loaded robot
can support it. `Scene` queries it once per registered mode at
attach time and quietly skips modes that cannot render the given
robot.

### `SkeletonMode`

Always available. For every articulated joint `j` (`nv > 0`), draws
a sphere at `data.joint_pose_world[j, :3]`. For every non-root joint
with parent `p`, draws a cylinder between
`data.joint_pose_world[p, :3]` and `data.joint_pose_world[j, :3]`.
Degenerate (zero-length) cylinders are skipped. This is the only
mode that works for programmatically-built robots (SMPL-like bodies,
Python-DSL humanoids) that never go through a URDF.

### `URDFMeshMode`

Loads `IRBody.visual_geoms` off the `IRModel` captured at load time
(`model.meta["ir"]`). Mesh URIs resolve through the `AssetResolver`
on `model.meta["asset_resolver"]` (set by the URDF / MJCF parser at
parse time). The viewer never re-implements URDF path logic; it
delegates to the resolver:

```python
class URDFMeshMode(RenderMode):
    def __init__(self, *, resolver: AssetResolver | None = None) -> None:
        self._resolver = resolver  # if None, read model.meta["asset_resolver"]
```

Each geom is tessellated through `trimesh` and pushed to the backend
via `add_mesh`. Analytical primitives (`<box>`, `<cylinder>`,
`<sphere>`, `<capsule>`) are handled by `trimesh.creation.*`.
Loading failures skip the geom silently rather than crashing the
whole viewer.

### Mode registry

```python
MODE_REGISTRY: dict[str, type[RenderMode]] = {}

def register_mode(cls: type[RenderMode]) -> type[RenderMode]:
    MODE_REGISTRY[cls.__name__] = cls
    return cls
```

Built-in modes register themselves at import time. Third-party modes
use the same decorator; `Scene` iterates `MODE_REGISTRY.values()` at
attach time, so future custom modes show up automatically.

## Overlays

Overlays are `RenderMode`s with a different mental category: they
paint *on top of* a primary mode, they are small, and they compose
freely.

| Overlay | Status | Pulls from |
|---------|--------|------------|
| `GridOverlay` | V1 | static — ground plane + world-frame triad |
| `FrameAxesOverlay` | V1 | `data.frame_pose_world[frame_id]` |
| `TargetsOverlay` | V1 | `dict[str, SE3]` — draggable IK gizmos |
| `ForceVectorsOverlay` | V1 | per-frame wrench vectors |
| `PathTraceOverlay` | stub | trajectory + frame_id |
| `ComOverlay` | stub | `dynamics.center_of_mass` |
| `ResidualPlotOverlay` | stub | `optim.history` |

## `Scene` — layer composition

```python
class Scene:
    def __init__(
        self,
        model: Model,
        *,
        backend: RendererBackend,
        namespace: str = "/robot",
        theme: Theme | None = None,
    ) -> None: ...

    def add_mode(self, mode: RenderMode) -> None: ...
    def remove_mode(self, mode_name: str) -> None: ...
    def available_modes(self) -> list[str]: ...
    def set_mode_visible(self, mode_name: str, visible: bool) -> None: ...

    def update(self, data: Data) -> None: ...
    def update_from_q(self, q: torch.Tensor) -> None:
        """Run FK (+ update_frame_placements) and push to all modes."""

    @classmethod
    def default(cls, model, *, backend, theme=None) -> "Scene":
        """Attach URDFMeshMode if available else SkeletonMode, plus
        GridOverlay and FrameAxesOverlay."""
```

`Scene.default(...)` is what `Visualizer` uses. Power users who want
a non-default composition construct a `Scene` by hand and `add_mode`
their own.

## `Visualizer` — top-level facade

```python
class Visualizer:
    """Top-level viser-backed visualiser.

    Example
    -------
    >>> viewer = Visualizer(model, port=8080)
    >>> viewer.show()                   # opens browser, blocks for lifetime
    >>> viewer.update(q)                # single-pose update
    >>> viewer.add_trajectory(traj)     # play through the sequence
    """

    def __init__(self, model: Model, *, port: int = 8080,
                 theme: Theme | None = None) -> None: ...

    def show(self, *, block: bool = True) -> None: ...
    def close(self) -> None: ...

    def update(self, q_or_data: torch.Tensor | Data) -> None: ...

    @property
    def last_q(self) -> torch.Tensor: ...

    def add_trajectory(self, trajectory: Trajectory) -> "TrajectoryPlayer": ...

    def add_ik_result(self, result: IKResult) -> None: ...

    def add_ik_targets(
        self,
        targets: dict[str, torch.Tensor],
        *,
        on_change: Callable[[dict[str, torch.Tensor]], None] | None = None,
        scale: float = 0.15,
    ) -> "TargetsOverlay": ...
```

`Visualizer` is intentionally thin: it owns one `ViserBackend`, one
`Scene.default`, and forwards `update` to the scene. `ViserBackend`
is constructed lazily inside `_ensure_server()` — the first call to
`show()` / `update()` / `add_trajectory()` triggers the viser
import. `record`, `add_robot`, and `set_batch_index` live as stubs
that raise `NotImplementedError` pointing at the roadmap.

### Interactive IK with `add_ik_targets`

`add_ik_targets` drops a `TargetsOverlay` onto the active scene
through the same `Scene.add_mode` lifecycle as any other render mode.
Each target renders as a frame triad (visible everywhere) plus a
draggable SE(3) transform control (on interactive backends only). On
every drag the overlay updates its internal targets dict and fires
the caller's `on_change` hook with the full updated dict — the
caller re-solves IK and pushes the new configuration via
`Visualizer.update`.

```python
viewer = Visualizer(model)
viewer.update(q0)

def on_move(new_targets):
    r = br.solve_ik(model, new_targets, initial_q=viewer.last_q)
    viewer.update(r.q)

viewer.add_ik_targets({ee_frame: T_target}, on_change=on_move)
viewer.show()
```

`examples/01_basic_ik.py` is the single-target Panda version;
`examples/02_g1_ik.py` is the floating-base G1 version with four
simultaneous whole-body targets. Both warm-start each IK solve from
the previous configuration so interactive dragging converges in
milliseconds.

On non-interactive backends (`MockBackend`) the overlay still draws
the frame triads but no gizmos — useful for static rendering of an
IK target dataset. Callbacks never fire on those backends.

## Trajectory playback

```python
class TrajectoryPlayer:
    """Drives a Scene through a (B, T, nq) Trajectory frame-by-frame.

    V1 is deliberately tiny: show a given frame index, or loop
    through all frames at a fixed fps. No scrub bar, no speed, no
    loop toggle, no ghost / trace, no manifold interpolation between
    keyframes.
    """

    def __init__(self, scene: Scene, trajectory: Trajectory) -> None: ...

    @property
    def horizon(self) -> int: ...

    def show_frame(self, k: int) -> None: ...

    def play(self, *, fps: float = 30.0) -> None: ...
```

That is it — no seek, no step, no speed, no loop. Users who want
looping wrap `play()` in a `while True:`. Users who want scrubbing
get it back when the transport-controls milestone lands.

Because V1 plays only integer frame indices, there is no need for
`model.integrate` / `model.difference`-based interpolation between
keyframes — that correctness requirement returns when seek-to-arbitrary-
cursor comes back with the transport controls.

## `RendererBackend` Protocol

```python
@runtime_checkable
class RendererBackend(Protocol):
    is_interactive: bool
    supports_gui: bool

    def add_mesh(self, name, vertices, faces, *, rgba, parent=None) -> None: ...
    def add_sphere(self, name, *, radius, rgba, parent=None) -> None: ...
    def add_cylinder(self, name, *, radius, length, rgba, parent=None) -> None: ...
    def add_capsule(self, name, *, radius, length, rgba, parent=None) -> None: ...
    def add_frame(self, name, *, axes_length=0.1) -> None: ...

    def remove(self, name) -> None: ...
    def set_transform(self, name, pose) -> None:    # pose is (7,) xyzw
        ...
    def set_visible(self, name, visible) -> None: ...

    # Optional GUI hooks (viser only)
    def add_gui_button(self, label, callback) -> None: ...
    def add_gui_slider(self, label, *, min, max, step, value, callback) -> None: ...
    def add_gui_checkbox(self, label, *, value, callback) -> None: ...
```

Source: `src/better_robot/viewer/renderers/base.py`. The protocol is
deliberately small: mesh, sphere, cylinder, capsule, frame, transform
/ visibility updates, plus three GUI primitives. `capture_frame` and
`set_camera` are *not* on the V1 protocol — they come back with the
offscreen backend milestone.

### `ViserBackend`

The one real V1 backend. Lazily imports `viser` on construction and
maps each backend primitive onto the corresponding `server.scene.*`
call. `wxyz` / `xyzw` quaternion conversions go through
`viewer/helpers.py`. GUI hooks (`add_gui_button`, slider, checkbox)
map onto `server.gui.*`.

### `MockBackend`

Pure-Python implementation of the protocol — every call is recorded
into a list for test assertions. Render-mode unit tests attach a
mode to a `MockBackend`, call `update(data)`, and check the recorded
calls. No viser, no pyrender, no ffmpeg needed.

## Future expansion

Everything below is a placeholder in the source tree: the file,
class, or function exists so the surrounding code has a named
target, but the body raises `NotImplementedError` pointing at the
roadmap.

- **Video recording.** `viewer/recorder.py` defines `VideoRecorder`
  and `render_trajectory`. The future implementation routes frames
  through the `RendererBackend` capture path and encodes via
  `imageio-ffmpeg`.
- **Offscreen rendering.** `viewer/renderers/offscreen_backend.py`
  defines `OffscreenBackend`. The future implementation uses
  `pyrender` with EGL (Linux) or OSMesa (pure-CPU). When it lands,
  `RendererBackend` gains `capture_frame() → ndarray` and
  `set_camera(camera)`.
- **Transport controls.** `TrajectoryPlayer` gains `play / pause /
  seek / seek_frame / step / set_speed / set_loop`, plus ghost
  (faded skeleton every N frames) and trace (swept frame path). The
  cursor becomes a normalised float `t ∈ [0, 1]` with manifold
  interpolation between keyframes (geodesic for SE(3) / SO(3)).
- **Collision capsules.** `CollisionMode` becomes a real mode backed
  by `RobotCollision.world_capsules(data)`.
- **Secondary overlays.** `ComOverlay` requires `center_of_mass`
  (live); `ResidualPlotOverlay` requires `optim.history`;
  `PathTraceOverlay` is `frame_id + Trajectory` plus a series of
  sphere markers on the world-frame sweep.
- **Camera paths.** `CameraPath.orbit` / `follow_frame` / `static`
  become real, and `RendererBackend.set_camera` comes back.
- **Multi-robot sessions.** `Visualizer.add_robot(model, *, name,
  namespace)` gets a real body, with one `Scene` per robot and
  independent UI groups.
- **Batch-axis picker.** `Scene.set_batch_index(idx)` and a `(B, T,
  nq)` picker in the UI panel.

## Dependency hygiene

| Import | Allowed in | Why |
|--------|------------|-----|
| `viser` | `viewer/renderers/viser_backend.py` only | V1 backend |
| `pyrender` | `viewer/renderers/offscreen_backend.py` only | Future offscreen |
| `trimesh` | `viewer/render_modes/urdf_mesh.py` only | Mesh I/O |
| `imageio` / `imageio-ffmpeg` | `viewer/recorder.py` only | Future video |
| `playwright` | `viewer/renderers/viser_backend.py` (lazy) | Future browser capture |
| `numpy` | anywhere in `viewer/*` | Conversion layer |
| `torch` | anywhere in `viewer/*` | Canonical type |

**Render modes** in `viewer/render_modes/*` may import neither
`viser` nor `pyrender` — they only talk to the `RendererBackend`
Protocol. This is what keeps the same mode code compatible with
every future backend. `tests/contract/test_layer_dependencies.py`
enforces these rules with the same AST walker as the rest of the
DAG.

## Public API

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
    "Visualizer", "Scene",
    "RenderMode", "RenderContext",
    "SkeletonMode", "URDFMeshMode",
    "RendererBackend", "ViserBackend",
    "TrajectoryPlayer", "Camera", "helpers",
]
```

The viewer exports live under `better_robot.viewer.*`, not in
`better_robot.__all__`. The 26-symbol public-API ceiling is
non-negotiable.

## Sharp edges

- **`viser` is lazily imported.** A machine without viser installed
  can `import better_robot.viewer` cleanly; only `Visualizer.show()`
  triggers the actual import.
- **Quaternion conventions.** The library is scalar-last
  `[qx, qy, qz, qw]`; viser is scalar-first `[w, x, y, z]`. The
  conversion lives in `viewer/helpers.py` so it happens in exactly
  one place.
- **`URDFMeshMode` requires the IR.** A robot built programmatically
  through `ModelBuilder` does not have URDF visual geoms; render it
  with `SkeletonMode` instead.
- **`TrajectoryPlayer.play(fps=)` is blocking.** Wrap it in a
  thread or use the (future) transport-control widget if you need
  asynchronous playback.
- **Stubs raise immediately.** Calling `VideoRecorder.write_frame`
  or `OffscreenBackend.capture_frame` raises `NotImplementedError`
  with a roadmap link, not at import time.

## Where to look next

- {doc}`tasks` — `solve_ik` returns an `IKResult` whose
  `frame_pose("name")` is what the viewer plots.
- {doc}`/conventions/extension` §8 — recipe for adding a custom
  render mode.
- {doc}`/reference/roadmap` — what is currently stubbed in the
  viewer.
