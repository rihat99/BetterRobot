# Viewer

The viewer is the topmost layer in the dependency graph: nothing in the
robotics core imports it. It turns `Model`, `Data`, and `Trajectory` objects
into an interactive browser scene without making rendering dependencies part
of the core import path.

Names such as `RendererBackend`, `ViserBackend`, and `MockBackend` in this
chapter refer to scene rendering. They are unrelated to the Torch and Warp
algorithm lanes.

## Shipped surface

- `Visualizer` is the interactive viser-backed facade.
- `Scene` composes render modes and overlays for one robot.
- `SkeletonMode` works for every `Model`; `URDFMeshMode` renders visual
  geometry when parser IR is available.
- `GridOverlay`, `FrameAxesOverlay`, `TargetsOverlay`, and
  `ForceVectorsOverlay` are live composable overlays.
- `TrajectoryPlayer` supports integer-frame `show_frame`, its `seek_frame`
  alias, and blocking straight-through `play`.
- `PrimitiveHandle` provides backend-neutral colour and scale updates for
  rendered joint spheres.
- `ViserBackend` is the interactive backend. `MockBackend` is the headless,
  in-memory implementation used by tests.

Unsupported capabilities are omitted from the public surface rather than
represented by importable placeholders.

## Directory layout

```text
src/better_robot/viewer/
├── __init__.py
├── visualizer.py              # Interactive facade
├── scene.py                   # Render-mode composition
├── trajectory_player.py       # Integer-frame playback
├── primitive.py               # Public primitive style handle
├── themes.py                  # Theme and DEFAULT_THEME
├── helpers.py                 # xyzw ↔ wxyz conversions
├── panels.py                  # Interactive joint sliders
├── renderers/
│   ├── base.py                # RendererBackend protocol
│   ├── viser_backend.py       # Interactive implementation
│   └── testing.py             # Headless MockBackend
├── render_modes/
│   ├── base.py                # RenderMode and RenderContext
│   ├── skeleton.py            # Model-independent skeleton
│   └── urdf_mesh.py           # Parser-IR visual geometry
└── overlays/
    ├── grid.py
    ├── frame_axes.py
    ├── targets.py
    └── force_vectors.py
```

## Render modes

`RenderMode` is the layer contract. A mode knows whether it can render a
model, how to attach its primitives to a backend, how to update those
primitives from new kinematics data, and how to detach them.

```python
@dataclass
class RenderContext:
    backend: RendererBackend
    namespace: str
    batch_index: int = 0
    theme: Theme | None = None

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

### `SkeletonMode`

`SkeletonMode` is always available. It draws a sphere for every articulated
joint and a cylinder from each non-root joint to its parent. Each update reads
`data.joint_pose_world`; zero-length cylinders are skipped.

The sphere for an articulated joint has a public styling seam:

```python
handle = scene.joint_primitive(joint_id)
handle.set_color((0.2, 0.5, 0.9, 1.0))
handle.set_scale(1.4)
```

`Scene.joint_primitive` attaches a skeleton layer lazily when the default
scene chose mesh rendering as its primary mode. Callers therefore do not need
to inspect the scene's modes or renderer backend.

### `URDFMeshMode`

`URDFMeshMode` reads `IRBody.visual_geoms` from the parser IR stored in
`model.meta["ir"]`. Mesh URIs are resolved through the `AssetResolver` stored
in `model.meta["asset_resolver"]`, or through an explicit resolver passed to
the mode. The viewer does not duplicate parser path rules.

Analytical primitives are tessellated through `trimesh`. A visual geometry
that cannot be loaded is skipped without preventing the rest of the robot
from rendering.

### Registry

Built-in render modes are registered in `MODE_REGISTRY`. Third-party code can
use `register_mode` for discovery, then explicitly add an instance with
`Scene.add_mode`.

## Overlays

Overlays implement the same lifecycle as render modes and compose on top of a
primary robot representation.

| Overlay | Input |
|---|---|
| `GridOverlay` | Static ground grid and world triad |
| `FrameAxesOverlay` | `data.frame_pose_world` |
| `TargetsOverlay` | Named SE(3) targets, with viser transform controls |
| `ForceVectorsOverlay` | Per-frame anchor positions and force vectors |

`TargetsOverlay` fires an optional callback after an interactive target drag.
On `MockBackend` it still draws target frames, but does not create interactive
controls.

`ForceVectorsOverlay.update_frame(anchors, forces)` is driven by the caller's
animation loop. Near-zero vectors are hidden; nonzero vectors are drawn as
world-frame arrows with length proportional to force magnitude.

## `Scene`

`Scene` owns one model, one backend, and a set of modes. It routes a kinematics
update only to attached, available, visible modes.

```python
class Scene:
    def add_mode(self, mode: RenderMode) -> None: ...
    def remove_mode(self, mode_name: str) -> None: ...
    def available_modes(self) -> list[str]: ...
    def set_mode_visible(self, mode_name: str, visible: bool) -> None: ...

    def update(self, data: Data) -> None: ...
    def update_from_q(self, q: torch.Tensor) -> None: ...
    def joint_primitive(self, joint_id: int) -> PrimitiveHandle: ...

    @classmethod
    def default(cls, model, *, backend, theme=None) -> "Scene": ...
```

`Scene.default` selects `URDFMeshMode` when visual parser IR exists and
otherwise selects `SkeletonMode`. It also adds the grid and frame-axes
overlays.

Power users can construct `Scene(model, backend=...)` directly and add only
the layers they need.

## `Visualizer`

`Visualizer` owns a `ViserBackend` and a default `Scene`. Backend creation is
lazy: importing `better_robot.viewer` does not import viser, and the server is
created only when an operation needs the scene.

```python
viewer = Visualizer(model, port=8080)
viewer.update(q)

player = viewer.add_trajectory(trajectory)
viewer.show_frame(12)              # same playback path as player.show_frame(12)

joint = viewer.joint_primitive(2)
joint.set_color((0.9, 0.25, 0.15, 1.0))
joint.set_scale(1.25)

viewer.show()
```

The facade also provides:

- `add_ik_result(result)` to display a solved configuration;
- `add_ik_targets(targets, on_change=...)` for an interactive IK loop;
- `current_player()` to read the attached player;
- `scene()` for callers that need to add a live mode or overlay; and
- `close()` for best-effort session teardown.

### Interactive IK

```python
viewer = Visualizer(model)
viewer.update(q0)

def on_move(new_targets):
    result = br.solve_ik(model, new_targets, initial_q=viewer.last_q)
    viewer.update(result.q)

viewer.add_ik_targets({end_effector: target_pose}, on_change=on_move)
viewer.show()
```

On the interactive backend, the draggable transform control is itself the
target visual; dragging it updates the target dictionary and calls
`on_change`. A non-interactive backend renders a static frame triad instead.

## Trajectory playback

`TrajectoryPlayer` drives batch element zero of a `(B, T, nq)` `Trajectory`
by integer frame index.

```python
player = TrajectoryPlayer(scene, trajectory)
player.show_frame(4)
player.seek_frame(4)  # alias
player.play(fps=30.0)
```

Construction pushes frame zero immediately. Frame indices are clamped to the
valid range. `play` is a blocking pass from frame zero through `T - 1`; an
`fps` value of zero disables sleeping, which is useful in headless tests.

Playback intentionally operates on stored knots. It does not interpolate an
arbitrary cursor between configurations.

## `RendererBackend`

Render modes depend on a small protocol rather than on viser directly. The
protocol covers geometry creation, removal, pose and visibility updates,
colour and uniform-scale updates, and optional GUI controls.

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
    def set_transform(self, name, pose) -> None: ...
    def set_visible(self, name, visible) -> None: ...
    def set_color(self, name, rgba) -> None: ...
    def set_scale(self, name, scale) -> None: ...
```

`ViserBackend` maps these operations to `server.scene` nodes. Quaternion
conversion is centralized in `viewer/helpers.py`: BetterRobot uses scalar-last
`[qx, qy, qz, qw]`, while viser uses scalar-first `[w, x, y, z]`.

`MockBackend` records every call and keeps the latest transforms, visibility,
colours, and scales in memory. This supports full scene, styling, and playback
tests without a browser or rendering process.

## Dependency hygiene

| Import | Allowed in | Reason |
|---|---|---|
| `viser` | `viewer/renderers/viser_backend.py` only | Interactive backend |
| `trimesh` | `viewer/render_modes/urdf_mesh.py` only | Visual mesh loading |
| `torch` | Anywhere in `viewer/*` | Canonical tensor type |

Render modes and overlays do not import viser. `import better_robot.viewer`
therefore remains safe on machines where the optional interactive dependency
is absent.

## Public API

The main viewer surface is exported from `better_robot.viewer`:

```python
from better_robot.viewer import (
    PrimitiveHandle,
    RenderContext,
    RendererBackend,
    RenderMode,
    Scene,
    SkeletonMode,
    TrajectoryPlayer,
    URDFMeshMode,
    ViserBackend,
    Visualizer,
)
```

The live overlays are exported from `better_robot.viewer.overlays`, including
`ForceVectorsOverlay`. Viewer symbols are not added to
`better_robot.__all__`; the root robotics API remains compact.

## Sharp edges

- `Visualizer.show()` blocks by default. Pass `block=False` when the caller
  owns the surrounding event loop.
- `TrajectoryPlayer.play()` is also blocking.
- `URDFMeshMode` requires parser IR. Programmatically constructed models use
  `SkeletonMode`.
- A joint primitive handle exists only for an articulated joint rendered by
  the skeleton layer; requesting another joint raises `ValueError`.
- `PrimitiveHandle.set_color` accepts four components in `[0, 1]`, and
  `set_scale` requires a positive value.

## Where to look next

- {doc}`tasks` for IK and trajectory task results.
- {doc}`/conventions/extension` for the custom render-mode recipe.
- {doc}`/reference/api/better_robot/better_robot.viewer` for generated API
  documentation.
