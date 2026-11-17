"""``RobotVisual``: one robot in the viewer, as its meshes or as a skeleton, with a collision overlay and frames.

```python
import better_robot as br
from better_robot.viewer import RobotVisual, Viewer

model = br.load("panda.urdf")
viewer = Viewer()
robot = viewer.add(RobotVisual(model, style="maximal"))
robot.update(model.q_neutral)  # a new configuration, or a clip (T, nq) of as many steps
viewer.serve()
```

The visual runs forward kinematics once per ``q`` it is given; its parts receive numpy arrays and never touch the
model again. The skeleton's layout is ``src/better_robot/viewer/skeleton.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, final, override

import numpy as np
import torch

from ..collision import RobotSpheres
from ..exceptions import ArgumentError, ModelError, ShapeError
from ..kinematics import forward_kinematics
from ..lie import se3, so3
from ..model.geometry import Box, Cylinder, Geometry, Mesh, Sphere
from ..model.joints import JointFixed
from ._extras import needs_extra
from .layer import Bounds, Layer
from .layers.frames import FramesLayer
from .layers.mesh import MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.skinned import SkinnedMeshLayer
from .sidebar import Sidebar
from .skeleton import MAXIMAL, MINIMAL, Skeleton, SkeletonRecipe, build_skeleton, unit_ball, unit_box, unit_cylinder
from .theme import LIGHT, RGB, Theme, linear_uint8

if TYPE_CHECKING:
    from ..model.model import Model


# ------------------------------------------------------------------------------------------------ the robot visual

Style = Literal["meshes", "minimal", "maximal"]

_STYLES = ("meshes", "minimal", "maximal")
_TRIAD = (2.0, 0.05)  # length and thickness of a frame's axes, over the skeleton's glyph radius


@final
@dataclass(frozen=True)
class Skin:
    """A skinned surface over a model's joints: ``vertices (V, 3)`` in the world at ``bind``, ``faces (F, 3)`` and
    ``weights (V, njoints)``; ``bind (nq,)`` is the configuration of the rest surface, ``None`` for ``q_neutral``."""

    vertices: np.ndarray
    faces: np.ndarray
    _: KW_ONLY
    weights: np.ndarray
    bind: np.ndarray | None = None

    @override
    def __repr__(self) -> str:
        return f"Skin(V={len(self.vertices)}, F={len(self.faces)}, njoints={self.weights.shape[-1]})"


@final
class _Blocks(Layer):
    """The nodes of one skeleton: a batched node per unit mesh in use and a line node for limit arcs."""

    def __init__(self, name: str, skeleton: Skeleton, poses: np.ndarray, hue: RGB, theme: Theme) -> None:
        super().__init__(name, poses.shape[0])
        self.skeleton, self.poses, self.hue, self.theme = skeleton, poses, hue, theme
        self._nodes: dict[str, Any] = {}
        self._lines: Any = None

    @override
    def build(self, scene: Any, root: str) -> None:
        meshes = {"cylinders": unit_cylinder(), "balls": unit_ball(), "boxes": unit_box(taper=self.skeleton.taper)}
        colors = self.skeleton.colors(self.hue, self.theme.shade)
        for mesh, (world, scale) in self.skeleton.place(self.poses[0]).items():
            if len(world):  # viser's batched nodes do not take zero instances
                vertices, faces = meshes[mesh]
                self._nodes[mesh] = self._track(
                    scene.add_batched_meshes_simple(
                        f"{root}/{mesh}", vertices, faces, batched_wxyzs=world[:, [6, 3, 4, 5]],
                        batched_positions=world[:, :3], batched_scales=scale.astype(np.float32),
                        batched_colors=linear_uint8(colors[mesh]), lod="off", flat_shading=mesh == "boxes",
                        receive_shadow=self.skeleton.taper == 1.0,  # a tapered block's own shadow streaks its faces
                    )
                )  # fmt: skip
        if len(self.skeleton.line_body):
            points = self.skeleton.lines(self.poses[0])
            color = linear_uint8(self.theme.shade(self.hue, "face"))
            self._lines = self._track(scene.add_line_segments(f"{root}/limits", points, color, line_width=2.5))

    @override
    def set_step(self, step: int) -> None:
        for mesh, (world, scale) in self.skeleton.place(self.poses[step]).items():
            if mesh in self._nodes:
                self._nodes[mesh].batched_positions = world[:, :3]
                self._nodes[mesh].batched_wxyzs = world[:, [6, 3, 4, 5]]
                if mesh == "cylinders" and (self.skeleton.cylinders.anchors[:, 0] >= 0).any():  # sticks stretch
                    self._nodes[mesh].batched_scales = scale.astype(np.float32)
        if self._lines is not None:
            self._lines.points = self.skeleton.lines(self.poses[step])

    def recolor(self, hue: RGB, theme: Theme) -> None:
        """Draw the blocks in ``theme``'s shades of ``hue``."""
        self.hue, self.theme = hue, theme
        colors = self.skeleton.colors(hue, theme.shade)
        for mesh, node in self._nodes.items():
            node.batched_colors = linear_uint8(colors[mesh])
        if self._lines is not None:
            self._lines.colors = linear_uint8(theme.shade(hue, "face"))


@final
class RobotVisual(Layer):
    """One robot: its ``model`` at the configurations ``q``, shown in one of three styles.

    ``"meshes"`` draws the visual shapes of ``geometry`` (mesh files are found under ``root``, the description file
    or its folder, and under each folder above it) or the ``skin``; ``"minimal"`` and ``"maximal"`` draw the
    skeleton of the recipes ``minimal`` and ``maximal``. ``style=None`` is ``"meshes"`` when there is something to
    show as meshes, else ``"maximal"``. ``collision`` overlays the ``spheres``, or without them the collision
    shapes of ``geometry``; ``frames`` draws a triad per moving joint (``True``) or per frame given, by id or name.
    Every part is built the first time it is shown; ``set_style``, ``set_collision`` and ``set_frames`` switch them
    while the viewer runs, and the sidebar gets the same three controls, shared by every robot in the viewer: a
    switch from code moves the control and leaves the other robots as they are.

    ``q`` is ``(nq,)`` or a clip ``(T, nq)``; ``None`` is ``model.q_neutral``. ``hue=None`` takes the theme's next
    hue, one per robot. ``name`` defaults to the model's name.
    """

    def __init__(
        self,
        model: Model,
        q: torch.Tensor | np.ndarray | None = None,
        *,
        name: str | None = None,
        geometry: Geometry | None = None,
        root: str | Path | None = None,
        spheres: RobotSpheres | None = None,
        skin: Skin | None = None,
        style: Style | None = None,
        minimal: SkeletonRecipe = MINIMAL,
        maximal: SkeletonRecipe = MAXIMAL,
        hue: RGB | None = None,
        collision: bool = False,
        frames: bool | Sequence[int | str] = False,
        tags: Sequence[str] = (),
    ) -> None:
        self.model, self.geometry, self.spheres, self.skin = model, geometry, spheres, skin
        self.q, self._poses = self._solve(q)
        super().__init__(name or model.name, self._poses.shape[0], tags=tags)
        self._root_folder = None if root is None else Path(root)
        self._recipes = {"minimal": minimal, "maximal": maximal}
        self._skeletons: dict[str, Skeleton] = {}
        visual = geometry is not None and any(not attached.collision for attached in geometry.shapes)
        self.styles: tuple[Style, ...] = _STYLES if visual or skin is not None else _STYLES[1:]
        self.style: Style = style or ("meshes" if "meshes" in self.styles else "maximal")
        if self.style not in self.styles:
            raise ArgumentError(
                f"style must be one of {list(self.styles)} for robot {self.name!r}, got {self.style!r}; 'meshes' "
                "needs a geometry with visual shapes or a skin"
            )
        self._has_collision = spheres is not None or (
            geometry is not None and any(attached.collision for attached in geometry.shapes)
        )
        self._collision = False
        self._frames: bool | tuple[int, ...] = False  # what is shown now: every moving joint, or frame ids
        self._triads: bool | tuple[int, ...] = False  # the last set shown: what the frames part draws once built
        self._hue, self._index = hue, None
        self._theme: Theme = LIGHT  # until the viewer applies its own
        self._scene: Any = None
        self._path = ""
        self._step = 0
        self._parts: dict[str, tuple[Any, list[Layer]]] = {}  # part -> (its frame node, its layers)
        if collision:
            self.set_collision(visible=True)
        self.set_frames(frames)

    # ------------------------------------------------------------------ the configurations

    def _solve(self, q: torch.Tensor | np.ndarray | None) -> tuple[torch.Tensor, np.ndarray]:
        """``q`` as ``(T, nq)`` and the joint poses ``(T, njoints, 7)`` of one forward-kinematics pass."""
        model = self.model
        q = (
            model.q_neutral
            if q is None
            else torch.as_tensor(q, dtype=model.q_neutral.dtype, device=model.q_neutral.device)
        )
        if q.ndim not in (1, 2) or q.shape[-1] != model.nq:
            raise ShapeError(f"q must be (nq,) or (T, nq) with nq = {model.nq}, got {tuple(q.shape)}")
        q = q[None] if q.ndim == 1 else q
        with torch.no_grad():
            poses = forward_kinematics(model, q).joint_poses
        return q, np.ascontiguousarray(poses.cpu().numpy(), dtype=np.float32)

    def update(self, q: torch.Tensor | np.ndarray) -> None:
        """Show the configurations ``q``, ``(nq,)`` or ``(T, nq)`` with the ``T`` the visual was made with: one
        forward-kinematics pass, then the shown parts are redrawn. A clip of another length is a new visual
        (``Viewer.replace``)."""
        q, poses = self._solve(q)
        if poses.shape[0] != self.nsteps:
            raise ShapeError(f"q must keep the {self.nsteps} steps of robot {self.name!r}, got {poses.shape[0]}")
        with self._lock():
            self.q = q
            self._poses[...] = poses  # the parts read this array
            for part in ("collision", "frames"):
                if part in self._parts:
                    self._derive(part, self._parts[part][1][0])
            if self.viewer is not None:
                self.redraw()
            elif self._scene is not None:
                self.set_step(self._step)

    def _lock(self) -> Any:
        return self.viewer.lock if self.viewer is not None else nullcontext()

    # ------------------------------------------------------------------ the parts

    def _shown(self) -> tuple[str, ...]:
        return (self.style, *(("collision",) if self._collision else ()), *(("frames",) if self._frames else ()))

    def _skeleton(self, part: str) -> Skeleton:
        if part not in self._skeletons:
            self._skeletons[part] = build_skeleton(
                self.model,
                self._recipes[part],
                spheres=self.spheres,
                geometry=self.geometry,
                mesh_vertices=self._collision_vertices,
            )
        return self._skeletons[part]

    def _collision_vertices(self, mesh: Mesh) -> np.ndarray | None:
        """The vertices ``(V, 3)`` of a collision mesh file, which size the skeleton's bodies; ``None`` when the
        file is not found, and the body is sized from its inertia."""
        with needs_extra("meshes"):
            import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

        try:
            path = _mesh_file(-1, mesh.uri, self._root_folder)
        except ModelError:
            return None
        return np.asarray(trimesh.load(path, force="mesh").vertices)

    def _color(self) -> RGB:
        return self._hue if self._hue is not None else self._theme.hue(self._index or 0)

    def _make(self, part: str) -> list[Layer]:
        """The layers of ``part``, not yet built."""
        if part in ("minimal", "maximal"):
            return [_Blocks(part, self._skeleton(part), self._poses, self._color(), self._theme)]
        if part == "meshes" and self.skin is not None:
            bind = self._solve(self.skin.bind)[1][0]
            return [
                SkinnedMeshLayer(
                    part, self.skin.vertices, faces=self.skin.faces, weights=self.skin.weights, bind_poses=bind,
                    bone_poses=self._poses, color=self._theme.shade(self._color(), "body"),
                )
            ]  # fmt: skip
        if part == "frames":
            radius = self._skeleton("maximal").radius  # a triad reaches one glyph radius out of its joint's drum
            poses = np.zeros((self.nsteps, self._frame_count(), 7), np.float32)
            layer = FramesLayer(part, poses, axes_length=_TRIAD[0] * radius, axes_radius=_TRIAD[1] * radius)
            return [self._derive(part, layer)]
        if part == "collision" and self.spheres is not None:
            radii = self.spheres.radii.detach().cpu().numpy()
            centers = np.zeros((self.nsteps, len(radii), 3), np.float32)
            theme = self._theme
            layer = SpheresLayer(part, centers, radii=radii, colors=theme.collision, opacity=theme.collision_opacity)
            return [self._derive(part, layer)]
        shapes = self.geometry.shapes if self.geometry is not None else ()
        collision = part == "collision"
        color = self._theme.collision if collision else self._theme.shade(self._color(), "body")
        specs = [
            _shape_spec(attached.body, attached.shape, self._root_folder, color, own_colors=not collision)
            for attached in shapes
            if attached.collision == collision
        ]
        opacity = self._theme.collision_opacity if collision else 1.0
        return [RigidMeshLayer(part, specs, body_poses=self._poses, opacity=opacity)]

    def _frame_count(self) -> int:
        return len(self._moving()) if self._triads is True else len(self._triads or ())

    def _moving(self) -> list[int]:
        return [j for j in range(1, self.model.njoints) if not isinstance(self.model.joints[j], JointFixed)]

    def _derive(self, part: str, layer: Any) -> Any:
        """Fill the arrays of the ``frames`` or sphere ``collision`` layer from the joint poses."""
        model, poses = self.model, torch.from_numpy(self._poses)
        if part == "frames":  # the set the node was built for, also while it is hidden
            if self._triads is True:
                layer.poses[...] = self._poses[:, self._moving()]
            else:
                ids = list(self._triads or ())
                parents = [model.frame_parent_joint_ids[f] for f in ids]
                layer.poses[...] = se3.compose(poses[:, parents], model.frame_placements[ids].to(poses)).numpy()
        elif self.spheres is not None:
            ids = self.spheres.frame_ids.tolist()
            parents = [model.frame_parent_joint_ids[f] for f in ids]
            frames = se3.compose(poses[:, parents], model.frame_placements[ids].to(poses))
            layer.centers[...] = se3.act(frames, self.spheres.offsets.to(poses)).numpy()
        return layer

    def _drop(self, part: str) -> None:
        """Remove the nodes of ``part``; it is built again the next time it is shown."""
        if part in self._parts:
            frame, layers = self._parts.pop(part)
            for layer in layers:
                layer.dispose()
            frame.remove()

    def _apply(self) -> None:
        """Build what is shown for the first time, show and hide the parts' frames, push the step to the new ones."""
        if self._scene is None:
            return
        with self._lock():
            for part in self._shown():
                if part not in self._parts:
                    frame = self._scene.add_frame(f"{self._path}/{part}", show_axes=False)
                    layers = self._make(part)
                    for layer in layers:
                        layer.build(self._scene, f"{self._path}/{part}")
                    self._parts[part] = (frame, layers)
                    frame.visible = False  # so that the loop below pushes its step
            for part, (frame, layers) in self._parts.items():
                show = self._visible and part in self._shown()
                if show and not frame.visible:
                    for layer in layers:
                        layer.set_step(self._step)
                if frame.visible != show:
                    frame.visible = show

    def set_style(self, style: Style) -> None:
        """Show the robot as ``style``; the first use of a style builds its nodes, later ones only show them."""
        if style not in self.styles:
            raise ArgumentError(f"style must be one of {list(self.styles)} for robot {self.name!r}, got {style!r}")
        self.style = style
        self._apply()
        self._show("Style", style)

    def set_collision(self, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        """Show or hide the collision overlay: the spheres, or the collision shapes of the geometry."""
        if visible and not self._has_collision:
            raise ArgumentError(f"robot {self.name!r} has no spheres and no collision shapes to overlay")
        self._collision = visible
        self._apply()
        self._show("Collision", visible)

    def set_frames(self, frames: bool | Sequence[int | str]) -> None:  # noqa: FBT001 - the switch is the main operand
        """Draw a triad per moving joint (``True``), per frame of ``frames`` (ids or names), or none (``False``)."""
        chosen = frames if isinstance(frames, bool) else tuple(self.model.frame_id(frame) for frame in frames)
        if chosen and chosen != self._triads:  # another set of frames is another node
            self._drop("frames")
            self._triads = chosen
        self._frames = chosen
        self._apply()
        self._show("Frames", bool(chosen))

    def _show(self, label: str, value: Any) -> None:
        """Move the Robot section's control, shared by every robot, to what code set; no robot's callback runs."""
        if self.viewer is not None:
            self.viewer.sidebar.set_value("Robot", label, value=value)

    # ------------------------------------------------------------------ the Layer contract

    @override
    def build(self, scene: Any, root: str) -> None:
        self._scene, self._path, self._parts = scene, root, {}
        self._apply()

    @override
    def set_step(self, step: int) -> None:
        self._step = step
        for part in self._shown():
            for layer in self._parts[part][1] if part in self._parts else ():
                layer.set_step(step)

    @override
    def build_gui(self, sidebar: Sidebar) -> None:
        def style(value: str) -> None:  # the dropdown is shared: another robot may have a style this one lacks
            if value in self.styles:
                self.set_style(value)  # type: ignore[arg-type]

        sidebar.dropdown("Robot", "Style", options=self.styles, value=self.style, on_change=style)
        if self._has_collision:
            sidebar.checkbox("Robot", "Collision", value=self._collision, on_change=self.set_collision)
        sidebar.checkbox(
            "Robot",
            "Frames",
            value=bool(self._frames),
            on_change=lambda on: self.set_frames(on and (self._triads or True)),
        )

    @override
    def apply_theme(self, theme: Theme) -> None:
        if self._index is None:  # the robot's place among the robots of its viewer picks its hue
            others = self.viewer.layers if self.viewer is not None else ()
            self._index = sum(isinstance(layer, RobotVisual) and layer is not self for layer in others)
        body, overlay = self._theme.shade(self._color(), "body"), (self._theme.collision, self._theme.collision_opacity)
        self._theme = theme
        for part in ("minimal", "maximal"):
            if part in self._parts:
                self._parts[part][1][0].recolor(self._color(), theme)  # type: ignore[attr-defined]
        stale = {
            "meshes": body != theme.shade(self._color(), "body"),
            "collision": overlay != (theme.collision, theme.collision_opacity),
        }
        for part, old in stale.items():
            if old:  # their colours are set when they are built
                self._drop(part)
        self._apply()

    @override
    def bounds(self, step: int) -> Bounds | None:
        """A box per joint (a point) and per block of the maximal skeleton, ``(lo (N, 3), hi (N, 3))``: the robot's
        extent in any style."""
        joints = self._poses[step, min(1, self.model.njoints - 1) :, :3]  # not the universe, unless it is all there is
        lows, highs = [joints], [joints]
        for mesh, (world, scale) in self._skeleton("maximal").place(self._poses[step]).items():
            half = scale * {"cylinders": (1.0, 1.0, 0.5), "balls": (1.0, 1.0, 1.0)}.get(mesh, (0.5, 0.5, 0.5))
            rotations, eye = torch.from_numpy(world[:, None, 3:]), torch.from_numpy(np.eye(3, dtype=world.dtype))
            axes = np.abs(so3.act(rotations, eye).numpy())  # (N, 3, 3): each block's axes
            reach = np.einsum("nk,nkc->nc", half, axes)  # the half extent of its world box
            lows.append(world[:, :3] - reach)
            highs.append(world[:, :3] + reach)
        return np.concatenate(lows), np.concatenate(highs)

    @override
    def set_visible(self, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        super().set_visible(visible)
        self._apply()

    @override
    def dispose(self) -> None:
        for part in list(self._parts):
            self._drop(part)
        self._scene = None
        super().dispose()


# ------------------------------------------------------------------------------------------------ mesh files


def _mesh_file(body: int, uri: str, root: Path | None) -> Path:
    """The file a mesh ``uri`` names: as written when absolute, else under ``root`` (a description file or its
    folder) or a folder above it; ``package://<package>/<path>`` is tried with and without its package folder."""
    path = Path(uri.removeprefix("package://"))
    if path.is_absolute():
        return path
    relative = [path, *([Path(*path.parts[1:])] if uri.startswith("package://") and len(path.parts) > 1 else [])]
    start = Path.cwd() if root is None else (root if root.is_dir() else root.parent).resolve()
    folders = [start, *start.parents]
    for folder in folders:
        for candidate in relative:
            if (folder / candidate).is_file():
                return folder / candidate
    raise ModelError(
        f"body {body}: mesh {uri!r} is not a file under {start} or a folder above it; pass root=, the description file"
    )


def _shape_spec(body: int, shape: Any, root: Path | None, color: RGB, *, own_colors: bool) -> MeshSpec:
    """One shape as a ``MeshSpec``: a mesh file loaded with trimesh, a primitive tessellated by it."""
    with needs_extra("meshes"):
        import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    source = None
    if isinstance(shape, Mesh):
        path = _mesh_file(body, shape.uri, root)
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as exc:  # noqa: BLE001 - trimesh raises many loader-specific errors
            raise ModelError(f"body {body}: mesh {shape.uri!r} does not load: {exc}") from exc
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            raise ModelError(f"body {body}: mesh {shape.uri!r} has no faces")
        mesh.apply_scale(shape.scale)
        visual = mesh.visual  # a texture or more than one vertex or face colour: the mesh keeps its own look
        kind = None if visual is None else visual.kind
        varied = kind == "texture" or (
            kind in ("vertex", "face") and len(np.unique(np.asarray(getattr(visual, f"{kind}_colors")), axis=0)) > 1
        )
        source = mesh if own_colors and varied else None
    elif isinstance(shape, Box):
        mesh = trimesh.creation.box(extents=shape.size)
    elif isinstance(shape, Cylinder):
        mesh = trimesh.creation.cylinder(radius=shape.radius, height=shape.length)
    elif isinstance(shape, Sphere):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=shape.radius)
    else:
        mesh = trimesh.creation.capsule(radius=shape.radius, height=shape.length)
    own = shape.color[:3] if own_colors and shape.color is not None else color
    return MeshSpec(
        np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32), body=body,
        origin=np.asarray(shape.placement, dtype=np.float32), color=own, trimesh=source,
    )  # fmt: skip
