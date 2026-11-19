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

import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, final, override

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
from .skeleton import (
    MAXIMAL,
    MEDIUM,
    MINIMAL,
    Skeleton,
    SkeletonRecipe,
    build_skeleton,
    unit_ball,
    unit_box,
    unit_cylinder,
)
from .theme import LIGHT, RGB, Theme, linear_uint8

if TYPE_CHECKING:
    from ..model.model import Model


# ------------------------------------------------------------------------------------------------ the robot visual

Style = Literal["meshes", "minimal", "medium", "maximal"]

_STYLES = ("meshes", "minimal", "medium", "maximal")
_SKELETONS = _STYLES[1:]
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
    """The nodes of one skeleton: a batched node per unit mesh in use and a line node for limit arcs.

    With ``values`` ``(T, njoints)`` a step also colours the joint glyphs, one write of colours to each node that
    holds one (``_paint``).
    """

    def __init__(
        self, name: str, skeleton: Skeleton, poses: np.ndarray, hue: RGB, theme: Theme, *, opacity: float,
        values: np.ndarray | None, value_range: tuple[float, float],
    ) -> None:  # fmt: skip
        super().__init__(name, poses.shape[0])
        self.skeleton, self.poses, self.hue, self.theme = skeleton, poses, hue, theme
        self.values, self.value_range = values, value_range
        self._store_opacity(opacity)
        self._nodes: dict[str, Any] = {}
        self._lines: Any = None
        self._step = 0
        self._colors: dict[str, np.ndarray] = {}  # per node, the colours of its blocks without values
        self._ends: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # per node, its glyphs (_shade)

    @override
    def build(self, scene: Any, root: str) -> None:
        meshes = {"cylinders": unit_cylinder(), "balls": unit_ball(), "boxes": unit_box(taper=self.skeleton.taper)}
        colors = self._shade()
        for mesh, (world, scale) in self.skeleton.place(self.poses[0]).items():
            if len(world):  # viser's batched nodes do not take zero instances
                vertices, faces = meshes[mesh]
                self._nodes[mesh] = self._track(
                    scene.add_batched_meshes_simple(
                        f"{root}/{mesh}", vertices, faces, batched_wxyzs=world[:, [6, 3, 4, 5]],
                        batched_positions=world[:, :3], batched_scales=scale.astype(np.float32),
                        batched_colors=linear_uint8(colors[mesh]), lod="off", flat_shading=mesh == "boxes",
                        **self._surface(),
                    )
                )  # fmt: skip
        if len(self.skeleton.line_body):
            points = self.skeleton.lines(self.poses[0])
            color = linear_uint8(self.theme.shade(self.hue, "face"))
            self._lines = self._track(
                scene.add_line_segments(f"{root}/limits", points, color, thickness=2.5, thickness_units="screen")
            )

    @override
    def set_step(self, step: int) -> None:
        self._step = step
        for mesh, (world, scale) in self.skeleton.place(self.poses[step]).items():
            if mesh in self._nodes:
                self._nodes[mesh].batched_positions = world[:, :3]
                self._nodes[mesh].batched_wxyzs = world[:, [6, 3, 4, 5]]
                if mesh == "cylinders" and (self.skeleton.cylinders.anchors[:, 0] >= 0).any():  # sticks stretch
                    self._nodes[mesh].batched_scales = scale.astype(np.float32)
        if self._lines is not None:
            self._lines.points = self.skeleton.lines(self.poses[step])
        if self.values is not None:
            self._paint(self.values[step])

    def _shade(self) -> dict[str, np.ndarray]:
        """The colours ``(N, 3)`` of every block in the shades of the hue, kept with, per unit mesh, its glyphs: their
        rows, the joints each stands for and the colours of the two ends of the theme's ramp in each one's shade (a
        drum face darker, as the recipe's colour of a glyph is)."""
        self._colors, self._ends = self.skeleton.colors(self.hue, self.theme), {}
        ramp = np.array(self.theme.ramp, dtype=np.float32)  # (2, 3)
        dark = np.array([self.theme.shade(end, "face") for end in self.theme.ramp], dtype=np.float32)
        for mesh in self._colors:
            rows, joints, faces = self.skeleton.glyphs(mesh, self.poses.shape[1])
            if len(rows):
                self._ends[mesh] = (rows, joints, np.where(faces[:, None, None], dark, ramp))  # (n, 2, 3)
        return self._colors

    def _paint(self, values: np.ndarray) -> None:
        """Colour each glyph by its joint's value in ``values (njoints,)``, a ball of the stick figure by the value of
        the largest magnitude among its joints; a glyph without a value (NaN) keeps its colour."""
        low, high = self.value_range
        for mesh, (rows, joints, ends) in self._ends.items():
            picked = np.where(joints, values, np.nan)  # (n, njoints)
            picked = picked[np.arange(len(rows)), np.nan_to_num(np.abs(picked), nan=-1.0).argmax(axis=1)]
            share = np.clip((picked - low) / (high - low), 0.0, 1.0)[:, None]
            valued = ~np.isnan(picked)
            colors = self._colors[mesh].copy()
            colors[rows[valued]] = ends[valued, 0] + share[valued] * (ends[valued, 1] - ends[valued, 0])
            self._nodes[mesh].batched_colors = linear_uint8(colors)

    @override
    def _surface(self) -> dict[str, Any]:
        surface = super()._surface()
        surface["receive_shadow"] &= self.skeleton.taper == 1.0  # a tapered block's own shadow streaks its faces
        return surface

    @override
    def set_opacity(self, opacity: float) -> None:
        self._fade(opacity, list(self._nodes.values()))  # the limit arcs are lines: they stay as they are

    def recolor(self, hue: RGB, theme: Theme) -> None:
        """Draw the blocks in ``theme``'s shades of ``hue``, the glyphs with values in its ramp."""
        self.hue, self.theme = hue, theme
        colors = self._shade()
        for mesh, node in self._nodes.items():
            node.batched_colors = linear_uint8(colors[mesh])
        if self._lines is not None:
            self._lines.colors = linear_uint8(theme.shade(hue, "face"))
        if self.values is not None:
            self._paint(self.values[self._step])


@final
class RobotVisual(Layer):
    """One robot: its ``model`` at the configurations ``q``, shown in one of four styles.

    ``"meshes"`` draws the visual shapes of ``geometry`` (mesh files are found under ``root``, the description file
    or its folder, and under each folder above it) or the ``skin``; ``"minimal"``, ``"medium"`` and ``"maximal"``
    draw the skeleton of the recipes ``minimal``, ``medium`` and ``maximal``. ``style=None`` is ``"meshes"`` when
    there is something to show as meshes, else ``"maximal"``. ``collision`` overlays the ``spheres``, or without
    them the collision shapes of ``geometry``, as a pale glass shell over any style; ``frames`` draws a triad per
    moving joint (``True``) or per frame given, by id or name. Every part is built the first time it is shown;
    ``set_style``, ``set_collision`` and ``set_frames`` switch them while the viewer runs, and the sidebar gets the
    same three controls, shared by every robot in the viewer: a switch from code moves the control and leaves the
    other robots as they are.

    ``q`` is ``(nq,)`` or a clip ``(T, nq)``; ``None`` is ``model.q_neutral``. ``hue=None`` takes the first of the
    theme's hues that no other robot of the viewer is drawn in (``hue``), and a ``hue`` that is not three numbers
    raises ``ShapeError``. ``name`` defaults to the model's name. ``opacity`` in ``(0, 1]`` draws every style see-through,
    a ghost of the robot beside or over another (``set_opacity`` changes it); below 1 it casts and catches no shadow,
    a textured mesh file stays opaque, and the overlay and the triads keep their own look. ``joint_values`` colours
    the joint glyphs of the skeleton styles at each step through the theme's ``ramp`` over ``value_range``, such as a
    joint's load (``set_joint_values``).
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
        medium: SkeletonRecipe = MEDIUM,
        maximal: SkeletonRecipe = MAXIMAL,
        hue: RGB | None = None,
        opacity: float = 1.0,
        joint_values: torch.Tensor | np.ndarray | None = None,
        value_range: tuple[float, float] = (0.0, 1.0),
        collision: bool = False,
        frames: bool | Sequence[int | str] = False,
        tags: Sequence[str] = (),
    ) -> None:
        if hue is not None and np.shape(hue) != (3,):
            raise ShapeError(f"hue must be an RGB triple, got {hue!r}")
        self.model, self.geometry, self.spheres, self.skin = model, geometry, spheres, skin
        self.q, self._poses = self._solve(q)
        super().__init__(name or model.name, self._poses.shape[0], tags=tags)
        self._store_opacity(opacity)
        self._root_folder = None if root is None else Path(root)
        self._recipes = {"minimal": minimal, "medium": medium, "maximal": maximal}
        self._skeletons: dict[str, Skeleton] = {}
        self._fit: _Fit | None = None  # what bounds needs that no step changes, the first time it is asked
        self._meshes: dict[str, Any] = {}  # mesh file name -> its trimesh, read once: a theme or style switch reuses it
        visual = geometry is not None and any(not attached.collision for attached in geometry.shapes)
        self.styles: tuple[Style, ...] = _STYLES if visual or skin is not None else _SKELETONS
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
        self.joint_values: np.ndarray | None = None
        self.value_range = (0.0, 1.0)
        self.set_joint_values(joint_values, value_range=value_range)
        if collision:
            self.set_collision(visible=True)
        self.set_frames(frames)

    # ------------------------------------------------------------------ the configurations

    def _solve(self, q: torch.Tensor | np.ndarray | None) -> tuple[torch.Tensor, np.ndarray]:
        """``q`` as a detached copy ``(T, nq)`` and the joint poses ``(T, njoints, 7)`` of one forward-kinematics
        pass."""
        model = self.model
        q = (
            model.q_neutral
            if q is None
            else torch.as_tensor(q, dtype=model.q_neutral.dtype, device=model.q_neutral.device)
        )
        if q.ndim not in (1, 2) or q.shape[-1] != model.nq:
            raise ShapeError(f"q must be (nq,) or (T, nq) with nq = {model.nq}, got {tuple(q.shape)}")
        q = (q[None] if q.ndim == 1 else q).detach().clone()  # neither the caller's tensor nor its graph
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

    def set_joint_values(
        self, values: torch.Tensor | np.ndarray | None, *, value_range: tuple[float, float] | None = None
    ) -> None:
        """Colour the joint glyphs of the skeleton styles by ``values (T, njoints)`` or ``(njoints,)``, one per model
        joint and step, such as each joint's load ``|tau| / effort limit``: ``T`` is the visual's, and the joints run in
        model order with joint 0, the universe, as ``Skin`` weights do. ``value_range`` ``(low, high)`` maps a value
        onto the theme's ``ramp``, from its first colour at ``low`` to its second at ``high`` and clamped beyond;
        ``None`` keeps the range in use. A drum takes the colour and its faces a darker shade of it, a ball of the
        minimal stick figure the value of the largest magnitude among the joints it stands for; a NaN value, the
        universe, a free base and the joints without a glyph keep the robot's colours, and the ``"meshes"`` style draws
        no joint, so it shows no value. ``values=None`` takes the values away. A step costs one more write, of the
        colours, to each skeleton node that holds a glyph. ``(T, njoints)`` float32 values are kept without a copy
        (``joint_values``): an edit in place and ``redraw()`` show it.

        Values of another shape raise ``ShapeError``; a ``value_range`` that is not two finite numbers, low below
        high, raises ``ArgumentError``.
        """
        if value_range is not None:
            if np.shape(value_range) != (2,):
                raise ShapeError(f"value_range must be two numbers (low, high), got {value_range!r}")
            low, high = (float(end) for end in value_range)
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ArgumentError(f"value_range must be two finite numbers, low below high, got {value_range!r}")
            value_range = (low, high)
        if values is not None:
            values = np.asarray(values.detach().cpu() if isinstance(values, torch.Tensor) else values, np.float32)
            values = np.ascontiguousarray(values[None] if values.ndim == 1 else values)
            if values.shape != (self.nsteps, self.model.njoints):
                raise ShapeError(
                    f"joint_values must be (njoints,) or (T, njoints) = ({self.nsteps}, {self.model.njoints}) for robot "
                    f"{self.name!r}, one value per model joint (the universe included) and step, got {values.shape}"
                )
        with self._lock():
            self.joint_values = values
            self.value_range = self.value_range if value_range is None else value_range
            for part in _SKELETONS:
                if part in self._parts:
                    blocks = self._parts[part][1][0]
                    blocks.values, blocks.value_range = values, self.value_range  # type: ignore[attr-defined]
                    if values is None:
                        blocks.recolor(self.hue, self._theme)  # type: ignore[attr-defined]
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
                mesh_vertices=self._vertices,
            )
        return self._skeletons[part]

    def _vertices(self, body: int, mesh: Mesh) -> np.ndarray | None:
        """The vertices ``(V, 3)`` of the collision mesh file of ``mesh`` on ``body``, which a skeleton sizes the body
        by. A file that ``_mesh`` cannot read gives ``None`` and a warning, and the skeleton sizes the body without
        it: a skeleton does not need the file, while the collision overlay, which draws it, raises."""
        try:
            return np.asarray(self._mesh(body, mesh).vertices)
        except ModelError as error:
            warnings.warn(  # Python's default filter shows it once, however many skeletons and steps read the file
                f"{error}; the skeleton sizes body {body} without it, from its other shapes, spheres or inertia",
                stacklevel=2,
            )
            return None

    def _mesh(self, body: int, mesh: Mesh) -> Any:
        """The ``trimesh.Trimesh`` of the mesh file of ``mesh`` on ``body``, before its scale; read once per visual.
        ``ModelError`` names the body and the file when it is not found, does not load or has no faces."""
        if mesh.uri not in self._meshes:
            self._meshes[mesh.uri] = _read_mesh(body, mesh.uri, self._root_folder)
        return self._meshes[mesh.uri]

    @property
    def hue(self) -> RGB:
        """The colour the robot is drawn in: the ``hue`` it was given, else its theme's hue for it, the first one no
        other robot of its viewer is drawn in when it was added."""
        return self._hue if self._hue is not None else self._theme.hue(self._index or 0)

    def _make(self, part: str) -> list[Layer]:
        """The layers of ``part``, not yet built."""
        if part in _SKELETONS:
            return [
                _Blocks(
                    part, self._skeleton(part), self._poses, self.hue, self._theme, opacity=self.opacity,
                    values=self.joint_values, value_range=self.value_range,
                )
            ]  # fmt: skip
        if part == "meshes" and self.skin is not None:
            bind = self._solve(self.skin.bind)[1][0]
            return [
                SkinnedMeshLayer(
                    part, self.skin.vertices, faces=self.skin.faces, weights=self.skin.weights, bind_poses=bind,
                    bone_poses=self._poses, color=self._theme.shade(self.hue, "body"), opacity=self.opacity,
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
        color = self._theme.collision if collision else self._theme.shade(self.hue, "body")
        specs = [
            _shape_spec(attached.body, attached.shape, self._mesh, color, own_colors=not collision)
            for attached in shapes
            if attached.collision == collision
        ]
        opacity = self._theme.collision_opacity if collision else self.opacity
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
            layer.centers[...] = _centers(model, self.spheres, poses)
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
        with self._lock():  # the playback thread must not push a step to the node being dropped
            if chosen and chosen != self._triads:  # another set of frames is another node
                self._drop("frames")
                self._triads = chosen
            self._frames = chosen
            self._apply()
        self._show("Frames", bool(chosen))

    @override
    def set_opacity(self, opacity: float) -> None:
        """Draw every style at ``opacity`` in ``(0, 1]`` (1 is opaque), live and in the styles built later; the
        collision overlay and the triads keep their own look."""
        with self._lock():
            self._store_opacity(opacity)
            for part, (_, layers) in self._parts.items():
                for layer in layers if part in _STYLES else ():
                    layer.set_opacity(opacity)

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
        if self._index is None:  # the first hue no other robot has: a robot that replaces one takes its hue back
            others = self.viewer.layers if self.viewer is not None else ()
            taken = {layer.hue for layer in others if isinstance(layer, RobotVisual) and layer is not self}
            free = [k for k in range(len(theme.hues)) if theme.hue(k) not in taken]
            self._index = free[0] if free else len(taken)
        body, overlay = self._theme.shade(self.hue, "body"), (self._theme.collision, self._theme.collision_opacity)
        self._theme = theme
        for part in _SKELETONS:
            if part in self._parts:
                self._parts[part][1][0].recolor(self.hue, theme)  # type: ignore[attr-defined]
        stale = {
            "meshes": body != theme.shade(self.hue, "body"),
            "collision": overlay != (theme.collision, theme.collision_opacity),
        }
        for part, old in stale.items():
            if old:  # their colours are set when they are built
                self._drop(part)
        self._apply()

    @override
    def bounds(self, step: int) -> Bounds | None:
        """A box per joint (a point), per collision sphere and per collision shape that is not a mesh (each exact),
        per block of the maximal skeleton on a rigid body without one of those, and per bone of the ``skin`` that
        carries vertices (``SkinnedMeshLayer.bounds``: no vertex is skinned), ``(lo (N, 3), hi (N, 3))``: the robot's
        extent in any style. A sole or a foot with such a shape gives the lowest point to stand a robot on.
        """
        model, poses = self.model, torch.from_numpy(self._poses[step])
        fit = self._fit = self._fit or self._fitted()
        joints = self._poses[step, min(1, model.njoints - 1) :, :3]  # not the universe, unless it is all there is
        lows, highs = [joints], [joints]
        if self.spheres is not None:
            centers = _centers(model, self.spheres, poses)
            lows.append(centers - fit.radii)
            highs.append(centers + fit.radii)
        if fit.bodies:
            world = se3.compose(poses[fit.bodies], fit.placements.to(poses.dtype)).numpy()
            axes = _axes(world)
            axis, sizes = axes[:, 2], fit.sizes.astype(world.dtype)  # each shape's own z
            reach = np.einsum("nk,nkc->nc", sizes[:, :3], axes) + sizes[:, 3:4] * axis
            reach += sizes[:, 4:5] * np.sqrt(np.clip(1.0 - axis**2, 0.0, None)) + sizes[:, 5:]
            lows.append(world[:, :3] - reach)
            highs.append(world[:, :3] + reach)
        for mesh, (world, scale) in fit.skeleton.place(self._poses[step]).items():
            keep = fit.keep[mesh]
            reach = np.einsum("nk,nkc->nc", (scale * _HALF_EXTENT[mesh])[keep], _axes(world[keep]))  # its world box
            lows.append(world[keep, :3] - reach)
            highs.append(world[keep, :3] + reach)
        if fit.skin is not None:
            low, high = cast("Bounds", fit.skin.bounds(step))  # a skin's layer always has its boxes per bone
            lows.append(low)
            highs.append(high)
        return np.concatenate(lows), np.concatenate(highs)

    def _fitted(self) -> _Fit:
        """What ``bounds`` reads that no step changes: the sphere radii, the shapes that are not meshes, which
        blocks of the maximal skeleton sit on a body that neither covers, and the skin's boxes per bone."""
        model = self.model
        anchor = list(range(model.njoints))  # the joint each body is welded to
        for j in range(1, model.njoints):
            anchor[j] = anchor[model.parent_joint_ids[j]] if isinstance(model.joints[j], JointFixed) else j
        covered, radii = set(), np.zeros((0, 1))
        if self.spheres is not None:
            radii = self.spheres.radii.detach().cpu().numpy()[:, None]
            covered |= {anchor[model.frame_parent_joint_ids[f]] for f in self.spheres.frame_ids.tolist()}
        shapes = [shape for shape in self.geometry.shapes if shape.collision] if self.geometry is not None else []
        shapes = [attached for attached in shapes if not isinstance(attached.shape, Mesh)]
        covered |= {anchor[attached.body] for attached in shapes}
        skeleton = self._skeleton("maximal")
        skin = None
        if self.skin is not None:  # never built: its boxes per bone follow the joint poses that update writes
            skin = SkinnedMeshLayer(
                "skin", self.skin.vertices, faces=self.skin.faces, weights=self.skin.weights,
                bind_poses=self._solve(self.skin.bind)[1][0], bone_poses=self._poses,
            )  # fmt: skip
        return _Fit(
            radii=radii,
            bodies=[attached.body for attached in shapes],
            placements=torch.tensor([attached.shape.placement for attached in shapes], dtype=torch.float64),
            sizes=np.array([_sizes(attached.shape) for attached in shapes], dtype=np.float64).reshape(-1, 6),
            skeleton=skeleton,
            keep={
                mesh: np.array([anchor[body] not in covered for body in getattr(skeleton, mesh).body], dtype=bool)
                for mesh in ("cylinders", "balls", "boxes")
            },
            skin=skin,
        )

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


def load_meshes(geometry: Geometry, *, root: str | Path | None = None, collision: bool = False) -> list[MeshSpec]:
    """The display shapes of ``geometry``, or its collision shapes with ``collision=True``, as arrays: one
    ``MeshSpec`` per shape, in description order, for code that needs a robot's surface without a viewer.

    Each spec holds ``vertices (V, 3)`` (float32, scaled) in the shape's own frame, ``faces (F, 3)``, the ``body``
    (a joint id: the body hangs on that joint's frame) and ``origin``, the shape's placement ``(7,)`` in that frame,
    so ``se3.act(origin, vertices)`` gives the vertices in the body's frame. Mesh files are found as
    ``RobotVisual(root=)`` finds them, under ``root`` (the description file or its folder, ``None`` for the working
    directory) and each folder above it; primitives are tessellated. A file that is not found, does not load or has
    no faces raises ``ModelError`` naming the body and the file. ``RigidMeshLayer(name, specs, body_poses=)`` draws
    the result.
    """
    folder = None if root is None else Path(root)
    read: dict[str, Any] = {}  # each file once, however many shapes name it

    def load(body: int, mesh: Mesh) -> Any:
        if mesh.uri not in read:
            read[mesh.uri] = _read_mesh(body, mesh.uri, folder)
        return read[mesh.uri]

    return [
        _shape_spec(attached.body, attached.shape, load, MeshSpec.color, own_colors=True)
        for attached in geometry.shapes
        if attached.collision == collision
    ]


def _read_mesh(body: int, uri: str, root: Path | None) -> Any:
    """The ``trimesh.Trimesh`` of the mesh file ``uri`` on ``body`` (``_mesh_file``), before its scale.
    ``ModelError`` names the body and the file when it is not found, does not load or has no faces."""
    with needs_extra("meshes"):
        import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    path = _mesh_file(body, uri, root)
    try:
        loaded = trimesh.load(path, force="mesh")
    except Exception as exc:  # noqa: BLE001 - trimesh raises many loader-specific errors
        raise ModelError(f"body {body}: mesh {uri!r} does not load: {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.shape[0] == 0:
        raise ModelError(f"body {body}: mesh {uri!r} has no faces")
    return loaded


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


def _centers(model: Model, spheres: RobotSpheres, poses: torch.Tensor) -> np.ndarray:
    """World centres ``(..., S, 3)`` of ``spheres`` at joint poses ``(..., njoints, 7)``."""
    ids = spheres.frame_ids.tolist()
    parents = [model.frame_parent_joint_ids[f] for f in ids]
    frames = se3.compose(poses[..., parents, :], model.frame_placements[ids].to(poses))
    return se3.act(frames, spheres.offsets.to(poses)).numpy()


def _axes(world: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)``: row ``k`` holds the absolute world components of axis ``k`` of each pose of ``world (N, 7)``."""
    rotations, eye = torch.from_numpy(world[:, None, 3:]), torch.from_numpy(np.eye(3, dtype=world.dtype))
    return np.abs(so3.act(rotations, eye).numpy())


#: Per unit mesh, its half extent along x, y and z over its scale.
_HALF_EXTENT = {"cylinders": (1.0, 1.0, 0.5), "balls": (1.0, 1.0, 1.0), "boxes": (0.5, 0.5, 0.5)}


@final
@dataclass(frozen=True)
class _Fit:
    """What ``RobotVisual.bounds`` reads that no step changes."""

    radii: np.ndarray  # (S, 1): the radius of each collision sphere
    bodies: list[int]  # per collision shape that is not a mesh, the body it is on
    placements: torch.Tensor  # (P, 7): its placement on that body
    sizes: np.ndarray  # (P, 6): what its world box needs (``_sizes``)
    skeleton: Skeleton  # the maximal skeleton
    keep: dict[str, np.ndarray]  # per unit mesh, (N,): whether a block is on a body that no sphere or shape covers
    skin: SkinnedMeshLayer | None  # the skin's layer, never built, for its boxes per bone


def _sizes(shape: Any) -> list[float]:
    """What the world box of a primitive needs: its half extents (3), the half length and radius of a round side
    along its z, and the radius of a ball swept round it."""
    if isinstance(shape, Box):
        return [*(np.asarray(shape.size, dtype=np.float64) / 2.0), 0.0, 0.0, 0.0]
    if isinstance(shape, Sphere):
        return [0.0, 0.0, 0.0, 0.0, 0.0, shape.radius]
    if isinstance(shape, Cylinder):
        return [0.0, 0.0, 0.0, shape.length / 2.0, shape.radius, 0.0]
    return [0.0, 0.0, 0.0, shape.length / 2.0, 0.0, shape.radius]  # a capsule: a ball swept along a segment


def _shape_spec(body: int, shape: Any, load: Callable[[int, Mesh], Any], color: RGB, *, own_colors: bool) -> MeshSpec:
    """One shape as a ``MeshSpec``: a mesh file's trimesh from ``load`` (``RobotVisual._mesh``), a primitive
    tessellated by trimesh."""
    with needs_extra("meshes"):
        import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    source = None
    if isinstance(shape, Mesh):
        mesh = load(body, shape).copy()  # the loaded one stays unscaled for the next shape that names its file
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
