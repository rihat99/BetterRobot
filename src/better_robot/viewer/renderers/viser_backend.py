"""``renderers/viser_backend.py`` — interactive viser renderer.

``viser`` is imported lazily here and ONLY here. No other viewer file
may import viser.

See ``docs/12_VIEWER.md §9.3``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..helpers import quat_xyzw_to_wxyz


class ViserBackend:
    """Interactive renderer backed by viser (browser-based 3D viewer)."""

    is_interactive: bool = True
    supports_gui: bool = True

    def __init__(self, *, port: int = 8080) -> None:
        try:
            import viser  # noqa: F401  (lazy — only imported here)
        except ImportError as exc:
            raise ImportError(
                "viser is required for the interactive backend. "
                "Install it with: pip install viser"
            ) from exc

        import viser as _viser

        self._server = _viser.ViserServer(port=port)
        self._nodes: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Geometry primitives
    # ------------------------------------------------------------------

    def add_mesh(
        self,
        name: str,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
        parent: Any = None,
    ) -> None:
        import numpy as np

        verts_np = vertices.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.uint32)
        colour = tuple(int(c * 255) for c in rgba[:3])
        node = self._server.scene.add_mesh_simple(
            name,
            vertices=verts_np,
            faces=faces_np,
            color=colour,
            opacity=float(rgba[3]),
        )
        self._nodes[name] = node

    def add_mesh_trimesh(self, name: str, mesh: Any, *, scale: Any = 1.0) -> None:
        """Add a trimesh object directly, preserving its materials/colors."""
        node = self._server.scene.add_mesh_trimesh(name, mesh, scale=scale)
        self._nodes[name] = node

    def add_sphere(
        self,
        name: str,
        *,
        radius: float,
        rgba: tuple[float, float, float, float],
        parent: Any = None,
    ) -> None:
        colour = tuple(int(c * 255) for c in rgba[:3])
        node = self._server.scene.add_icosphere(
            name,
            radius=radius,
            color=colour,
            opacity=float(rgba[3]),
        )
        self._nodes[name] = node

    def add_cylinder(
        self,
        name: str,
        *,
        radius: float,
        length: float,
        rgba: tuple[float, float, float, float],
        parent: Any = None,
    ) -> None:
        colour = tuple(int(c * 255) for c in rgba[:3])
        node = self._server.scene.add_mesh_simple(
            name,
            vertices=self._cylinder_vertices(radius, length),
            faces=self._cylinder_faces(),
            color=colour,
            opacity=float(rgba[3]),
        )
        self._nodes[name] = node

    def add_capsule(
        self,
        name: str,
        *,
        radius: float,
        length: float,
        rgba: tuple[float, float, float, float],
        parent: Any = None,
    ) -> None:
        # Fall back to cylinder shape for viser (capsule = cylinder + hemispheres)
        self.add_cylinder(name, radius=radius, length=length, rgba=rgba, parent=parent)

    def add_frame(self, name: str, *, axes_length: float = 0.1) -> None:
        node = self._server.scene.add_frame(name, axes_length=axes_length)
        self._nodes[name] = node

    def add_grid(self, name: str, **kwargs: Any) -> None:
        node = self._server.scene.add_grid(name, **kwargs)
        self._nodes[name] = node

    # ------------------------------------------------------------------
    # Scene graph updates
    # ------------------------------------------------------------------

    def remove(self, name: str) -> None:
        node = self._nodes.pop(name, None)
        if node is not None:
            node.remove()

    def set_transform(self, name: str, pose: torch.Tensor) -> None:
        """Set pose from a 7-vector ``[tx, ty, tz, qx, qy, qz, qw]`` (xyzw)."""
        node = self._nodes.get(name)
        if node is None:
            return
        p = pose.detach().cpu()
        xyz = p[:3].numpy()
        q_wxyz = quat_xyzw_to_wxyz(p[3:]).numpy()
        node.wxyz = q_wxyz
        node.position = xyz

    def set_visible(self, name: str, visible: bool) -> None:
        node = self._nodes.get(name)
        if node is not None:
            node.visible = visible

    def set_camera(self, camera: Any) -> None:
        raise NotImplementedError("see docs/12_VIEWER.md §9")

    def capture_frame(self) -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
        raise NotImplementedError("see docs/12_VIEWER.md §9.6")

    # ------------------------------------------------------------------
    # GUI controls
    # ------------------------------------------------------------------

    def add_gui_button(self, label: str, callback: Any) -> None:
        btn = self._server.gui.add_button(label)
        btn.on_click(callback)

    def add_gui_slider(
        self,
        label: str,
        *,
        min: float,
        max: float,
        step: float,
        value: float,
        callback: Any,
    ) -> None:
        slider = self._server.gui.add_slider(
            label, min=min, max=max, step=step, initial_value=value
        )
        slider.on_update(callback)

    def add_gui_checkbox(self, label: str, *, value: bool, callback: Any) -> None:
        cb = self._server.gui.add_checkbox(label, initial_value=value)
        cb.on_update(callback)

    def add_transform_control(
        self,
        name: str,
        pose: torch.Tensor,
        *,
        scale: float = 0.15,
        on_update: "Callable[[torch.Tensor], None] | None" = None,  # noqa: F821
    ) -> Any:
        """Add a draggable SE(3) gizmo and return the viser handle.

        The returned handle exposes live ``.position`` and ``.wxyz``
        attributes that viser keeps in sync with the browser as the
        user drags — callers can poll those attributes from a main-
        thread loop without any callback plumbing. ``on_update`` is
        still supported as an optional convenience; it fires with the
        new pose 7-vector ``[tx, ty, tz, qx, qy, qz, qw]`` on each
        drag update.

        ``scale`` controls the visual size of the gizmo in metres;
        viser's own default is ``1.0`` which is enormous next to a
        typical robot scene.
        """
        from ..helpers import quat_wxyz_to_xyzw

        p = pose.detach().cpu()
        xyz = p[:3].numpy()
        q_wxyz = quat_xyzw_to_wxyz(p[3:]).numpy()

        handle = self._server.scene.add_transform_controls(
            name,
            scale=scale,
            position=xyz,
            wxyz=q_wxyz,
        )
        self._nodes[name] = handle

        if on_update is not None:
            @handle.on_update
            def _cb(_evt: Any) -> None:
                import numpy as np
                pos_np = np.array(handle.position, dtype=np.float32)
                wxyz_np = np.array(handle.wxyz, dtype=np.float32)
                xyzw = quat_wxyz_to_xyzw(torch.from_numpy(wxyz_np))
                new_pose = torch.cat([torch.from_numpy(pos_np), xyzw])
                on_update(new_pose)

        return handle

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cylinder_vertices(radius: float, length: float) -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
        import numpy as np

        n = 16
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
        cos_a = np.cos(angles) * radius
        sin_a = np.sin(angles) * radius
        half_l = length / 2.0
        bottom = np.stack([cos_a, sin_a, np.full(n, -half_l, dtype=np.float32)], axis=1)
        top = np.stack([cos_a, sin_a, np.full(n, half_l, dtype=np.float32)], axis=1)
        center_bottom = np.array([[0.0, 0.0, -half_l]], dtype=np.float32)
        center_top = np.array([[0.0, 0.0, half_l]], dtype=np.float32)
        return np.concatenate([bottom, top, center_bottom, center_top], axis=0)

    @staticmethod
    def _cylinder_faces() -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
        import numpy as np

        n = 16
        faces: list[list[int]] = []
        cb_idx = 2 * n       # center bottom
        ct_idx = 2 * n + 1   # center top
        for i in range(n):
            j = (i + 1) % n
            # side quad as two triangles
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
            # bottom cap
            faces.append([cb_idx, j, i])
            # top cap
            faces.append([ct_idx, n + i, n + j])
        return np.array(faces, dtype=np.uint32)
