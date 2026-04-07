"""Viser-based robot visualizer — encapsulates all scene and GUI setup."""
from __future__ import annotations

import numpy as np
import torch
import yourdfpy
import viser
import viser.extras

from ..models.robot_model import RobotModel
from ..algorithms.geometry.robot_collision import RobotCollision
from ..algorithms.geometry.primitives import Capsule
from .helpers import wxyz_pos_to_se3, qxyzw_to_wxyz, build_joint_dict


def _capsule_mesh(radius: float, height: float):
    """Create a trimesh capsule aligned along Z, centered at origin.

    Returns (vertices (V,3), faces (F,3)) as numpy arrays.
    """
    import trimesh as tm
    mesh = tm.creation.capsule(radius=radius, height=height, count=[8, 16])
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _capsule_pose(pa: np.ndarray, pb: np.ndarray):
    """Compute (center, wxyz) to transform a Z-aligned capsule to match endpoints pa→pb.

    Returns:
        center: (3,) float32 ndarray
        wxyz: (4,) float32 ndarray — scalar-first quaternion for viser
    """
    center = (pa + pb) / 2.0
    vec = pb - pa
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return center.astype(np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis_world = vec / length

    dot = float(np.clip(np.dot(z, axis_world), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        # Already aligned with Z
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif dot < -1.0 + 1e-7:
        # Anti-aligned — rotate 180° around X
        wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    else:
        rot_axis = np.cross(z, axis_world)
        rot_axis /= np.linalg.norm(rot_axis)
        half_angle = np.arccos(dot) / 2.0
        w = float(np.cos(half_angle))
        xyz = rot_axis * np.sin(half_angle)
        wxyz = np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)

    return center.astype(np.float32), wxyz


class Visualizer:
    """Interactive robot visualizer backed by viser.

    Typical usage (fixed-base)::

        vis = Visualizer(urdf, model)
        vis.add_target("panda_hand")
        vis.add_restart_button()
        vis.reset_targets(model, q)

        while True:
            if vis.restart_requested:
                q = model._q_default.clone()
                vis.reset_targets(model, q)
            q = solve_ik(model, targets=vis.get_targets(), ...)
            vis.update(q)

    Typical usage (floating-base)::

        vis = Visualizer(urdf, model, floating_base=True)
        vis.add_grid()
        vis.add_target("left_rubber_hand")
        vis.add_timing_display()
        vis.add_restart_button()
        vis.reset_targets(model, q, base_pose)

        while True:
            if vis.restart_requested:
                q, base_pose = zeros_q, initial_base_pose
                vis.reset_targets(model, q, base_pose)
            base_pose, q = solve_ik(model, targets=vis.get_targets(), ...)
            vis.set_timing(elapsed_ms)
            vis.update(q, base_pose=base_pose)
    """

    def __init__(
        self,
        urdf: yourdfpy.URDF,
        model: RobotModel,
        port: int = 8080,
        floating_base: bool = False,
    ) -> None:
        self._model = model
        self._floating_base = floating_base
        self._server = viser.ViserServer(port=port)

        if floating_base:
            self._base_frame = self._server.scene.add_frame("/base", show_axes=False)
            self._urdf_handle = viser.extras.ViserUrdf(
                self._server, urdf, root_node_name="/base"
            )
        else:
            self._base_frame = None
            self._urdf_handle = viser.extras.ViserUrdf(self._server, urdf)

        self._targets: dict[str, viser.TransformControlsHandle] = {}
        self._timing_handle: viser.GuiInputHandle | None = None
        self._restart_flag: bool = False
        self._robot_coll: RobotCollision | None = None
        self._sphere_handles: list = []
        self._capsule_handles: list = []  # mesh handles used to visualise capsules
        self._collision_toggle: viser.GuiInputHandle | None = None

        print(f"Open http://localhost:{port} in your browser")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def add_target(self, link_name: str, scale: float = 0.12) -> None:
        """Add a draggable transform-control handle for an end-effector link.

        Args:
            link_name: Name of the robot link this handle controls.
            scale: Visual size of the transform gizmo.
        """
        handle = self._server.scene.add_transform_controls(
            f"/targets/{link_name}",
            scale=scale,
        )
        self._targets[link_name] = handle

    def add_grid(self, size: float = 4.0) -> None:
        """Add a ground-plane grid to the scene.

        Args:
            size: Grid side length in metres.
        """
        self._server.scene.add_grid("/ground", width=size, height=size)

    # ------------------------------------------------------------------
    # GUI elements
    # ------------------------------------------------------------------

    def add_timing_display(self) -> None:
        """Add a read-only elapsed-time display to the GUI panel."""
        self._timing_handle = self._server.gui.add_number(
            "Elapsed (ms)", initial_value=0.001, disabled=True
        )

    def add_restart_button(self) -> None:
        """Add a Restart button that signals :attr:`restart_requested`.

        When clicked the flag is set to True; the next call to
        :attr:`restart_requested` consumes it (returns True once, then False).
        """
        btn = self._server.gui.add_button("Restart")

        @btn.on_click
        def _(_event) -> None:
            self._restart_flag = True

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    @property
    def restart_requested(self) -> bool:
        """True once after the Restart button is clicked, then False."""
        if self._restart_flag:
            self._restart_flag = False
            return True
        return False

    def reset_targets(
        self,
        model: RobotModel,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Snap all target handles to the robot's FK poses.

        Args:
            model: RobotModel instance (used to compute FK).
            q: Current joint configuration, shape (n_joints,).
            base_pose: Current base pose [tx, ty, tz, qx, qy, qz, qw],
                required for floating-base robots.
        """
        fk = model.forward_kinematics(q, base_pose=base_pose) if base_pose is not None \
            else model.forward_kinematics(q)
        for link_name, handle in self._targets.items():
            idx = model.link_index(link_name)
            handle.position = fk[idx, :3].detach().numpy()
            handle.wxyz = qxyzw_to_wxyz(fk[idx, 3:7].detach())

    def get_targets(self) -> dict[str, torch.Tensor]:
        """Return current handle poses as SE3 tensors.

        Returns:
            Dict mapping link name to (7,) SE3 tensor [tx, ty, tz, qx, qy, qz, qw].
        """
        return {
            link_name: wxyz_pos_to_se3(handle.wxyz, handle.position)
            for link_name, handle in self._targets.items()
        }

    def update(
        self,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Update the robot mesh to reflect a new configuration.

        Args:
            q: Joint configuration, shape (n_joints,).
            base_pose: Base pose [tx, ty, tz, qx, qy, qz, qw],
                required for floating-base robots.
        """
        if self._floating_base and base_pose is not None and self._base_frame is not None:
            self._base_frame.position = base_pose[:3].detach().numpy()
            self._base_frame.wxyz = qxyzw_to_wxyz(base_pose[3:7].detach())
        self._urdf_handle.update_cfg(build_joint_dict(self._model, q))

    def add_collision_spheres(
        self,
        robot_coll: RobotCollision,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
        color: tuple[int, int, int] = (255, 80, 80),
        opacity: float = 0.3,
    ) -> None:
        """Add transparent sphere overlays for all collision spheres.

        Call once after constructing the visualizer. Then call
        :meth:`update_collision_spheres` every frame to keep positions in sync.

        Args:
            robot_coll: Sphere decomposition model.
            q: Initial joint configuration used to place spheres.
            base_pose: Initial base pose for floating-base robots.
            color: RGB colour for the spheres (0-255 per channel).
            opacity: Sphere transparency (0 = invisible, 1 = solid).
        """
        self._robot_coll = robot_coll
        self._sphere_handles = []
        spheres = robot_coll._get_world_spheres(self._model, q, base_pose=base_pose)
        for i, sphere in enumerate(spheres):
            handle = self._server.scene.add_icosphere(
                f"/collision_spheres/sphere_{i}",
                radius=float(sphere.radius),
                color=color,
                opacity=opacity,
                position=sphere.center.detach().numpy(),
            )
            self._sphere_handles.append(handle)

    def update_collision_spheres(
        self,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Update collision sphere positions to match the current configuration.

        Args:
            q: Current joint configuration.
            base_pose: Current base pose for floating-base robots.
        """
        if not self._sphere_handles or self._robot_coll is None:
            return
        spheres = self._robot_coll._get_world_spheres(self._model, q, base_pose=base_pose)
        for handle, sphere in zip(self._sphere_handles, spheres):
            handle.position = sphere.center.detach().numpy()

    def add_collision_capsules(
        self,
        robot_coll: RobotCollision,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
        color: tuple[int, int, int] = (255, 80, 80),
        opacity: float = 0.3,
    ) -> None:
        """Add transparent capsule mesh overlays for the capsule collision model.

        Each capsule is rendered as a proper cylinder-with-hemispherical-caps mesh.
        Call once, then call :meth:`update_collision_capsules` every frame.

        Args:
            robot_coll: Capsule collision model (``_mode == 'capsule'``).
            q: Initial joint configuration.
            base_pose: Initial base pose for floating-base robots.
            color: RGB colour (0-255 per channel).
            opacity: Transparency (0 = invisible, 1 = solid).
        """
        self._robot_coll = robot_coll
        self._capsule_handles = []
        # Store per-capsule (radius, height) so we can reuse the mesh geometry each frame.
        self._capsule_geom: list[tuple[float, float]] = []

        capsules = robot_coll._get_world_capsules(self._model, q, base_pose=base_pose)
        for ci, capsule in enumerate(capsules):
            pa = capsule.point_a.detach().numpy().astype(np.float64)
            pb = capsule.point_b.detach().numpy().astype(np.float64)
            radius = capsule.radius
            height = float(np.linalg.norm(pb - pa))

            verts, faces = _capsule_mesh(radius, height)
            center, wxyz = _capsule_pose(pa, pb)

            handle = self._server.scene.add_mesh_simple(
                f"/collision_capsules/cap_{ci}",
                vertices=verts,
                faces=faces,
                color=color,
                opacity=opacity,
                side="double",
                wxyz=wxyz,
                position=center,
            )
            self._capsule_handles.append(handle)
            self._capsule_geom.append((radius, height))

    def update_collision_capsules(
        self,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Update capsule mesh positions/orientations to match the current configuration.

        Args:
            q: Current joint configuration.
            base_pose: Current base pose for floating-base robots.
        """
        if not self._capsule_handles or self._robot_coll is None:
            return
        capsules = self._robot_coll._get_world_capsules(self._model, q, base_pose=base_pose)
        for handle, capsule in zip(self._capsule_handles, capsules):
            pa = capsule.point_a.detach().numpy().astype(np.float64)
            pb = capsule.point_b.detach().numpy().astype(np.float64)
            center, wxyz = _capsule_pose(pa, pb)
            handle.position = center
            handle.wxyz = wxyz

    def add_collision_geometry(
        self,
        robot_coll: RobotCollision,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
        color: tuple[int, int, int] = (255, 80, 80),
        opacity: float = 0.3,
    ) -> None:
        """Add collision geometry overlays, dispatching on mode.

        Calls :meth:`add_collision_capsules` for capsule mode and
        :meth:`add_collision_spheres` for sphere mode.

        Also adds a GUI checkbox to toggle the overlays on/off.
        """
        if robot_coll._mode == "capsule":
            self.add_collision_capsules(robot_coll, q, base_pose=base_pose, color=color, opacity=opacity)
            label = "Show capsules"
        else:
            self.add_collision_spheres(robot_coll, q, base_pose=base_pose, color=color, opacity=opacity)
            label = "Show spheres"

        self._collision_toggle = self._server.gui.add_checkbox(label, initial_value=True)

        @self._collision_toggle.on_update
        def _(_event) -> None:
            visible = self._collision_toggle.value  # type: ignore[union-attr]
            for h in self._capsule_handles:
                h.visible = visible
            for h in self._sphere_handles:
                h.visible = visible

    def update_collision_geometry(
        self,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Update collision geometry overlays, dispatching on mode."""
        if self._robot_coll is None:
            return
        if self._robot_coll._mode == "capsule":
            self.update_collision_capsules(q, base_pose=base_pose)
        else:
            self.update_collision_spheres(q, base_pose=base_pose)

    def set_timing(self, elapsed_ms: float) -> None:
        """Update the timing display with an exponential moving average.

        Args:
            elapsed_ms: Latest IK solve duration in milliseconds.
        """
        if self._timing_handle is not None:
            self._timing_handle.value = (
                0.99 * self._timing_handle.value + 0.01 * elapsed_ms
            )
