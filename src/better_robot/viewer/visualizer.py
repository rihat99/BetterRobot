"""Viser-based robot visualizer — encapsulates all scene and GUI setup."""
from __future__ import annotations

import torch
import yourdfpy
import viser
import viser.extras

from ..models.robot_model import RobotModel
from .helpers import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict


class Visualizer:
    """Interactive robot visualizer backed by viser.

    Typical usage (fixed-base)::

        vis = Visualizer(urdf, model)
        vis.add_target("panda_hand")
        vis.add_restart_button()
        vis.reset_targets(model, cfg)

        while True:
            if vis.restart_requested:
                cfg = model._default_cfg.clone()
                vis.reset_targets(model, cfg)
            cfg = solve_ik(model, targets=vis.get_targets(), ...)
            vis.update(cfg)

    Typical usage (floating-base)::

        vis = Visualizer(urdf, model, floating_base=True)
        vis.add_grid()
        vis.add_target("left_rubber_hand")
        vis.add_timing_display()
        vis.add_restart_button()
        vis.reset_targets(model, cfg, base_pose)

        while True:
            if vis.restart_requested:
                cfg, base_pose = zeros_cfg, initial_base_pose
                vis.reset_targets(model, cfg, base_pose)
            base_pose, cfg = solve_ik(model, targets=vis.get_targets(), ...)
            vis.set_timing(elapsed_ms)
            vis.update(cfg, base_pose=base_pose)
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
        cfg: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Snap all target handles to the robot's FK poses.

        Args:
            model: RobotModel instance (used to compute FK).
            cfg: Current joint configuration, shape (n_joints,).
            base_pose: Current base pose [tx, ty, tz, qx, qy, qz, qw],
                required for floating-base robots.
        """
        fk = model.forward_kinematics(cfg, base_pose=base_pose) if base_pose is not None \
            else model.forward_kinematics(cfg)
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
        cfg: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> None:
        """Update the robot mesh to reflect a new configuration.

        Args:
            cfg: Joint configuration, shape (n_joints,).
            base_pose: Base pose [tx, ty, tz, qx, qy, qz, qw],
                required for floating-base robots.
        """
        if self._floating_base and base_pose is not None and self._base_frame is not None:
            self._base_frame.position = base_pose[:3].detach().numpy()
            self._base_frame.wxyz = qxyzw_to_wxyz(base_pose[3:7].detach())
        self._urdf_handle.update_cfg(build_cfg_dict(self._model, cfg))

    def set_timing(self, elapsed_ms: float) -> None:
        """Update the timing display with an exponential moving average.

        Args:
            elapsed_ms: Latest IK solve duration in milliseconds.
        """
        if self._timing_handle is not None:
            self._timing_handle.value = (
                0.99 * self._timing_handle.value + 0.01 * elapsed_ms
            )
