"""Thin viser wrapper for robot visualization."""

from __future__ import annotations

import torch
import yourdfpy
import viser
import viser.extras


class Visualizer:
    """Wraps viser for interactive robot visualization."""

    def __init__(self, urdf: yourdfpy.URDF, port: int = 8080) -> None:
        """Initialize viser server and load URDF mesh.

        Args:
            urdf: Loaded yourdfpy.URDF.
            port: Port for the viser web server.
        """
        self._server = viser.ViserServer(port=port)
        self._urdf_handle = viser.extras.ViserUrdf(self._server, urdf)

    def update_cfg(self, cfg: torch.Tensor) -> None:
        """Update the robot visualization to a new joint configuration.

        Args:
            cfg: Shape (num_actuated_joints,). Joint configuration.
        """
        cfg_np = cfg.detach().cpu().numpy()
        self._urdf_handle.update_cfg(cfg_np)
