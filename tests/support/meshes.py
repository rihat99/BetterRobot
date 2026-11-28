"""The test meshes that several test files build."""

from __future__ import annotations

import torch


def box_mesh(dtype: torch.dtype, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """The closed, outward-wound box of size ``(2, 3, 4)`` centred at the origin: ``vertices (8, 3)`` in ``dtype`` and
    ``faces (12, 3)`` int64, both on ``device``."""
    vertices = torch.tensor(
        [
            [-1.0, -1.5, -2.0],
            [1.0, -1.5, -2.0],
            [1.0, 1.5, -2.0],
            [-1.0, 1.5, -2.0],
            [-1.0, -1.5, 2.0],
            [1.0, -1.5, 2.0],
            [1.0, 1.5, 2.0],
            [-1.0, 1.5, 2.0],
        ],
        dtype=dtype,
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=torch.int64,
        device=device,
    )
    return vertices, faces
