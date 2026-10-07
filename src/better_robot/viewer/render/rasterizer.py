"""A small torch z-buffer rasterizer: world triangles + camera → RGB image. No GPU libraries.

Triangles are binned into screen tiles; every (triangle, tile) pair is evaluated on the tile's
pixels with edge functions, depths go through ``scatter_reduce(amin)`` into the z-buffer and the
winning pair writes its flat-shaded colour. Everything is vectorised, so it runs the same on CPU
and CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

TILE = 16


@dataclass(frozen=True)
class Camera:
    """Pinhole camera: ``position`` looking at ``target`` with ``up``; ``fov`` is vertical, radians."""

    position: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = math.radians(45.0)

    def view_matrix(self) -> np.ndarray:
        """``(3, 4)`` world → camera (OpenCV: ``+z`` forward, ``-y`` up, ``+x`` right)."""
        position = np.asarray(self.position, dtype=np.float64)
        forward = np.asarray(self.target, dtype=np.float64) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(self.up, dtype=np.float64))
        if np.linalg.norm(right) < 1e-9:  # looking along ``up``: any perpendicular axis will do
            right = np.cross(forward, np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])  # rows: camera axes in world coordinates
        return np.concatenate([rotation, (-rotation @ position)[:, None]], axis=1)

    @classmethod
    def framing(
        cls,
        points: np.ndarray,
        *,
        azimuth: float = math.radians(35.0),
        elevation: float = math.radians(25.0),
        fov: float = math.radians(45.0),
        margin: float = 1.3,
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> Camera:
        """Camera on a 3/4 view that fits the bounding sphere of ``points (N, 3)``; ``up`` is the world up axis."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        center = (points.min(0) + points.max(0)) / 2.0
        radius = max(float(np.linalg.norm(points - center, axis=-1).max()), 1e-3)
        distance = margin * radius / math.sin(fov / 2.0)
        up_axis = np.asarray(up, dtype=np.float64)
        up_axis /= np.linalg.norm(up_axis)
        side = np.cross(np.array([0.0, 0.0, 1.0]) if abs(up_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0]), up_axis)
        side /= np.linalg.norm(side)
        forward = np.cross(up_axis, side)
        direction = (
            math.cos(elevation) * math.cos(azimuth) * side
            + math.cos(elevation) * math.sin(azimuth) * forward
            + math.sin(elevation) * up_axis
        )
        return cls(tuple(center + distance * direction), tuple(center), up=tuple(up_axis), fov=fov)


def rasterize(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    camera: Camera,
    size: tuple[int, int],
    *,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    light_direction: tuple[float, float, float] = (-0.4, -0.6, -1.0),
) -> torch.Tensor:
    """Render ``faces (F, 3)`` over ``vertices (V, 3)`` with per-face ``colors (F, 3)`` in ``[0, 1]``.

    ``size`` is ``(width, height)``. Returns ``(H, W, 3)`` float in ``[0, 1]``; faces are lit
    double-sided by one directional light plus ambient.
    """
    width, height = size
    device = vertices.device
    image = torch.empty(height * width, 3, device=device).copy_(torch.tensor(background, device=device))
    tri, tri_depth, keep = _project(vertices, faces, camera, size)
    if keep.numel() == 0:
        return image.view(height, width, 3)
    area = _edge(tri[:, 0], tri[:, 1], tri[:, 2])
    shade = _flat_shading(vertices[faces[keep]], colors[keep], light_direction)

    # Small triangles get one block anchored at their bounding box; large ones are binned into tiles.
    extent = (tri.amax(dim=1) - tri.amin(dim=1)).amax(dim=1)
    indices, depths, owners = [], [], []
    # An anchored block starts at floor(lo), so it needs one pixel more than the bound to reach hi.
    classes = [(extent <= 4, 5, True), ((extent > 4) & (extent <= 8), 9, True), (extent > 8, TILE, False)]
    for mask, block, anchored in classes:
        selected = mask.nonzero().squeeze(1)
        if selected.numel() == 0:
            continue
        pair_face, pixel, index = _block_pixels(tri[selected], size, block, anchored)
        face = selected[pair_face]
        a, b, c = tri[face, 0, None], tri[face, 1, None], tri[face, 2, None]
        w0, w1, w2 = _edge(b, c, pixel), _edge(c, a, pixel), _edge(a, b, pixel)
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        inside &= (pixel >= 0).all(-1) & (pixel[..., 0] < width) & (pixel[..., 1] < height)
        bary = torch.stack([w0, w1, w2], dim=-1) / area[face, None, None]
        pixel_depth = 1.0 / (bary / tri_depth[face, None]).sum(-1).clamp(min=1e-9)
        indices.append(index[inside])
        depths.append(pixel_depth[inside])
        owners.append(face[:, None].expand_as(inside)[inside])

    flat_index, flat_depth, flat_face = torch.cat(indices), torch.cat(depths), torch.cat(owners)
    zbuffer = torch.full((height * width,), float("inf"), device=device, dtype=vertices.dtype)
    zbuffer.scatter_reduce_(0, flat_index, flat_depth, reduce="amin")
    winner = flat_depth <= zbuffer[flat_index]
    image[flat_index[winner]] = shade[flat_face[winner]]
    return image.view(height, width, 3)


def _project(
    vertices: torch.Tensor, faces: torch.Tensor, camera: Camera, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Screen triangles ``(K, 3, 2)``, their vertex depths ``(K, 3)`` and the kept face indices ``(K,)``."""
    width, height = size
    view = torch.as_tensor(camera.view_matrix(), dtype=vertices.dtype, device=vertices.device)
    cam = vertices @ view[:, :3].T + view[:, 3]
    focal = height / (2.0 * math.tan(camera.fov / 2.0))
    depth = cam[:, 2]
    safe = depth.clamp(min=1e-6)
    screen = torch.stack([width / 2.0 + focal * cam[:, 0] / safe, height / 2.0 + focal * cam[:, 1] / safe], dim=-1)
    tri, tri_depth = screen[faces], depth[faces]
    lo, hi = tri.amin(dim=1), tri.amax(dim=1)
    visible = (tri_depth > 1e-6).all(dim=1) & (_edge(tri[:, 0], tri[:, 1], tri[:, 2]).abs() > 1e-12)
    visible &= (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < width) & (lo[:, 1] < height)
    keep = visible.nonzero().squeeze(1)
    return tri[keep], tri_depth[keep], keep


def _block_pixels(
    tri: torch.Tensor, size: tuple[int, int], block: int, anchored: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(triangle, block) pairs: owning triangle ``(P,)``, pixel centres ``(P, block², 2)``, flat pixel index.

    ``anchored`` places one block at each triangle's bounding-box corner (small triangles); otherwise
    blocks are the fixed screen tiles overlapped by the triangle.
    """
    width, height = size
    device = tri.device
    lo, hi = tri.amin(dim=1), tri.amax(dim=1)
    if anchored:
        pair_face = torch.arange(tri.shape[0], device=device)
        origin = lo.floor().long()
    else:
        tiles_x, tiles_y = math.ceil(width / block), math.ceil(height / block)
        t0 = (lo / block).floor().long()
        t1 = (hi / block).floor().long()
        tx0, ty0 = t0[:, 0].clamp(0, tiles_x - 1), t0[:, 1].clamp(0, tiles_y - 1)
        tx1, ty1 = t1[:, 0].clamp(0, tiles_x - 1), t1[:, 1].clamp(0, tiles_y - 1)
        span_x, span_y = tx1 - tx0 + 1, ty1 - ty0 + 1
        count = span_x * span_y
        pair_face = torch.repeat_interleave(torch.arange(count.numel(), device=device), count)
        local = torch.arange(int(count.sum()), device=device) - (torch.cumsum(count, 0) - count)[pair_face]
        origin = torch.stack(
            [tx0[pair_face] + local % span_x[pair_face], ty0[pair_face] + local // span_x[pair_face]], -1
        )
        origin = origin * block
    px = torch.arange(block, device=device, dtype=tri.dtype) + 0.5
    grid_x = (origin[:, 0, None] + px[None]).repeat_interleave(block, dim=1)
    grid_y = (origin[:, 1, None] + px[None]).repeat(1, block)
    index = (grid_y.long() * width + grid_x.long()).clamp(0, height * width - 1)
    return pair_face, torch.stack([grid_x, grid_y], dim=-1), index


def _edge(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])


def _flat_shading(
    triangles: torch.Tensor, colors: torch.Tensor, light_direction: tuple[float, float, float]
) -> torch.Tensor:
    normal = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    light = -torch.tensor(light_direction, dtype=triangles.dtype, device=triangles.device)
    light = light / light.norm()
    diffuse = (normal @ light).abs()[:, None]
    return (colors * (0.55 + 0.45 * diffuse)).clamp(0.0, 1.0)
