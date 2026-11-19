"""Ray, sphere-brush and rectangle-marquee selection over vertices (numpy, no viser import).

viser conventions this reproduces (viser 1.1.1): pointer ``screen_pos``, ``screen_min`` and ``screen_max`` are
image coordinates in ``[0, 1]`` with ``(0, 0)`` top-left; the camera frame follows OpenCV (look ``+z``, up ``-y``,
right ``+x``); ``fov`` is the vertical field of view in radians and ``aspect`` is width over height.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import Any, final, override

import numpy as np
import torch

from ...exceptions import ShapeError
from ...lie import so3


@final
@dataclass(frozen=True)
class CameraFrustum:
    """A browser's camera as picking needs it, to project world points the way viser's client does.

    ``pose (7,)`` ``[x, y, z, qx, qy, qz, qw]`` is camera-to-world with OpenCV axes (``+z`` forward, ``+x`` right,
    ``+y`` down); ``fov`` is the vertical field of view in radians and ``aspect`` the view's width over its height.
    Raises ``ShapeError`` for a ``pose`` that is not ``(7,)``.
    """

    pose: np.ndarray
    _: KW_ONLY
    fov: float
    aspect: float

    def __post_init__(self) -> None:
        """Raise ``ShapeError`` unless ``pose`` is ``(7,)``."""
        if np.shape(self.pose) != (7,):
            raise ShapeError(f"pose must be (7,) = [x, y, z, qx, qy, qz, qw], got {np.shape(self.pose)}")

    @override
    def __repr__(self) -> str:
        pose = np.round(np.asarray(self.pose, dtype=np.float64), 3).tolist()
        return f"CameraFrustum(pose={pose}, fov={self.fov:.3f}, aspect={self.aspect:.3f})"

    @classmethod
    def from_camera(cls, camera: Any) -> CameraFrustum:
        """The frustum of a ``viser.CameraHandle`` (``client.camera``) now, its scalar-first ``wxyz`` reordered."""
        wxyz = np.asarray(camera.wxyz, dtype=np.float64)
        position = np.asarray(camera.position, dtype=np.float64)
        return cls(np.concatenate([position, wxyz[[1, 2, 3, 0]]]), fov=float(camera.fov), aspect=float(camera.aspect))


def ray_hit(
    origin: np.ndarray, direction: np.ndarray, *, vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, int] | None:
    """``(point (3,), face)``: where the ray from ``origin (3,)`` along ``direction (3,)`` first meets a triangle
    ``faces (F, 3)`` of ``vertices (V, 3)``, from either side (Möller–Trumbore), and that triangle's index; ``None``
    when it meets none ahead of ``origin``."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0
    pvec = np.cross(direction, edge2)
    det = (edge1 * pvec).sum(-1)
    parallel = np.abs(det) < 1e-8
    inv_det = 1.0 / np.where(parallel, 1.0, det)
    tvec = origin - v0
    u = (tvec * pvec).sum(-1) * inv_det
    qvec = np.cross(tvec, edge1)
    v = (direction * qvec).sum(-1) * inv_det
    t = (edge2 * qvec).sum(-1) * inv_det
    tol = 1e-6
    valid = ~parallel & (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol) & (t > 1e-8)
    if not valid.any():
        return None
    face = int(np.argmin(np.where(valid, t, np.inf)))
    return origin + t[face] * direction, face


def nearest_on_ray(points: np.ndarray, origin: np.ndarray, *, direction: np.ndarray, radius: float) -> int | None:
    """Index of the point of ``points (N, 3)`` within ``radius`` of the ray from ``origin (3,)`` that lies nearest
    ``origin`` along it, or ``None``.

    ``direction (3,)`` need not be unit length; points behind the origin are ignored.
    """
    direction = direction / np.linalg.norm(direction)
    along = (points - origin) @ direction
    distance = np.linalg.norm(points - origin - along[:, None] * direction, axis=-1)
    hits = np.flatnonzero((distance <= radius) & (along > 0))
    return None if hits.size == 0 else int(hits[np.argmin(along[hits])])


def vertices_in_sphere(vertices: np.ndarray, center: np.ndarray, *, radius: float) -> np.ndarray:
    """Indices ``(K,)`` of ``vertices (V, 3)`` within ``radius`` of ``center (3,)``."""
    distance_squared = ((np.asarray(vertices) - np.asarray(center)) ** 2).sum(-1)
    return np.nonzero(distance_squared <= radius * radius)[0]


def vertices_in_rect(
    vertices: np.ndarray, camera: CameraFrustum, *, rect_min: np.ndarray, rect_max: np.ndarray
) -> np.ndarray:
    """Indices ``(K,)`` of ``vertices (V, 3)`` in front of ``camera`` that project inside the screen rectangle of
    corners ``rect_min (2,)`` and ``rect_max (2,)`` (in ``[0, 1]``, top-left origin, either order), as viser's box
    selection gives them."""
    lower = np.minimum(rect_min, rect_max)
    upper = np.maximum(rect_min, rect_max)
    screen, depth = project(vertices, camera)
    inside = (depth > 0) & (screen >= lower).all(-1) & (screen <= upper).all(-1)
    return np.nonzero(inside)[0]


def project(vertices: np.ndarray, camera: CameraFrustum) -> tuple[np.ndarray, np.ndarray]:
    """``(screen (V, 2), depth (V,))`` of world points ``vertices (V, 3)`` seen by ``camera``: screen coordinates in
    ``[0, 1]`` with ``(0, 0)`` top left, as viser's pointer events give them, and the distance along the camera's
    ``+z`` (positive in front of it)."""
    pose = np.asarray(camera.pose, dtype=np.float64)
    xyzw = pose[3:] / max(float(np.linalg.norm(pose[3:])), 1e-8)
    world = torch.from_numpy(np.asarray(vertices, dtype=np.float64) - pose[:3])
    camera_space = so3.act(so3.inverse(torch.from_numpy(xyzw)), world).numpy()
    depth = camera_space[..., 2]
    safe_depth = np.where(depth > 0, depth, 1.0)
    tan_half = math.tan(camera.fov / 2.0)
    ndc_x = camera_space[..., 0] / safe_depth / (tan_half * camera.aspect)
    ndc_y = camera_space[..., 1] / safe_depth / tan_half
    return np.stack([0.5 + 0.5 * ndc_x, 0.5 + 0.5 * ndc_y], axis=-1), depth
