"""Ray, sphere-brush and rectangle-marquee selection over vertices (numpy, no viser import).

viser conventions this reproduces (viser 1.0.26): pointer ``screen_pos`` are image coordinates in
``[0, 1]`` with ``(0, 0)`` top-left; ``CameraHandle.wxyz`` is scalar-first with the camera frame
following OpenCV (look ``+z``, up ``-y``, right ``+x``); ``fov`` is the vertical field of view in
radians and ``aspect`` is width over height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraFrustum:
    """What is needed to project world points the way viser's client does."""

    position: np.ndarray  #: ``(3,)`` world position.
    wxyz: np.ndarray  #: ``(4,)`` scalar-first orientation, ``R_world_camera``.
    fov: float  #: vertical field of view, radians.
    aspect: float  #: width / height.

    @classmethod
    def from_camera(cls, camera: Any) -> CameraFrustum:
        """Snapshot a ``viser.CameraHandle`` (``client.camera``)."""
        return cls(
            np.asarray(camera.position, dtype=np.float64),
            np.asarray(camera.wxyz, dtype=np.float64),
            float(camera.fov),
            float(camera.aspect),
        )


def ray_hit(
    origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, int] | None:
    """Nearest double-sided Möller–Trumbore intersection ahead of ``origin``: ``(point (3,), face)`` or ``None``."""
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


def nearest_on_ray(points: np.ndarray, origin: np.ndarray, direction: np.ndarray, radius: float) -> int | None:
    """Index of the point ``(N, 3)`` within ``radius`` of the ray nearest its ``origin (3,)``, or ``None``.

    ``direction (3,)`` need not be unit length; points behind the origin are ignored.
    """
    direction = direction / np.linalg.norm(direction)
    along = (points - origin) @ direction
    distance = np.linalg.norm(points - origin - along[:, None] * direction, axis=-1)
    hits = np.flatnonzero((distance <= radius) & (along > 0))
    return None if hits.size == 0 else int(hits[np.argmin(along[hits])])


def vertices_in_sphere(vertices: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Indices of ``vertices (V, 3)`` within ``radius`` of ``center (3,)``."""
    distance_squared = ((np.asarray(vertices) - np.asarray(center)) ** 2).sum(-1)
    return np.nonzero(distance_squared <= radius * radius)[0]


def vertices_in_rect(
    vertices: np.ndarray, camera: CameraFrustum, rect_min: np.ndarray, rect_max: np.ndarray
) -> np.ndarray:
    """Indices of vertices inside a screen marquee (corners in ``[0, 1]``, top-left origin), in front of the camera."""
    lower = np.minimum(rect_min, rect_max)
    upper = np.maximum(rect_min, rect_max)
    screen, depth = project(vertices, camera)
    inside = (depth > 0) & (screen >= lower).all(-1) & (screen <= upper).all(-1)
    return np.nonzero(inside)[0]


def project(vertices: np.ndarray, camera: CameraFrustum) -> tuple[np.ndarray, np.ndarray]:
    """``(screen (V, 2) in [0, 1], depth (V,))`` of world points under viser's screen convention."""
    rotation = _rotation_from_wxyz(camera.wxyz)
    camera_space = (np.asarray(vertices, dtype=np.float64) - camera.position) @ rotation
    depth = camera_space[..., 2]
    safe_depth = np.where(depth > 0, depth, 1.0)
    tan_half = math.tan(camera.fov / 2.0)
    ndc_x = camera_space[..., 0] / safe_depth / (tan_half * camera.aspect)
    ndc_y = camera_space[..., 1] / safe_depth / tan_half
    return np.stack([0.5 + 0.5 * ndc_x, 0.5 + 0.5 * ndc_y], axis=-1), depth


def _rotation_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64) / max(float(np.linalg.norm(wxyz)), 1e-8)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
