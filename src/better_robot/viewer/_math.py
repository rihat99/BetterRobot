"""Small numpy helpers shared by the viewer layers (no torch, no viser)."""

from __future__ import annotations

import numpy as np

RGB = tuple[float, float, float]


def as_frames(array: np.ndarray | object, event_ndim: int) -> np.ndarray:
    """Promote an array to ``(T, ...)`` float32: a static array (no leading axis) becomes ``T = 1``.

    ``event_ndim`` is the number of trailing axes of one step, e.g. 2 for joints ``(J, 3)``.
    """
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == event_ndim:
        out = out[None]
    if out.ndim != event_ndim + 1:
        raise ValueError(f"expected {event_ndim} or {event_ndim + 1} axes, got shape {out.shape}")
    return np.ascontiguousarray(out)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Reorder quaternions ``(..., 4)`` from scalar-last (BetterRobot) to scalar-first (viser)."""
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def compose_poses(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose xyzw poses ``(..., 7)``: ``a ∘ b`` (apply ``b`` in the frame of ``a``), broadcasting."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
    qa, qb = a[..., 3:], b[..., 3:]
    xa, ya, za, wa = (qa[..., i] for i in range(4))
    xb, yb, zb, wb = (qb[..., i] for i in range(4))
    quat = np.stack(
        [
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
            wa * wb - xa * xb - ya * yb - za * zb,
        ],
        axis=-1,
    )
    return np.concatenate([a[..., :3] + rotate(qa, b[..., :3]), quat], axis=-1)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v (..., 3)`` by unit quaternions ``q (..., 4)`` in xyzw order."""
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def colormap(values: np.ndarray, *stops: RGB) -> np.ndarray:
    """Piecewise-linear colours ``(..., 3)`` through evenly spaced ``stops`` (at least two) for ``values`` in
    ``[0, 1]`` (clamped); two stops are a plain blend from the first to the second."""
    stops_array = np.asarray(stops, dtype=np.float32)
    position = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0) * (len(stops_array) - 1)
    segment = np.minimum(position.astype(np.int64), len(stops_array) - 2)
    t = (position - segment)[..., None]
    return (1.0 - t) * stops_array[segment] + t * stops_array[segment + 1]


def quat_z_to(direction: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion (wxyz) rotating ``+z`` onto each ``direction`` ``(..., 3)``.

    Zero directions map to the identity; anti-parallel ones to a half turn about ``x``.
    """
    d = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d = np.where(norm > 1e-9, d / np.maximum(norm, 1e-9), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    # q = (1 + z·d, z × d) is the un-normalised half-angle quaternion between z and d.
    w = 1.0 + d[..., 2]
    xyz = np.stack([-d[..., 1], d[..., 0], np.zeros_like(w)], axis=-1)
    q = np.concatenate([w[..., None], xyz], axis=-1)
    flipped = w < 1e-6
    q[flipped] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def axis_wxyz(quat: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Scalar-first quaternions ``(..., 4)`` placing a ``+z``-aligned primitive along ``axis (..., 3)`` (given in
    the joint frame) for joint rotations ``quat (..., 4)`` (xyzw): ``quat * shortest_arc(z, axis)``."""
    align = np.roll(quat_z_to(axis), -1, axis=-1)  # wxyz -> xyzw
    quat = np.asarray(quat, dtype=np.float32)
    a = np.concatenate([np.zeros_like(quat[..., :3]), quat], axis=-1)
    b = np.concatenate([np.zeros_like(align[..., :3]), align], axis=-1)
    return xyzw_to_wxyz(compose_poses(a, b)[..., 3:])


def unit_box() -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned cube from ``-0.5`` to ``0.5``: ``(vertices (8, 3), faces (12, 3))``, outward winding."""
    vertices = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=np.float32)
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.uint32,
    )  # fmt: skip
    return vertices, faces


def rgb_uint8(color: np.ndarray | RGB, shape: tuple[int, ...] = ()) -> np.ndarray:
    """Convert colours in ``[0, 1]`` to uint8 ``(*shape, 3)``, broadcasting a single colour."""
    out = np.asarray(color, dtype=np.float32)
    if out.ndim == 1 and shape:
        out = np.broadcast_to(out, (*shape, 3))
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def unit_cylinder(segments: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder of radius 1 from ``z = -0.5`` to ``z = 0.5``: ``(vertices (V, 3), faces (F, 3))``."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    bottom = np.concatenate([ring, np.full((segments, 1), -0.5, dtype=np.float32)], axis=1)
    top = np.concatenate([ring, np.full((segments, 1), 0.5, dtype=np.float32)], axis=1)
    vertices = np.concatenate([bottom, top, [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]], axis=0).astype(np.float32)
    i = np.arange(segments)
    j = (i + 1) % segments
    faces = np.concatenate(
        [
            np.stack([i, j, segments + j], axis=1),
            np.stack([i, segments + j, segments + i], axis=1),
            np.stack([np.full_like(i, 2 * segments), j, i], axis=1),
            np.stack([np.full_like(i, 2 * segments + 1), segments + i, segments + j], axis=1),
        ]
    )
    return vertices, faces.astype(np.uint32)


def unit_cone(segments: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Cone of base radius 1 at ``z = 0`` and apex at ``z = 1``: ``(vertices, faces)``."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    base = np.stack([np.cos(angles), np.sin(angles), np.zeros(segments, dtype=np.float32)], axis=1)
    vertices = np.concatenate([base, [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], axis=0).astype(np.float32)
    i = np.arange(segments)
    j = (i + 1) % segments
    apex, center = segments, segments + 1
    faces = np.concatenate(
        [np.stack([i, j, np.full_like(i, apex)], axis=1), np.stack([np.full_like(i, center), j, i], axis=1)]
    )
    return vertices, faces.astype(np.uint32)


def unit_sphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Icosphere of radius 1: ``(vertices, faces)``."""
    phi = (1.0 + 5.0**0.5) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )  # fmt: skip
    for _ in range(subdivisions):
        vertex_list = list(vertices)
        midpoint: dict[tuple[int, int], int] = {}

        def split(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                midpoint[key] = len(vertex_list)
                vertex_list.append((vertex_list[a] + vertex_list[b]) / 2.0)
            return midpoint[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = split(a, b), split(b, c), split(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        vertices = np.array(vertex_list)
        faces = np.array(new_faces, dtype=np.int64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices.astype(np.float32), faces.astype(np.uint32)
