"""Gizmo pose round trip and picking math."""

from __future__ import annotations

import numpy as np

from better_robot.viewer.tools import CameraFrustum, PoseGizmo, project, ray_hit, vertices_in_rect, vertices_in_sphere
from better_robot.viewer.viewer import Viewer

from .fake_server import FakeServer


def test_gizmo_reports_poses_in_xyzw_order_and_fires_callbacks() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    seen: list[np.ndarray] = []
    gizmo = PoseGizmo(viewer, "target", [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0], on_change=seen.append)
    handle = server.scene.nodes["/tools/target"]
    np.testing.assert_allclose(handle.props["wxyz"], [0.0, 0.0, 0.0, 1.0])
    handle.position = np.array([4.0, 5.0, 6.0])
    handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    handle.fire()
    np.testing.assert_allclose(seen[0], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    gizmo.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(handle.wxyz, [0.0, 1.0, 0.0, 0.0])


def test_ray_hit_sphere_and_rect_selection() -> None:
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-5.0, 5.0, 5.0]])
    faces = np.array([[0, 1, 2]])
    hit = ray_hit(np.array([0.2, 0.2, -1.0]), np.array([0.0, 0.0, 1.0]), vertices, faces)
    assert hit is not None
    np.testing.assert_allclose(hit[0], [0.2, 0.2, 1.0])
    assert hit[1] == 0
    assert ray_hit(np.array([0.2, 0.2, 2.0]), np.array([0.0, 0.0, 1.0]), vertices, faces) is None
    assert vertices_in_sphere(vertices, np.zeros(3), 1.5).tolist() == [0, 1, 2]

    # Camera at the origin looking down +z (identity orientation): the triangle sits in front, the far
    # vertex off-centre; a marquee around the image centre selects the first three vertices only.
    camera = CameraFrustum(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), fov=np.pi / 2, aspect=1.0)
    screen, depth = project(vertices, camera)
    np.testing.assert_allclose(screen[0], [0.5, 0.5])
    np.testing.assert_allclose(screen[1], [1.0, 0.5])
    assert (depth > 0).all()
    assert vertices_in_rect(vertices, camera, np.array([0.4, 0.4]), np.array([1.0, 1.0])).tolist() == [0, 1, 2]
