"""The data primitives on a fake server: lines, per-vertex coloured meshes, ragged clouds, a camera from its K."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh

from better_robot.exceptions import ArgumentError
from better_robot.testing import FakeServer
from better_robot.viewer import DARK, LIGHT, CameraLayer, LinesLayer, MeshLayer, PointsLayer, Viewer
from better_robot.viewer.theme import linear_uint8


def test_lines_step_their_segments_and_take_the_theme_ink_unless_coloured() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    points = np.zeros((3, 2, 2, 3), dtype=np.float32)
    points[:, :, 1, 2] = np.arange(1, 4)[:, None]  # step k: two vertical segments of length k + 1
    plain = viewer.add(LinesLayer("path", points, tags="paths"))
    viewer.add(LinesLayer("red", points[0], colors=(1.0, 0.0, 0.0), width=4.0), tags="paths")
    lines, red = server.scene.nodes["/paths/path/lines"], server.scene.nodes["/paths/red/lines"]
    assert plain.nsteps == 3 and red.props["thickness"] == 4.0
    viewer.show(2)
    np.testing.assert_allclose(lines.points[:, 1, 2], [3.0, 3.0])
    np.testing.assert_allclose(plain.bounds(2), [[0.0] * 3, [0.0, 0.0, 3.0]])
    np.testing.assert_array_equal(lines.props["colors"], np.full((2, 2, 3), linear_uint8(LIGHT.text)))
    viewer.set_theme(DARK)
    np.testing.assert_array_equal(lines.colors, np.full((2, 2, 3), linear_uint8(DARK.text)))
    np.testing.assert_array_equal(red.props["colors"], np.full((2, 2, 3), [255, 0, 0]))


def test_a_mesh_takes_one_colour_per_vertex_static_as_one_glb_deforming_as_one_node_per_colour() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 2, 1]])
    colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.4, 1.0], [0.2, 0.4, 1.0]])
    viewer.add(MeshLayer("still", vertices, faces=faces, colors=colors))
    glb = trimesh.load(io.BytesIO(server.scene.nodes["/still/still/mesh"].props["glb_data"]), file_type="glb")
    drawn = glb.geometry["geometry_0"].visual.vertex_attributes["color"][:, :3]
    np.testing.assert_array_equal(drawn, [[255, 0, 0]] * 2 + [[8, 34, 255]] * 2)  # glTF holds them linear
    clip = np.stack([vertices, vertices + (0.0, 0.0, 1.0)])  # (T = 2, V, 3)
    moving = viewer.add(MeshLayer("moving", clip, faces=faces, colors=colors, opacity=0.5))
    blue, red = server.scene.nodes["/moving/moving/mesh_0"], server.scene.nodes["/moving/moving/mesh_1"]
    # Each face takes its first vertex's colour: face 0 red (vertices 0, 1, 2), face 1 blue (vertices 3, 2, 1).
    assert red.props["color"].tolist() == [255, 0, 0] and blue.props["color"].tolist() == [51, 102, 255]
    np.testing.assert_array_equal(clip[0][red.props["faces"]], clip[0][faces[:1]])
    np.testing.assert_array_equal(clip[0][[1, 2, 3]][blue.props["faces"]], clip[0][faces[1:]])
    viewer.show(1)
    np.testing.assert_allclose(red.vertices[:, 2], 1.0)
    moving.set_opacity(0.8)
    assert red.opacity == blue.opacity == pytest.approx(0.8)
    many = np.random.default_rng(0).random((65 * 3, 3))
    with pytest.raises(ArgumentError, match="at most 64 distinct colors"):
        MeshLayer("noise", np.zeros((2, 65 * 3, 3)), faces=np.arange(65 * 3).reshape(65, 3), colors=many)


def test_a_ragged_cloud_pads_with_each_steps_first_point_and_hides_an_empty_step() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    steps = [np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]), np.zeros((0, 3)), np.array([[0.0, 3.0, 0.0]])]
    colors = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros((0, 3)), np.array([[0.0, 0.0, 1.0]])]
    cloud = viewer.add(PointsLayer("detections", steps, colors=colors))
    node = server.scene.nodes["/detections/detections/points"]
    assert cloud.nsteps == 3 and cloud.counts.tolist() == [2, 0, 1] and node.props["points"].shape == (2, 3)
    viewer.show(2)
    np.testing.assert_allclose(node.points, [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])  # the padded point repeats the first
    np.testing.assert_array_equal(node.colors, [[0, 0, 255], [0, 0, 255]])
    np.testing.assert_allclose(cloud.bounds(0), [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    assert node.visible
    viewer.show(1)
    assert node.visible is False and cloud.visible and cloud.bounds(1) is None
    viewer.show(0)
    assert node.visible is True


def test_a_camera_draws_its_frustum_from_the_intrinsics_and_swaps_images() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    poses = np.zeros((3, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    poses[:, 0] = np.arange(3)
    images = np.zeros((3, 4, 6, 3), dtype=np.uint8)
    images[2] = 255
    k = np.array([[100.0, 0.0, 20.0], [0.0, 50.0, 10.0], [0.0, 0.0, 1.0]])  # principal point off the 60 x 40 centre
    camera = viewer.add(CameraLayer("cam", poses, intrinsics=k, size=(60, 40), scale=0.5, image=images))
    edges = server.scene.nodes["/cam/cam/camera/edges"].props["points"]  # (8, 2, 3) in the camera frame
    corners = np.unique(edges.reshape(-1, 3), axis=0)  # the apex and the four corners, (u - cx) / fx * 0.5 ...
    expected = [[-0.1, -0.1, 0.5], [-0.1, 0.3, 0.5], [0.0, 0.0, 0.0], [0.2, -0.1, 0.5], [0.2, 0.3, 0.5]]
    np.testing.assert_allclose(corners, expected, atol=1e-6)
    picture = server.scene.nodes["/cam/cam/camera/image"]
    assert (picture.props["render_width"], picture.props["render_height"]) == pytest.approx((0.3, 0.4))
    np.testing.assert_allclose(picture.props["position"], [0.05, 0.1, 0.5], atol=1e-6)
    viewer.show(2)
    frame = server.scene.nodes["/cam/cam/camera"]
    assert frame.position[0] == 2.0 and picture.image.min() == 255
    assert camera.bounds(2) is not None
    lazy = viewer.add(CameraLayer("lazy", poses, intrinsics=k, size=(60, 40), image=lambda step: images[step]))
    viewer.show(1)
    assert lazy.image_at(1) is not None and server.scene.nodes["/lazy/lazy/camera/image"].image.shape == (4, 6, 3)
