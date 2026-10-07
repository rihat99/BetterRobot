"""Skinned mesh (client bones + numpy LBS) and camera layers on a synthetic two-bone mesh."""

from __future__ import annotations

import math

import numpy as np

from better_robot.viewer import CameraLayer, SkinnedMeshLayer
from better_robot.viewer.layers.skinned import top_k_weights

from .fake_server import FakeServer


def test_two_bone_mesh_bends_at_the_second_bone() -> None:
    # Vertices along +z from 0 to 2; bone 0 at the origin, bone 1 at z = 1 (bind pose, no rotation).
    vertices = np.array([[0.1, 0.0, z] for z in np.linspace(0.0, 2.0, 9)], dtype=np.float32)
    faces = np.array([[i, i + 1, (i + 2) % 9] for i in range(7)])
    weights = np.zeros((9, 2), dtype=np.float32)
    weights[:, 0] = np.clip(1.0 - vertices[:, 2], 0.0, 1.0)
    weights[:, 1] = 1.0 - weights[:, 0]
    bind = np.array([[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 1.0, 0, 0, 0, 1.0]], dtype=np.float32)
    # Frame 1: bone 1 rotated 90° about x at its own position → the upper half folds towards -y.
    half = math.sqrt(0.5)
    posed = np.stack([bind, np.array([[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 1.0, half, 0, 0, half]], dtype=np.float32)])
    layer = SkinnedMeshLayer("body", vertices, faces, weights, bind, posed)
    assert layer.num_frames == 2
    np.testing.assert_allclose(layer.skin(0), vertices, atol=1e-6)
    bent = layer.skin(1)
    np.testing.assert_allclose(bent[[0, 4]], vertices[[0, 4]], atol=1e-6)  # on bone 0 only / at bone 1
    np.testing.assert_allclose(bent[8], [0.1, -1.0, 1.0], atol=1e-6)  # tip: (0.1, 0, 2) rotated about bone 1

    server = FakeServer()
    layer.build(server.scene, "/g/body")
    node = server.scene.nodes["/g/body/mesh"]
    assert node.props["skin_weights"].shape == (9, 2)
    np.testing.assert_allclose(node.props["bone_wxyzs"][1], [1.0, 0.0, 0.0, 0.0])
    node.bones = [server.scene.nodes["/g/body/mesh"].__class__(None, "bone", f"b{i}", {}) for i in range(2)]
    layer.set_frame(1)
    np.testing.assert_allclose(node.bones[1].wxyz, [half, half, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(node.bones[1].position, [0.0, 0.0, 1.0])

    sparse = top_k_weights(np.array([[0.4, 0.3, 0.2, 0.05, 0.05]], dtype=np.float32), 4)
    np.testing.assert_allclose(sparse.sum(), 1.0)
    assert (sparse > 0).sum() == 4


def test_camera_layer_moves_the_frustum_and_swaps_images() -> None:
    scene = FakeServer().scene
    poses = np.zeros((3, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    poses[:, 0] = np.arange(3)
    images = np.zeros((3, 4, 6, 3), dtype=np.uint8)
    images[2] = 255
    layer = CameraLayer("cam", poses, fov=1.0, aspect=1.5, image=images)
    layer.build(scene, "/g/cam")
    node = scene.nodes["/g/cam/frustum"]
    assert node.props["image"].max() == 0 and node.props["aspect"] == 1.5
    layer.set_frame(2)
    assert node.position[0] == 2.0 and node.image.min() == 255
    lazy = CameraLayer("lazy", poses, fov=1.0, aspect=1.0, image=lambda k: images[k])
    lazy.build(scene, "/g/lazy")
    lazy.set_frame(1)
    assert scene.nodes["/g/lazy/frustum"].image.shape == (4, 6, 3)
