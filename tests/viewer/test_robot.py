"""Robot adapter and mesh layers: poses compose correctly, FK runs once per clip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.lie.se3 import compose
from better_robot.viewer._math import compose_poses
from better_robot.viewer.layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from better_robot.viewer.robot import robot_layers, robot_meshes
from better_robot.viewer.viewer import Viewer

from .fake_server import FakeServer


def _random_poses(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(*shape, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.concatenate([rng.normal(size=(*shape, 3)), quat], axis=-1).astype(np.float32)


def test_numpy_pose_composition_matches_the_torch_lie_layer() -> None:
    a, b = _random_poses((5, 3), 1), _random_poses((5, 3), 2)
    expected = compose(torch.from_numpy(a).double(), torch.from_numpy(b).double()).numpy()
    got = compose_poses(a, b)
    sign = np.sign(np.sum(got[..., 3:] * expected[..., 3:], axis=-1, keepdims=True))
    np.testing.assert_allclose(got[..., :3], expected[..., :3], atol=1e-5)
    np.testing.assert_allclose(got[..., 3:] * sign, expected[..., 3:], atol=1e-5)


def test_rigid_mesh_layer_moves_each_mesh_with_its_body() -> None:
    scene = FakeServer().scene
    body_poses = _random_poses((4, 3))
    origin = _random_poses((), 7)
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    specs = [
        MeshSpec(triangle, np.array([[0, 1, 2]]), body=2, origin=origin),
        MeshSpec(triangle, np.array([[0, 1, 2]]), body=0, origin=np.array([0, 0, 0, 0, 0, 0, 1.0])),
    ]
    layer = RigidMeshLayer("meshes", specs, body_poses)
    layer.build(scene, "/r/meshes")
    layer.set_step(3)
    expected = compose_poses(body_poses[3, 2], origin)
    node = scene.nodes["/r/meshes/mesh_0"]
    np.testing.assert_allclose(node.position, expected[:3], atol=1e-6)
    np.testing.assert_allclose(node.wxyz, expected[[6, 3, 4, 5]], atol=1e-6)
    np.testing.assert_allclose(scene.nodes["/r/meshes/mesh_1"].position, body_poses[3, 0, :3])

    deforming = MeshLayer("soft", np.zeros((6, 3, 3)), np.array([[0, 1, 2]]))
    deforming.build(scene, "/r/soft")
    deforming.set_step(5)
    assert deforming.nsteps == 6 and ("/r/soft/mesh", "vertices") in scene.log
    SpheresLayer("s", np.zeros((2, 4, 3)), np.ones(4)).build(scene, "/r/s")
    assert scene.nodes["/r/s/spheres"].props["batched_scales"].shape == (4,)


@pytest.mark.skipif(pytest.importorskip("robot_descriptions") is None, reason="needs robot_descriptions")
def test_panda_layers_follow_forward_kinematics() -> None:
    from robot_descriptions import panda_description  # noqa: PLC0415 - optional test dependency

    pytest.importorskip("trimesh")
    model = br.load(panda_description.URDF_PATH)
    q = torch.stack([model.random_configuration(torch.Generator().manual_seed(i)) for i in range(5)])
    tau = torch.randn(5, model.nv)
    layers = {layer.name: layer for layer in robot_layers(model, q, skeleton=True, tau=tau, frames=["body_panda_hand"])}
    assert set(layers) == {"robot", "robot skeleton", "robot frames"}
    assert len(robot_meshes(model)) == 11

    from better_robot.kinematics import forward_kinematics_state  # noqa: PLC0415

    state = forward_kinematics_state(model, q)
    hand = state.frame_pose[:, model.frame_id("body_panda_hand")].numpy()
    np.testing.assert_allclose(layers["robot frames"].poses[:, 0], hand, atol=1e-6)
    skeleton = layers["robot skeleton"]
    np.testing.assert_allclose(skeleton.joints, state.joint_pose[:, 1:, :3].numpy(), atol=1e-6)
    assert skeleton.parents[0] == -1 and list(skeleton.parents[1:]) == [p - 1 for p in model.parents[2:]]
    assert skeleton.colors.shape == (5, model.njoints - 1, 3)

    server = FakeServer()
    viewer = Viewer(server=server)
    for layer in layers.values():
        viewer.add(layer, group="panda")
    viewer.show(4)
    hand_mesh = next(i for i, spec in enumerate(layers["robot"].meshes) if spec.body == model.body_id("panda_hand"))
    node = server.scene.nodes[f"/panda/robot/mesh_{hand_mesh}"]
    expected = compose_poses(
        state.joint_pose[4, model.body_id("panda_hand")].numpy(), layers["robot"].meshes[hand_mesh].origin
    )
    np.testing.assert_allclose(node.position, expected[:3], atol=1e-5)


def test_robot_view_update_rewrites_layer_arrays_in_place() -> None:
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415 - optional test dependency

    from better_robot.viewer.robot import RobotView  # noqa: PLC0415

    model = br.load(panda_description.URDF_PATH)
    q0, q1 = (model.random_configuration(torch.Generator().manual_seed(i)) for i in range(2))
    view = RobotView(model, q0, visual=False, skeleton=True, frames=["body_panda_hand"])
    server = FakeServer()
    viewer = Viewer(server=server)
    for layer in view:
        viewer.add(layer)
    node = "/scene/robot skeleton/joints_cylinder"  # Panda's revolute joints
    before = server.scene.nodes[node].batched_positions.copy()
    view.update(q1)
    viewer.show()
    after = server.scene.nodes[node].batched_positions
    assert not np.allclose(before, after)
    fresh = robot_layers(model, q1, visual=False, skeleton=True)[0]
    expected = fresh.joints[0, fresh.groups["cylinder"]]
    np.testing.assert_allclose(after, expected, atol=1e-6)


def test_torque_colours_skip_joints_without_a_finite_limit() -> None:
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415 - optional test dependency

    from better_robot.viewer.robot import _torque_colors  # noqa: PLC0415
    from better_robot.viewer.theme import LIGHT  # noqa: PLC0415

    model = br.load(g1_description.URDF_PATH, free_flyer=True)
    tau = np.zeros(model.nv, dtype=np.float32)
    tau[:6] = 1e6  # free-flyer "torques": no effort limit, must stay in the base colour
    colors = _torque_colors(model, tau, LIGHT, nsteps=3)
    assert colors.shape == (3, model.njoints - 1, 3)
    np.testing.assert_allclose(colors[:, 0], np.tile(LIGHT.joint, (3, 1)), atol=1e-6)
    with pytest.raises(ValueError):
        _torque_colors(model, np.zeros((2, model.nv), dtype=np.float32), LIGHT, nsteps=3)


def test_skeleton_markers_follow_joint_kind_and_axis() -> None:
    """Revolute joints become cylinders along their axis, fixed joints draw nothing, free-flyers get a triad."""
    from better_robot.viewer import SkeletonLayer  # noqa: PLC0415

    poses = np.zeros((1, 3, 7), dtype=np.float32)
    poses[..., 6] = 1.0
    layer = SkeletonLayer(
        "s",
        poses,
        [-1, 0, 1],
        kinds=["free_flyer", "revolute_unaligned", "fixed"],
        axes=[[0, 0, 1], [1, 0, 0], [0, 0, 1]],
    )
    server = FakeServer()
    Viewer(server=server).add(layer)
    assert {"/scene/s/joints_axes", "/scene/s/joints_cylinder"} <= set(server.scene.nodes)
    assert "/scene/s/joints" not in server.scene.nodes
    wxyz = server.scene.nodes["/scene/s/joints_cylinder"].batched_wxyzs[0]
    np.testing.assert_allclose(np.abs(wxyz), [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0], atol=1e-6)  # +z -> +x
    assert len(layer.geometry(0)) == 3 + 1 + 1  # triad cylinders, revolute marker, bones
