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
    np.testing.assert_allclose(layer.world_points(0), layer.joints[0])


def _every_axis_kind_model() -> br.Model:
    """A chain over rx, ry, rz, px, py, pz, an oblique revolute and prismatic, and a spherical joint (no axis),
    every joint placement rotated."""
    builder = br.ModelBuilder("axes")
    parent = builder.add_body("base")
    oblique = {"axis": torch.tensor([1.0, 2.0, 2.0]) / 3.0}
    joints = [
        ("add_revolute_x", {}), ("add_revolute_y", {}), ("add_revolute_z", {}), ("add_revolute", oblique),
        ("add_prismatic_x", {}), ("add_prismatic_y", {}), ("add_prismatic_z", {}), ("add_prismatic", oblique),
    ]  # fmt: skip
    for index, (method, kwargs) in enumerate(joints):
        child = builder.add_body(f"link{index}", mass=1.0)
        origin = torch.from_numpy(_random_poses((), seed=index))
        getattr(builder, method)(f"joint{index}", parent=parent, child=child, origin=origin, **kwargs)
        parent = child
    builder.add_spherical("ball", parent=parent, child=builder.add_body("tip", mass=1.0))
    return br.load(builder.finalize, dtype=torch.float64)


@pytest.mark.parametrize("robot", ["every_axis_kind", "g1"])
def test_skeleton_markers_lie_on_the_forward_kinematics_joint_axes(robot: str) -> None:
    """Each revolute cylinder and prismatic box the skeleton sends lies along its joint's world axis, measured by
    differentiating forward kinematics in that joint's coordinate (G1: rx / ry joints were drawn along z)."""
    from better_robot.lie.se3 import inverse  # noqa: PLC0415
    from better_robot.viewer import robot_skeleton  # noqa: PLC0415
    from better_robot.viewer.robot import skeleton_spec  # noqa: PLC0415

    if robot == "g1":
        pytest.importorskip("robot_descriptions")
        from robot_descriptions import g1_description  # noqa: PLC0415 - optional test dependency

        model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
    else:
        model = _every_axis_kind_model()
        kinds, axes = skeleton_spec(model)
        np.testing.assert_array_equal(axes[kinds.index("spherical")], [0.0, 0.0, 1.0])  # no axis: +z
    tangent = torch.randn(model.nv, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    q = model.integrate(model.q_neutral, tangent)
    joint_poses = br.forward_kinematics_state(model, q[None]).joint_pose.numpy().astype(np.float32)
    layer = robot_skeleton(model, joint_poses)
    server = FakeServer()
    Viewer(server=server).add(layer)

    checked = 0
    for marker in ("cylinder", "box"):
        if marker not in layer.groups:
            continue
        w, x, y, z = np.asarray(server.scene.nodes[f"/scene/skeleton/joints_{marker}"].batched_wxyzs, np.float64).T
        drawn = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)  # R @ +z
        for entry, joint in enumerate(layer.groups[marker] + 1):
            step = torch.zeros(model.nv, dtype=torch.float64)
            step[model.idx_vs[joint]] = 1e-6
            plus = br.forward_kinematics_state(model, model.integrate(q, step)).joint_pose[joint]
            minus = br.forward_kinematics_state(model, model.integrate(q, -step)).joint_pose[joint]
            change = compose(plus, inverse(minus)).numpy()  # world-frame increment: [translation, quaternion xyzw]
            true = change[3:6] if marker == "cylinder" else change[:3]
            true /= np.linalg.norm(true)
            angle = np.arctan2(np.linalg.norm(np.cross(drawn[entry], true)), drawn[entry] @ true)  # arccos: noisy at 0
            assert angle < 1e-4, f"{model.joint_names[joint]}: drawn {drawn[entry]}, FK axis {true}"
            checked += 1
    assert checked == (8 if robot == "every_axis_kind" else 29)


def test_robot_skeleton_hides_the_bones_given_a_zero_radius() -> None:
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415 - optional test dependency

    from better_robot.viewer import robot_skeleton  # noqa: PLC0415

    model = br.load(g1_description.URDF_PATH, free_flyer=True)
    poses = _random_poses((2, model.njoints))
    massless = (model.body_inertias[1:, 0] == 0).numpy()  # the G1's IMU / camera mounts
    assert massless.any()
    layer = robot_skeleton(model, poses, bone_radius=np.where(massless, 0.0, 0.01))
    server = FakeServer()
    Viewer(server=server).add(layer)
    scales = server.scene.nodes["/scene/skeleton/bones"].props["batched_scales"]
    np.testing.assert_allclose(scales[:, 0], np.where(massless, 0.0, 0.01)[layer.parents >= 0])
    assert robot_skeleton(model, poses).bone_radius == pytest.approx(0.012)  # unchanged default: 0.6 * radius


def test_torque_spheres_grow_and_redden_up_to_the_limit_and_mark_joints_past_it() -> None:
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415 - optional test dependency

    from better_robot.viewer import LIGHT, torque_spheres  # noqa: PLC0415
    from better_robot.viewer._math import rgb_uint8  # noqa: PLC0415

    model = br.load(g1_description.URDF_PATH, free_flyer=True)
    limits = model.values.effort_limit.numpy()
    half, over, idle = 3, 4, 5  # revolute joints (layer entries 2, 3, 4)
    tau = np.zeros((2, model.nv), dtype=np.float32)
    tau[:, :6] = 1e6  # free-flyer "torques": no effort limit, so no sphere
    tau[:, model.idx_vs[half]] = 0.5 * limits[model.idx_vs[half]]
    tau[1, model.idx_vs[over]] = -2.0 * limits[model.idx_vs[over]]
    poses = _random_poses((2, model.njoints))
    layer = torque_spheres(model, poses, tau=torch.from_numpy(tau), radius=(0.01, 0.05), over_limit=(1.0, 1.0, 0.0))
    np.testing.assert_allclose(layer.joints, poses[:, 1:, :3])
    assert (layer.parents == -1).all()  # spheres only
    np.testing.assert_allclose(layer.radius[:, [0, 1]], 0.0)  # the free flyer and a fixed joint draw nothing
    np.testing.assert_allclose(layer.radius[:, [half - 1, idle - 1]], [[0.03, 0.01], [0.03, 0.01]], atol=1e-6)
    np.testing.assert_allclose(layer.radius[:, over - 1], [0.01, 0.05], atol=1e-6)
    middle = (np.asarray(LIGHT.joint) + np.asarray(LIGHT.force)) / 2
    expected = rgb_uint8(np.stack([middle, LIGHT.joint, LIGHT.joint]))
    np.testing.assert_array_equal(layer.colors[0, [half - 1, idle - 1, over - 1]], expected)
    np.testing.assert_array_equal(layer.colors[1, over - 1], rgb_uint8((1.0, 1.0, 0.0)))
    one_frame = torque_spheres(model, poses, tau=tau[0])  # numpy (nv,): broadcast over the clip
    np.testing.assert_array_equal(one_frame.radius[1], torque_spheres(model, poses, tau=tau).radius[0])
