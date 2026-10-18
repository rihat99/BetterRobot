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
    fresh_server = FakeServer()
    Viewer(server=fresh_server).add(robot_layers(model, q1, visual=False, skeleton=True)[0])
    np.testing.assert_allclose(after, fresh_server.scene.nodes[node].batched_positions, atol=1e-6)


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


def _g1() -> br.Model:
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415 - optional test dependency

    return br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)


def _drawn_markers(layer, server: FakeServer, marker: str) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Centres ``(N, 3)`` and unit axes ``(N, 3)`` of the ``marker`` instances the skeleton sent."""
    node = server.scene.nodes[f"/scene/{layer.name}/joints_{marker}"]
    w, x, y, z = np.asarray(node.batched_wxyzs, np.float64).T
    axes = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)  # R @ +z
    return np.asarray(node.batched_positions, np.float64), axes


def _drawn_skeleton(model: br.Model, q: torch.Tensor, radius: float = 0.02, tau: np.ndarray | None = None) -> tuple:
    """The skeleton of ``model`` at ``q (nq,)`` sent to a fake server: poses, layer, server, cylinder centres, axes."""
    from better_robot.viewer import robot_skeleton  # noqa: PLC0415

    poses = br.forward_kinematics_state(model, q[None]).joint_pose.numpy().astype(np.float32)
    layer = robot_skeleton(model, poses, radius=radius, tau=tau)
    server = FakeServer()
    Viewer(server=server).add(layer)
    return (poses, layer, server, *_drawn_markers(layer, server, "cylinder"))


def test_spread_layout_slides_co_located_markers_along_their_own_axes() -> None:
    """G1: the drawn cylinder of a co-located joint stays on the joint's axis line, at the neutral and a random pose."""
    from better_robot.viewer import spread_offsets  # noqa: PLC0415

    model = _g1()
    offsets = spread_offsets(model, radius=0.02)
    assert np.count_nonzero(np.linalg.norm(offsets, axis=-1)) >= 6  # waist, shoulders, wrists, ankles slide
    tangent = torch.randn(model.nv, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    for q in (model.q_neutral, model.integrate(model.q_neutral, tangent)):
        poses, layer, _, centres, axes = _drawn_skeleton(model, q)
        origins = poses[0, 1:, :3][layer.groups["cylinder"]]
        off_axis = np.linalg.norm(np.cross(centres - origins, axes), axis=-1)
        assert off_axis.max() < 1e-5  # the marker slid along its own axis only


@pytest.mark.parametrize("load", [None, 1.0])
def test_spread_layout_draws_no_two_g1_cylinders_overlapping_and_joins_the_drawn_centres(load: float | None) -> None:
    """G1 at the neutral pose, plain and with every joint at its effort limit (every marker at its thickest): samples
    of each drawn cylinder's volume lie in no other cylinder, bones join the drawn centres, none has zero length, and
    the waist yaw is drawn off the root."""
    model = _g1()
    radius = 0.02
    limits = model.values.effort_limit.numpy()
    tau = None if load is None else load * np.where(np.isfinite(limits), limits, 0.0)  # (nv,)
    poses, layer, server, centres, axes = _drawn_skeleton(model, model.q_neutral, radius, tau=tau)
    scales = np.asarray(server.scene.nodes["/scene/skeleton/joints_cylinder"].props["batched_scales"], np.float64)
    rho, half = scales[:, 0], scales[:, 2] / 2  # (N,) each drawn cylinder's radius and half-length
    np.testing.assert_allclose(rho, radius * 0.8 * (1.0 if load is None else 1.8), rtol=1e-6)
    side = np.cross(axes, [[0.0, 0.0, 1.0]])
    side[np.linalg.norm(side, axis=-1) < 1e-6] = (1.0, 0.0, 0.0)
    side /= np.linalg.norm(side, axis=-1, keepdims=True)
    other = np.cross(axes, side)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 2 * np.pi, 16, endpoint=False),
                                np.linspace(-1, 1, 9), indexing="ij"), -1).reshape(-1, 3)  # fmt: skip
    for i in range(len(centres)):
        across_i, along_i = grid[:, :1] * rho[i], grid[:, 2:] * half[i]
        samples = (centres[i] + across_i * np.cos(grid[:, 1:2]) * side[i] + across_i * np.sin(grid[:, 1:2])
                   * other[i] + along_i * axes[i])  # fmt: skip
        for j in range(len(centres)):
            if i == j:
                continue
            along = (samples - centres[j]) @ axes[j]
            across = np.linalg.norm(samples - centres[j] - along[:, None] * axes[j], axis=-1)
            inside = (np.abs(along) < half[j] - 1e-4) & (across < rho[j] - 1e-4)
            names = [model.joint_names[k + 1] for k in layer.groups["cylinder"][[i, j]]]
            assert not inside.any(), f"{names[0]} overlaps {names[1]}"

    bones = server.scene.nodes["/scene/skeleton/bones"]
    drawn = np.zeros((model.njoints - 1, 3))
    drawn[layer.groups["cylinder"]] = centres
    child = np.nonzero(layer.parents >= 0)[0]
    marked = np.isin(child, layer.groups["cylinder"]) & np.isin(layer.parents[child], layer.groups["cylinder"])
    midpoints = (drawn[child] + drawn[layer.parents[child]]) / 2
    np.testing.assert_allclose(np.asarray(bones.batched_positions)[marked], midpoints[marked], atol=1e-5)
    scales = np.asarray(bones.batched_scales)
    assert ((scales[:, 2] > 1e-6) | (scales[:, 0] == 0.0)).all()  # a zero-length bone is hidden
    root, waist = (model.joint_names.index(name) - 1 for name in ("root_joint", "waist_yaw_joint"))
    assert np.linalg.norm(drawn[waist] - poses[0, root + 1, :3]) > radius  # the waist yaw leaves the root


def test_spread_layout_draws_left_and_right_joints_as_mirror_images() -> None:
    from better_robot.viewer import spread_offsets  # noqa: PLC0415

    model = _g1()
    offsets = spread_offsets(model, radius=0.02)
    _, layer, _, centres, _ = _drawn_skeleton(model, model.q_neutral)
    drawn = np.zeros((model.njoints - 1, 3))
    drawn[layer.groups["cylinder"]] = centres
    mirror = np.array([1.0, -1.0, 1.0])
    for name in (name for name in model.joint_names if name.startswith("left_")):
        k_left, k_right = (model.joint_names.index(n) - 1 for n in (name, name.replace("left_", "right_")))
        if k_left in layer.groups["cylinder"]:
            # The URDF's own left / right placements differ by up to 1e-5 m.
            np.testing.assert_allclose(drawn[k_right], drawn[k_left] * mirror, atol=1e-4, err_msg=name)
            np.testing.assert_allclose(np.linalg.norm(offsets[k_right]), np.linalg.norm(offsets[k_left]), atol=1e-7)


def test_spread_layout_slides_joints_with_mirrored_axis_signs_as_mirror_images() -> None:
    """Pitch joints at (0, +-0.1, -0.1), each with a co-located roll whose axis is +x on the left and -x on the right:
    the rolls tie (their origins lie in the root's plane normal to x) and both slide to +x, as mirror images."""
    from better_robot.viewer import spread_offsets  # noqa: PLC0415
    from better_robot.viewer._math import rotate  # noqa: PLC0415

    builder = br.ModelBuilder("mirror")
    for name in ("pelvis", "left_upper", "left_lower", "right_upper", "right_lower"):
        builder.add_body(name, mass=1.0)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        origin = torch.tensor([0.0, 0.1 * sign, -0.1, 0.0, 0.0, 0.0, 1.0])
        builder.add_revolute_y(f"{side}_pitch", parent="pelvis", child=f"{side}_upper", origin=origin,
                               effort_limit=10.0)  # fmt: skip
        builder.add_revolute(f"{side}_roll", parent=f"{side}_upper", child=f"{side}_lower",
                             axis=torch.tensor([sign, 0.0, 0.0]), effort_limit=10.0)  # fmt: skip
    model = br.load(builder.finalize, free_flyer=True, dtype=torch.float64)
    poses = br.forward_kinematics_state(model, model.q_neutral).joint_pose[1:].numpy()
    drawn = dict(zip(model.joint_names[1:], poses[:, :3] + rotate(poses[:, 3:], spread_offsets(model))))
    assert drawn["left_roll"][0] > 0.02  # slid off its pitch joint, to +x
    np.testing.assert_allclose(drawn["right_roll"], drawn["left_roll"] * [1.0, -1.0, 1.0], atol=1e-6)


def test_robot_skeleton_shows_the_joint_load_in_colour_and_thickness_at_a_fixed_instance_count() -> None:
    from better_robot.viewer import LIGHT, robot_skeleton  # noqa: PLC0415
    from better_robot.viewer._math import colormap, rgb_uint8  # noqa: PLC0415

    model = _g1()
    limits = model.values.effort_limit.numpy()
    half, over, idle = 3, 4, 5  # revolute joints (layer entries 2, 3, 4)
    tau = np.zeros((2, model.nv))
    tau[:, :6] = 1e6  # free-flyer "torques": no effort limit, plain look
    tau[:, model.idx_vs[half]] = 0.5 * limits[model.idx_vs[half]]
    tau[1, model.idx_vs[over]] = -2.0 * limits[model.idx_vs[over]]
    poses = _random_poses((2, model.njoints))
    entries = [half - 1, idle - 1, over - 1]

    layer = robot_skeleton(model, poses, tau=tau)
    middle = colormap(0.5, *LIGHT.torque_ramp)
    expected = rgb_uint8(np.stack([middle, LIGHT.torque_ramp[0], LIGHT.torque_ramp[0]]))
    np.testing.assert_array_equal(layer.colors[0, entries], expected)
    np.testing.assert_array_equal(layer.colors[1, over - 1], rgb_uint8(LIGHT.over_limit))
    np.testing.assert_array_equal(layer.colors[:, 0], rgb_uint8(np.tile(LIGHT.joint, (2, 1))))  # the free flyer
    np.testing.assert_allclose(layer.thickness[0, entries], [1.3, 0.8, 0.8])
    np.testing.assert_allclose(layer.thickness[1, over - 1], 1.8)  # past the limit: the maximum
    np.testing.assert_allclose(layer.thickness[:, 0], 1.0)
    with pytest.raises(ValueError, match="frames"):
        robot_skeleton(model, poses, tau=np.zeros((3, model.nv)))

    # Per step only scales and colours change: the instance count of every node is fixed.
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(layer)
    counts = {path: len(node.batched_positions) for path, node in server.scene.nodes.items() if "joints" in path}
    viewer.show(1)
    assert counts == {
        path: len(node.batched_positions) for path, node in server.scene.nodes.items() if "joints" in path
    }
    cylinders = server.scene.nodes["/scene/skeleton/joints_cylinder"].batched_scales
    index = list(layer.groups["cylinder"]).index(over - 1)
    np.testing.assert_allclose(cylinders[index], np.array([0.8, 0.8, 2.5]) * 0.02 * [1.8, 1.8, 1.0], rtol=1e-6)


def test_robot_skeleton_end_points_add_a_bone_from_the_frames_joint_to_the_point() -> None:
    """An end point (a frame and a point in it) draws a bone from the drawn marker of the frame's parent joint to the
    point, as thick as the bone into that joint, with no marker and no load of its own."""
    from better_robot.lie.se3 import act  # noqa: PLC0415
    from better_robot.viewer import LIGHT, robot_skeleton  # noqa: PLC0415
    from better_robot.viewer._math import rgb_uint8  # noqa: PLC0415

    builder = br.ModelBuilder("arm")
    base, upper, hand = (builder.add_body(name, mass=1.0) for name in ("base", "upper", "hand"))
    origin = torch.from_numpy(_random_poses((), seed=1))
    builder.add_revolute_y("shoulder", parent=base, child=upper, effort_limit=10.0)
    builder.add_revolute_x("wrist", parent=upper, child=hand, origin=origin, effort_limit=10.0)
    builder.add_frame("palm", parent_body=hand, placement=torch.from_numpy(_random_poses((), seed=2)))
    model = br.load(builder.finalize, dtype=torch.float64)
    q = torch.stack([model.random_configuration(torch.Generator().manual_seed(i)) for i in range(2)])
    state = br.forward_kinematics_state(model, q)
    point = torch.tensor([0.05, -0.02, 0.1], dtype=torch.float64)
    tau = np.full((2, model.nv), 20.0)  # past the limits
    radii = np.arange(1, model.njoints) * 0.002  # entry j: the bone into joint j + 1

    layer = robot_skeleton(model, state.joint_pose.numpy().astype(np.float32), tau=tau, bone_radius=radii,
                           end_points={"palm": point.tolist()})  # fmt: skip
    wrist = model.joint_names.index("wrist") - 1
    assert layer.parents[-1] == wrist and len(layer.parents) == model.njoints
    expected = act(state.frame_pose[:, model.frame_id("palm")], point).numpy()  # (2, 3)
    np.testing.assert_allclose(layer.joints[:, -1], expected, atol=1e-5)
    np.testing.assert_array_equal(layer.colors[:, -1], rgb_uint8(np.tile(LIGHT.joint, (2, 1))))
    np.testing.assert_allclose(layer.thickness[:, -1], 1.0)

    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(layer)
    for step in range(2):
        viewer.show(step)
        cylinders = server.scene.nodes["/scene/skeleton/joints_cylinder"]
        assert len(cylinders.batched_positions) == 2  # the end point has no marker
        start = cylinders.batched_positions[list(layer.groups["cylinder"]).index(wrist)]
        bones = server.scene.nodes["/scene/skeleton/bones"]
        np.testing.assert_allclose(bones.batched_positions[-1], (start + expected[step]) / 2, atol=1e-5)
        length = np.linalg.norm(expected[step] - start)
        np.testing.assert_allclose(bones.batched_scales[-1], [radii[wrist], radii[wrist], length], rtol=1e-5)
