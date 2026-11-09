"""MJCF kinematics parity. Oracle: MuJoCo (``mujoco.MjModel`` and ``mj_kinematics``), record 0036.

Body poses, site frames, geometry, joint limits and the neutral configuration of a model loaded through
``parse_mjcf`` match MuJoCo's own kinematics on ``cases.xml`` (body ``pos``/``quat``, ``euler`` and ``axisangle``
orientations, degree ranges, ``limited="auto"``, a ``fromto`` capsule, a ball joint at an offset ``jnt_pos``, a
jointless body, sites (one on the world body), a hinge and a slide with a ``ref``, a scaled mesh MuJoCo recentres)
and ``floating.xml`` (a free root away from the origin, a ball joint, a hinge); ``equality.xml`` couples two joints
to a third by linear joint equalities. Configurations are drawn from a seeded generator whose seed is in the test
id.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

import better_robot as br

FILES = {"cases": Path(__file__).parent / "cases.xml", "floating": Path(__file__).parent / "floating.xml"}
EQUALITY = Path(__file__).parent / "equality.xml"
TOLERANCE = 1e-12  # float64 end to end; quaternion-to-matrix rounding is a few eps


def _random_qpos(compiled: mujoco.MjModel, seed: int) -> np.ndarray:
    """A MuJoCo configuration: scalar joints inside their range (else in [-1, 1]), unit quaternions elsewhere."""
    generator = np.random.default_rng(seed)
    qpos = np.zeros(compiled.nq)
    for joint in range(compiled.njnt):
        start = int(compiled.jnt_qposadr[joint])
        joint_type = compiled.jnt_type[joint]
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            low, high = compiled.jnt_range[joint] if compiled.jnt_limited[joint] else (-1.0, 1.0)
            qpos[start] = generator.uniform(low, high)
        else:
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos[start : start + 3] = generator.uniform(-0.5, 0.5, size=3)
                start += 3
            quaternion = generator.normal(size=4)
            qpos[start : start + 4] = quaternion / np.linalg.norm(quaternion)
    return qpos


def _q_from_mujoco(model: br.Model, compiled: mujoco.MjModel, qpos: np.ndarray) -> torch.Tensor:
    """The BetterRobot configuration of ``qpos``: joints matched by name, quaternions moved from w-first to w-last."""
    q = torch.zeros(model.nq, dtype=torch.float64)
    for joint_id, name in enumerate(model.joint_names):
        width = model.nqs[joint_id]
        if width == 0:
            continue
        start = int(compiled.jnt_qposadr[compiled.joint(name).id])
        values = qpos[start : start + width]
        if width in (4, 7):
            values = np.concatenate([values[:-4], values[-3:], values[-4:-3]])
        q[model.idx_qs[joint_id] : model.idx_qs[joint_id] + width] = torch.from_numpy(values)
    return q


def _joint_offsets(compiled: mujoco.MjModel) -> np.ndarray:
    """``(nbody, 3)`` the ``jnt_pos`` of each body's joint: BetterRobot's joint frame is the body frame moved there."""
    offsets = np.zeros((compiled.nbody, 3))
    for joint in range(compiled.njnt):
        if compiled.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            offsets[int(compiled.jnt_bodyid[joint])] = compiled.jnt_pos[joint]
    return offsets


def _assert_bodies_and_sites_match(
    model: br.Model, state: br.KinematicState, compiled: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """Every body's joint frame and every site of ``state`` where ``mj_kinematics`` put them in ``data``."""
    offsets = _joint_offsets(compiled)
    rotations = br.lie.so3.to_matrix(state.joint_poses[:, 3:]).numpy()
    for joint_id in range(1, model.njoints):
        body = compiled.body(model.body_names[joint_id]).id
        expected = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offsets[body]
        translation_error = float(np.abs(state.joint_translations[joint_id].numpy() - expected).max())
        rotation_error = float(np.abs(rotations[joint_id] - data.xmat[body].reshape(3, 3)).max())
        assert translation_error <= TOLERANCE, model.body_names[joint_id]
        assert rotation_error <= TOLERANCE, model.body_names[joint_id]
    for site in range(compiled.nsite):
        pose = state.frame_pose_of(compiled.site(site).name)
        site_translation_error = float(np.abs(pose[:3].numpy() - data.site_xpos[site]).max())
        site_rotation_error = float(
            np.abs(br.lie.so3.to_matrix(pose[3:]).numpy() - data.site_xmat[site].reshape(3, 3)).max()
        )
        assert site_translation_error <= TOLERANCE, compiled.site(site).name
        assert site_rotation_error <= TOLERANCE, compiled.site(site).name


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("file", sorted(FILES))
def test_body_and_site_poses_match_mujoco(file: str, seed: int) -> None:
    compiled = mujoco.MjModel.from_xml_path(str(FILES[file]))
    data = mujoco.MjData(compiled)
    data.qpos[:] = _random_qpos(compiled, seed)
    mujoco.mj_kinematics(compiled, data)
    model = br.load(FILES[file], dtype=torch.float64)
    state = br.forward_kinematics(model, _q_from_mujoco(model, compiled, data.qpos))
    _assert_bodies_and_sites_match(model, state, compiled, data)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_linear_joint_equality_is_a_mimic_joint(seed: int) -> None:
    """``equality.xml`` couples a hinge and a slide to a driver hinge, all with a ``ref``: only the driver is a
    coordinate, and the followers sit where MuJoCo's equality puts them (its residual is checked to be zero)."""
    compiled = mujoco.MjModel.from_xml_path(str(EQUALITY))
    data = mujoco.MjData(compiled)
    model = br.load(EQUALITY, dtype=torch.float64)
    assert [name for name, width in zip(model.joint_names, model.nqs, strict=True) if width] == ["driver"]
    driver = float(np.random.default_rng(seed).uniform(-1.0, 1.0))
    qpos = compiled.qpos0.copy()
    qpos[int(compiled.jnt_qposadr[compiled.joint("driver").id])] = driver
    for equality in range(compiled.neq):  # y - y0 = a0 + a1 (x - x0), MuJoCo's linear joint equality
        follower, source = int(compiled.eq_obj1id[equality]), int(compiled.eq_obj2id[equality])
        a0, a1 = compiled.eq_data[equality, :2]
        follower_zero = compiled.qpos0[int(compiled.jnt_qposadr[follower])]
        source_zero = compiled.qpos0[int(compiled.jnt_qposadr[source])]
        qpos[int(compiled.jnt_qposadr[follower])] = follower_zero + a0 + a1 * (driver - source_zero)
    data.qpos[:] = qpos
    mujoco.mj_forward(compiled, data)
    equality_rows = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert int(equality_rows.sum()) == 2
    residual = float(np.abs(data.efc_pos[: data.nefc][equality_rows]).max())
    assert residual <= TOLERANCE, "MuJoCo's equality residual at the coupled qpos"
    state = br.forward_kinematics(model, torch.tensor([driver], dtype=torch.float64))
    _assert_bodies_and_sites_match(model, state, compiled, data)


#: The ``fromto`` capsule of each file: its body and the two end points the file writes in that body's frame.
FROMTO = {
    "cases": ("upper", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]),
    "floating": ("arm", [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)]),
}


def _obj_vertices(path: Path) -> np.ndarray:
    """``(n, 3)`` the ``v`` lines of an OBJ file."""
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.array(rows, dtype=np.float64)


def _furthest_nearest(points: np.ndarray, others: np.ndarray) -> float:
    """The largest distance from a point of ``points (n, 3)`` to its nearest point of ``others (m, 3)``."""
    return float(np.linalg.norm(points[:, None, :] - others[None, :, :], axis=-1).min(axis=1).max())


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("file", sorted(FILES))
def test_geometry_matches_mujoco(file: str, seed: int) -> None:
    """Each geom's world pose and size, a mesh's vertices where MuJoCo puts them, a ``fromto`` capsule's end points."""
    compiled = mujoco.MjModel.from_xml_path(str(FILES[file]))
    data = mujoco.MjData(compiled)
    data.qpos[:] = _random_qpos(compiled, seed)
    mujoco.mj_kinematics(compiled, data)
    model = br.load(FILES[file], dtype=torch.float64)
    shapes = br.io.load_geometry(FILES[file], model).shapes
    state = br.forward_kinematics(model, _q_from_mujoco(model, compiled, data.qpos))
    geoms = [geom for geom in range(compiled.ngeom) if compiled.geom_bodyid[geom] != 0]
    assert len(shapes) == len(geoms)
    for geom, attached in zip(geoms, shapes, strict=True):
        shape, size = attached.shape, compiled.geom_size[geom]
        assert model.body_names[attached.body] == compiled.body(int(compiled.geom_bodyid[geom])).name
        world = br.lie.se3.compose(state.joint_poses[attached.body], torch.tensor(shape.placement, dtype=torch.float64))
        rotation = br.lie.so3.to_matrix(world[3:]).numpy()
        if isinstance(shape, br.model.Mesh):
            mesh = int(compiled.geom_dataid[geom])
            start, count = int(compiled.mesh_vertadr[mesh]), int(compiled.mesh_vertnum[mesh])
            recentred = compiled.mesh_vert[start : start + count].astype(np.float64)
            local = _obj_vertices(FILES[file].parent / shape.uri) * np.array(shape.scale)
            ours = world[:3].numpy() + local @ rotation.T
            theirs = data.geom_xpos[geom] + recentred @ data.geom_xmat[geom].reshape(3, 3).T
            mesh_error = max(_furthest_nearest(ours, theirs), _furthest_nearest(theirs, ours))
            # MuJoCo stores the recentred vertices in float32: rounding noise of that storage, 4 eps at their scale.
            mesh_tolerance = 4.0 * float(np.finfo(np.float32).eps) * float(np.abs(recentred).max())
            assert mesh_error <= mesh_tolerance, f"mesh vertices of geom {geom}"
            continue
        translation_error = float(np.abs(world[:3].numpy() - data.geom_xpos[geom]).max())
        rotation_error = float(np.abs(rotation - data.geom_xmat[geom].reshape(3, 3)).max())
        assert translation_error <= TOLERANCE, f"geom {geom} position"
        assert rotation_error <= TOLERANCE, f"geom {geom} orientation"
        if isinstance(shape, br.model.Sphere):
            sizes, expected = [shape.radius], [size[0]]
        elif isinstance(shape, br.model.Box):
            sizes, expected = list(shape.size), list(2.0 * size)
        else:
            sizes, expected = [shape.radius, shape.length], [size[0], 2.0 * size[1]]
        size_error = float(np.abs(np.array(sizes) - np.array(expected)).max())
        assert size_error <= TOLERANCE, f"geom {geom} size"
    body_name, ends = FROMTO[file]
    capsule = next(attached.shape for attached in shapes if model.body_names[attached.body] == body_name)
    world = br.lie.se3.compose(
        state.joint_poses[model.body_id(body_name)], torch.tensor(capsule.placement, dtype=torch.float64)
    )
    half = torch.tensor([0.0, 0.0, capsule.length / 2.0], dtype=torch.float64)
    ours = torch.stack([br.lie.se3.act(world, half), br.lie.se3.act(world, -half)]).numpy()
    body = compiled.body(body_name).id
    theirs = data.xpos[body] + np.array(ends) @ data.xmat[body].reshape(3, 3).T
    fromto_error = _furthest_nearest(ours, theirs)
    assert fromto_error <= TOLERANCE, f"fromto end points of the {body_name} capsule"


def _velocity_map(model: br.Model, compiled: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """``(nv_mujoco, nv)`` the map from a BetterRobot velocity to MuJoCo's ``qvel``: joints matched by name; a free
    joint's linear velocity is in the body frame here and in the world frame in MuJoCo."""
    velocity_map = np.zeros((compiled.nv, model.nv))
    for joint_id, name in enumerate(model.joint_names):
        width = model.nvs[joint_id]
        if width == 0:
            continue
        joint = compiled.joint(name).id
        rows = slice(int(compiled.jnt_dofadr[joint]), int(compiled.jnt_dofadr[joint]) + width)
        columns = slice(model.idx_vs[joint_id], model.idx_vs[joint_id] + width)
        block = np.eye(width)
        if compiled.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            block[:3, :3] = data.xmat[int(compiled.jnt_bodyid[joint])].reshape(3, 3)
        velocity_map[rows, columns] = block
    return velocity_map


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("file", sorted(FILES))
def test_mass_matrix_and_gravity_torque_match_mujoco(file: str, seed: int) -> None:
    """Inertias and armature (``cases.xml`` gives every joint 0.01) through the mass matrix (``mj_fullM``), gravity
    through the bias force at rest; the floating file compares its full matrix, base block included."""
    compiled = mujoco.MjModel.from_xml_path(str(FILES[file]))
    data = mujoco.MjData(compiled)
    data.qpos[:] = _random_qpos(compiled, seed)
    mujoco.mj_forward(compiled, data)
    mass_matrix = np.zeros((compiled.nv, compiled.nv))
    mujoco.mj_fullM(compiled, mass_matrix, data.qM)
    model = br.load(FILES[file], dtype=torch.float64)
    state = br.forward_kinematics(model, _q_from_mujoco(model, compiled, data.qpos))
    velocity_map = _velocity_map(model, compiled, data)
    rest = torch.zeros(model.nv, dtype=torch.float64)
    ours = br.mass_matrix(model, state).numpy()
    expected = velocity_map.T @ mass_matrix @ velocity_map
    mass_matrix_error = float(np.abs(ours - expected).max())
    assert mass_matrix_error <= TOLERANCE, "mass matrix vs mj_fullM"
    gravity_torque = br.inverse_dynamics(model, state, v=rest, a=rest).numpy()
    gravity_torque_error = float(np.abs(gravity_torque - velocity_map.T @ data.qfrc_bias).max())
    assert gravity_torque_error <= TOLERANCE, "gravity torque vs qfrc_bias"


@pytest.mark.parametrize("file", sorted(FILES))
def test_scalar_joint_limits_match_mujoco(file: str) -> None:
    """Hinge and slide ranges in the compiled units; ``limited="auto"`` without a range is unlimited."""
    compiled = mujoco.MjModel.from_xml_path(str(FILES[file]))
    model = br.load(FILES[file], dtype=torch.float64)
    checked = 0
    for joint_id, name in enumerate(model.joint_names):
        if model.nqs[joint_id] != 1:
            continue
        joint = compiled.joint(name).id
        start = model.idx_qs[joint_id]
        expected = compiled.jnt_range[joint] if compiled.jnt_limited[joint] else (-np.inf, np.inf)
        assert float(model.lower_position_limit[start]) == expected[0], name
        assert float(model.upper_position_limit[start]) == expected[1], name
        checked += 1
    assert checked == (5 if file == "cases" else 1)


@pytest.mark.parametrize("file", sorted(FILES))
def test_neutral_configuration_is_qpos0(file: str) -> None:
    """``q_neutral`` is MuJoCo's ``qpos0`` (a ``ref``, a free body's ``pos``/``quat``) clamped into the limits."""
    compiled = mujoco.MjModel.from_xml_path(str(FILES[file]))
    model = br.load(FILES[file], dtype=torch.float64)
    qpos0 = _q_from_mujoco(model, compiled, compiled.qpos0)
    expected = torch.clamp(qpos0, model.lower_position_limit, model.upper_position_limit)
    neutral_error = float((model.q_neutral - expected).abs().max())
    assert neutral_error <= TOLERANCE, "q_neutral vs qpos0"
    name, coordinate, value = {"cases": ("ref_hinge", 0, math.radians(30.0)), "floating": ("root", 2, 1.0)}[file]
    start_error = abs(float(model.q_neutral[model.idx_qs[model.joint_id(name)] + coordinate]) - value)
    assert start_error <= TOLERANCE, f"{name} starts at its qpos0, not at the kind's neutral"
