"""MJCF kinematics parity. Oracle: MuJoCo (``mujoco.MjModel`` and ``mj_kinematics``), record 0036.

Body poses, site frames and joint limits of a model loaded through ``parse_mjcf`` match MuJoCo's own kinematics on
``cases.xml`` (body ``pos``/``quat``, ``euler`` and ``axisangle`` orientations, degree ranges, ``limited="auto"``,
a ``fromto`` capsule, a ball joint at an offset ``jnt_pos``, a jointless body, sites) and ``floating.xml`` (a free
root, a ball joint, a hinge). Configurations are drawn from a seeded generator whose seed is in the test id.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

import better_robot as br

FILES = {"cases": Path(__file__).parent / "cases.xml", "floating": Path(__file__).parent / "floating.xml"}
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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("file", sorted(FILES))
def test_body_and_site_poses_match_mujoco(file: str, seed: int) -> None:
    compiled = mujoco.MjModel.from_xml_path(str(FILES[file]))
    data = mujoco.MjData(compiled)
    data.qpos[:] = _random_qpos(compiled, seed)
    mujoco.mj_kinematics(compiled, data)
    model = br.load(FILES[file], dtype=torch.float64)
    state = br.forward_kinematics(model, _q_from_mujoco(model, compiled, data.qpos))
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
    assert checked == (3 if file == "cases" else 1)
