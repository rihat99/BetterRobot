"""Shared fixtures and adapters for Pinocchio parity tests.

All tests in this folder are skipped if pinocchio is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pin = pytest.importorskip("pinocchio", reason="dev-only dependency")


def pose_to_se3(pose: torch.Tensor) -> "pin.SE3":
    """BetterRobot ``[tx, ty, tz, qx, qy, qz, qw]`` → Pinocchio ``SE3``."""
    p = pose.detach().cpu().double().numpy()
    t = p[:3]
    q = p[3:]
    quat = pin.Quaternion(q[3], q[0], q[1], q[2])  # (w, x, y, z) constructor
    quat.normalize()
    return pin.SE3(quat.matrix(), t)


def se3_to_pose(se3: "pin.SE3") -> torch.Tensor:
    """Pinocchio ``SE3`` → BetterRobot ``[tx, ty, tz, qx, qy, qz, qw]`` (fp64)."""
    t = np.asarray(se3.translation)
    R = np.asarray(se3.rotation)
    from scipy.spatial.transform import Rotation as Rot
    quat_xyzw = Rot.from_matrix(R).as_quat()
    out = np.concatenate([t, quat_xyzw])
    return torch.from_numpy(out).double()


def rot_matrix_from_pose(pose: torch.Tensor) -> np.ndarray:
    """Extract rotation matrix (3,3) from BetterRobot pose."""
    from scipy.spatial.transform import Rotation as Rot
    q_xyzw = pose[3:].detach().cpu().double().numpy()
    return Rot.from_quat(q_xyzw).as_matrix()


@pytest.fixture(scope="module")
def panda_both():
    """Load Panda URDF in both BetterRobot and Pinocchio.

    Returns:
        br_model, pin_model, pin_data, frame_map
        where frame_map[br_frame_name] = pin_frame_id
    """
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    import better_robot as br

    br_model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    pin_model = pin.buildModelFromUrdf(panda_description.URDF_PATH)
    pin_data = pin_model.createData()

    # BetterRobot frames are prefixed with "body_". Strip prefix to get URDF link name.
    frame_map = {}
    for br_name in br_model.frame_names:
        pin_name = br_name[len("body_"):] if br_name.startswith("body_") else br_name
        if pin_model.existFrame(pin_name):
            frame_map[br_name] = pin_model.getFrameId(pin_name)

    return br_model, pin_model, pin_data, frame_map


def sample_panda_q(n: int = 16, seed: int = 0) -> torch.Tensor:
    """Random fp64 q configurations for Panda within a safe range."""
    rng = torch.Generator().manual_seed(seed)
    # Panda revolute joints have limits roughly ±[2.9, 1.8, 2.9, 0.07, 2.9, 3.75, 2.9];
    # fingers 0..0.04. Use a conservative ±1.0 to stay well inside limits.
    qs = torch.empty(n, 9, dtype=torch.float64)
    for i in range(n):
        qs[i] = torch.rand(9, generator=rng, dtype=torch.float64) * 2.0 - 1.0
    return qs
