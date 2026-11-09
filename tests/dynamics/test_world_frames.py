"""The world-frame core's matrix copies of the frame changes agree with `spatial`'s public ones (principle 1.9).

Oracle: `spatial.transform_motion`, `spatial.transform_force` and `spatial.transform_inertia` with quaternion poses
(`dynamics/_world.py` keeps matrix forms for speed: `WorldKinematics.to_joint_frames`, `to_world_force` and
`world_inertias`). Each joint's pose is taken about the state's reference ``origin`` as the copies use it.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot import kinematics, spatial
from better_robot.dynamics._world import world_kinematics
from better_robot.lie import se3, so3
from tests.support.robots import panda_urdf

TOLERANCE = 1e-12


@pytest.mark.parametrize("seed", [0, 1])
def test_matrix_frame_changes_match_spatial(seed: int) -> None:
    model = br.load(panda_urdf(), dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    q = torch.stack([model.random_configuration(generator=generator) for _ in range(3)])
    kin = world_kinematics(model, kinematics.forward_kinematics(model, q))
    poses = torch.cat((kin.translations, so3.from_matrix(kin.rotations)), dim=-1)  # (N, njoints, 7) about origin
    twists = torch.randn(*poses.shape[:-1], 6, generator=generator, dtype=torch.float64)
    wrenches = torch.randn(*poses.shape[:-1], 6, generator=generator, dtype=torch.float64)

    motion_error = float(
        (kin.to_joint_frames(twists) - spatial.transform_motion(se3.inverse(poses), twists)).abs().max()
    )
    force_error = float((kin.to_world_force(wrenches) - spatial.transform_force(poses, wrenches)).abs().max())
    oracle_inertias = spatial.inertia_matrix(spatial.transform_inertia(poses, model.body_inertias))
    inertia_error = float((kin.world_inertias() - oracle_inertias).abs().max())
    assert motion_error <= TOLERANCE, "twist to joint frames vs spatial.transform_motion"
    assert force_error <= TOLERANCE, "wrench to world vs spatial.transform_force"
    assert inertia_error <= TOLERANCE, "world inertias vs spatial.transform_inertia"
