"""``dynamics.contact_wrenches``: a world force at a frame enters inverse dynamics as minus its frame Jacobian's
transpose (oracle: virtual work with the linear rows of ``frame_jacobian``, float64, atol 1e-9)."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.dynamics import contact_wrenches, inverse_dynamics
from better_robot.exceptions import UnknownNameError
from better_robot.io import IRFrame, build_model, parse
from better_robot.kinematics import forward_kinematics, frame_jacobian, integrate
from tests.support.robots import g1_urdf

#: Two frames on the left ankle, both off its origin and one turned, and a turned frame off the right wrist.
FRAMES = {
    "heel": ("left_ankle_roll_link", [-0.05, 0.01, -0.03, 0.0, 0.0, 0.0, 1.0]),
    "toe": ("left_ankle_roll_link", [0.12, 0.0, -0.03, 0.0, 0.0, 0.6, 0.8]),
    "hand": ("right_wrist_yaw_link", [0.1, 0.0, 0.0, 0.6, 0.0, 0.0, 0.8]),
}


@pytest.fixture(scope="module")
def g1() -> br.Model:
    """The floating G1 with the frames of ``FRAMES``."""
    ir = parse(g1_urdf())
    ir.frames += [
        IRFrame(name=name, parent_body=body, placement=torch.tensor(pose)) for name, (body, pose) in FRAMES.items()
    ]
    return build_model(ir, root_joint=br.model.JointFreeFlyer(), dtype=torch.float64)


def test_a_frame_force_is_the_transpose_of_its_jacobian(g1: br.Model) -> None:
    """``inverse_dynamics(external_forces=contact_wrenches(f))`` equals ``inverse_dynamics() - J_linearᵀ f`` per frame;
    two frames on one joint add up, and a frame off its joint's origin carries the force's moment."""
    generator = torch.Generator().manual_seed(0)
    q = integrate(g1, g1.q_neutral, 0.3 * torch.randn(2, g1.nv, dtype=torch.float64, generator=generator))
    forces = torch.randn(2, len(FRAMES), 3, dtype=torch.float64, generator=generator)
    state = forward_kinematics(g1, q)
    zero = torch.zeros(2, g1.nv, dtype=torch.float64)

    free = inverse_dynamics(g1, state, v=zero, a=zero)
    wrenches = contact_wrenches(g1, state, forces=forces, frames=list(FRAMES))
    pushed = inverse_dynamics(g1, state, v=zero, a=zero, external_forces=wrenches)
    expected = free.clone()
    for index, name in enumerate(FRAMES):
        linear = frame_jacobian(g1, state, frame=name)[:, :3]  # (2, 3, nv), world axes at the frame's origin
        expected -= (linear.mT @ forces[:, index, :, None]).squeeze(-1)
    torch.testing.assert_close(pushed, expected, rtol=0.0, atol=1e-9)


def test_an_unknown_frame_raises(g1: br.Model) -> None:
    state = forward_kinematics(g1, g1.q_neutral)
    with pytest.raises(UnknownNameError, match="'nowhere'"):
        contact_wrenches(g1, state, forces=torch.zeros(1, 3, dtype=torch.float64), frames=["nowhere"])
