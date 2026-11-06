"""Shared fixtures and adapters for Pinocchio parity tests.

The oracles `pinocchio` and `scipy` come from the `test-oracles` dependency group; scipy converts rotations so the
oracle side never runs through `better_robot.lie` (principle 6.3).
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pinocchio as pin
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot

import better_robot as br
from tests.support.robots import g1_urdf, panda_urdf


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
    quat_xyzw = Rot.from_matrix(R).as_quat()
    out = np.concatenate([t, quat_xyzw])
    return torch.from_numpy(out).double()


def rot_matrix_from_pose(pose: torch.Tensor) -> np.ndarray:
    """Extract rotation matrix (3,3) from BetterRobot pose."""
    q_xyzw = pose[3:].detach().cpu().double().numpy()
    return Rot.from_quat(q_xyzw).as_matrix()


@pytest.fixture(scope="module")
def panda_both():
    """Load Panda URDF in both BetterRobot and Pinocchio.

    Returns:
        br_model, pin_model, pin_data, frame_map
        where frame_map[br_frame_name] = pin_frame_id
    """
    from better_robot.io import build_model, parse_urdf

    # Pinocchio's default URDF loader leaves mimic coordinates independent.
    # Make the BR side explicitly unconstrained too, preserving this suite as
    # a full-space recursion oracle. Dedicated mimic tests assert G-projected
    # kinematics and dynamics instead of claiming default-loader parity.
    ir = parse_urdf(panda_urdf())
    ir.joints = [
        dataclasses.replace(
            joint,
            mimic_source=None,
            mimic_multiplier=1.0,
            mimic_offset=0.0,
        )
        for joint in ir.joints
    ]
    br_model = build_model(ir, dtype=torch.float64)
    pin_model = pin.buildModelFromUrdf(str(panda_urdf()))
    pin_data = pin_model.createData()

    # A body's frame carries the body's name, which is the URDF link name Pinocchio knows.
    frame_map = {name: pin_model.getFrameId(name) for name in br_model.frame_names if pin_model.existFrame(name)}

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


def sample_panda_v(n: int = 16, seed: int = 1) -> torch.Tensor:
    """Random fp64 nonzero Panda velocities (every coordinate nonzero).

    Companion to :func:`sample_panda_q` for Jacobian time-variation parity; a
    zero velocity coordinate would silently green a wrong ``J̇`` term.
    """
    rng = torch.Generator().manual_seed(seed)
    vs = torch.empty(n, 9, dtype=torch.float64)
    for i in range(n):
        row = torch.rand(9, generator=rng, dtype=torch.float64) * 2.0 - 1.0
        # Push every coordinate away from zero, keeping the sign.
        vs[i] = torch.sign(row) * (row.abs() + 0.3)
    return vs


@pytest.fixture(scope="module")
def g1_both():
    """G1 with a free-flyer root in both libraries (nq != nv), plus the gather indices from Pinocchio's layout.

    Pinocchio orders the joints depth-first; BetterRobot keeps the URDF's order. ``q_pin[perm_q]`` and
    ``v_pin[perm_v]`` are the BetterRobot vectors, and ``tau_pin[perm_v]`` lines Pinocchio's torques up with
    BetterRobot's; the base's coordinates come first in both.
    """

    br_m = br.load(g1_urdf(), root_joint=br.model.JointFreeFlyer(), dtype=torch.float64)
    pin_m = pin.buildModelFromUrdf(str(g1_urdf()), pin.JointModelFreeFlyer())
    pin_d = pin_m.createData()
    perm_q, perm_v = br_m.q_permutation([pin_m.names[i] for i in range(1, pin_m.njoints)])
    return br_m, pin_m, pin_d, perm_q, perm_v


def _build_spherical_chain():
    """2-body chain: spherical joint + revolute RZ. Same in BR and Pinocchio."""
    from better_robot.io.build_model import build_model
    from better_robot.io.builder import ModelBuilder

    mass1, com1, I1 = 1.5, torch.tensor([0.0, 0.0, -0.2]), torch.diag(torch.tensor([0.04, 0.05, 0.01]))
    mass2, com2, I2 = 0.8, torch.tensor([0.0, 0.0, -0.15]), torch.diag(torch.tensor([0.02, 0.02, 0.005]))
    rz_offset = torch.tensor([0.0, 0.0, -0.4])

    b = ModelBuilder(name="sph_chain")
    base = b.add_body("base", mass=0.0)
    link1 = b.add_body("link1", mass=mass1, com=com1, inertia=I1)
    link2 = b.add_body("link2", mass=mass2, com=com2, inertia=I2)
    IDENT = torch.tensor([0.0, 0, 0, 0, 0, 0, 1.0])
    b.add_spherical("j_sph", parent=base, child=link1, origin=IDENT)
    b.add_revolute(
        "j_rz", axis=(0.0, 0.0, 1.0),
        parent=link1,
        child=link2,
        origin=torch.cat([rz_offset, torch.tensor([0.0, 0, 0, 1.0])]),
    )
    ir = b.finalize()
    br_m = build_model(ir).to(dtype=torch.float64)

    pin_m = pin.Model()
    j_sph_id = pin_m.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), "j_sph")
    pin_m.appendBodyToJoint(
        j_sph_id,
        pin.Inertia(float(mass1), com1.numpy().astype(float), I1.numpy().astype(float)),
        pin.SE3.Identity(),
    )
    j_rz_id = pin_m.addJoint(j_sph_id, pin.JointModelRZ(), pin.SE3(np.eye(3), rz_offset.numpy().astype(float)), "j_rz")
    pin_m.appendBodyToJoint(
        j_rz_id,
        pin.Inertia(float(mass2), com2.numpy().astype(float), I2.numpy().astype(float)),
        pin.SE3.Identity(),
    )
    pin_d = pin_m.createData()
    return br_m, pin_m, pin_d
