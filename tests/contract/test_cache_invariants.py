"""Cache-invariant tests for :class:`better_robot.data_model.Data`.

The contract is documented in ``docs/design/02_DATA_MODEL.md §3.1``:
reassigning :attr:`Data.q` / ``v`` / ``a`` invalidates strictly-higher
caches. In-place mutation of a tensor field is *not* detected — that is
called out as a known limitation.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.data_model import KinematicsLevel
from better_robot.exceptions import StaleCacheError


@pytest.fixture
def model() -> br.Model:
    builder = br.io.ModelBuilder("rrr")
    base = builder.add_body("base", mass=0.0)
    l1 = builder.add_body("l1", mass=1.0)
    l2 = builder.add_body("l2", mass=1.0)
    builder.add_revolute_z(
        "j1", parent=base, child=l1,
        origin=torch.tensor([0., 0., 0., 0., 0., 0., 1.]),
    )
    builder.add_revolute_z(
        "j2", parent=l1, child=l2,
        origin=torch.tensor([0.5, 0., 0., 0., 0., 0., 1.]),
    )
    return br.io.build_model(builder.finalize())


@pytest.fixture
def q(model: br.Model) -> torch.Tensor:
    return model.q_neutral.clone()


def test_fresh_data_is_at_level_none(model, q):
    data = model.create_data()
    assert data._kinematics_level == KinematicsLevel.NONE
    assert data.joint_pose_world is None


def test_compute_joint_jacobians_on_fresh_data_raises(model, q):
    data = model.create_data()
    data.q = q
    with pytest.raises(StaleCacheError):
        br.compute_joint_jacobians(model, data)


def test_forward_kinematics_advances_to_placements(model, q):
    data = br.forward_kinematics(model, q)
    assert data._kinematics_level == KinematicsLevel.PLACEMENTS
    assert data.joint_pose_world is not None


def test_q_reassignment_invalidates_to_none(model, q):
    data = br.forward_kinematics(model, q)
    assert data._kinematics_level == KinematicsLevel.PLACEMENTS
    new_q = q + 0.1
    data.q = new_q
    assert data.joint_pose_world is None
    assert data._kinematics_level == KinematicsLevel.NONE


def test_joint_pose_accessor_on_stale_data_raises(model, q):
    data = model.create_data()
    data.q = q
    with pytest.raises(StaleCacheError, match="forward_kinematics"):
        _ = data.joint_pose(0)


def test_invalidate_with_explicit_level(model, q):
    data = br.forward_kinematics(model, q)
    data.invalidate(KinematicsLevel.NONE)
    assert data._kinematics_level == KinematicsLevel.NONE
    assert data.joint_pose_world is None


def test_invalidate_default_clears_everything(model, q):
    data = br.forward_kinematics(model, q)
    data.invalidate()
    assert data._kinematics_level == KinematicsLevel.NONE


def test_inplace_q_mutation_is_not_detected(model, q):
    """Documented limitation: ``data.q[..., 0] += 1.0`` does not fire __setattr__.

    The cache stays at PLACEMENTS even though q (in-place) changed —
    this codifies the contract scope.
    """
    data = br.forward_kinematics(model, q)
    pre_pose = data.joint_pose_world.clone()
    data.q[..., 0] += 1.0
    # Cache invariants ARE NOT triggered by item assignment on the tensor.
    assert data._kinematics_level == KinematicsLevel.PLACEMENTS
    assert data.joint_pose_world is not None
    # The cache value is the OLD pose (proves we did not detect the mutation).
    assert torch.allclose(data.joint_pose_world, pre_pose)
