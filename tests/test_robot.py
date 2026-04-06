"""Tests for RobotModel and forward kinematics."""

import pytest
import torch
import better_robot as br
from better_robot import RobotModel, load_urdf
from better_robot.math import adjoint_se3, se3_identity, se3_inverse
from robot_descriptions.loaders.yourdfpy import load_robot_description


def _load_panda():
    urdf = load_robot_description("panda_description")
    return load_urdf(urdf)


def test_robot_from_urdf():
    """load_urdf should succeed on panda_description."""
    model = _load_panda()
    # 7 revolute + 2 prismatic finger joints = 9 actuated
    assert model.joints.num_actuated_joints == 9
    assert model.links.num_links > 0


def test_forward_kinematics_shape():
    """FK should return (num_links, 7) for a single config."""
    model = _load_panda()
    n = model.joints.num_actuated_joints
    poses = model.forward_kinematics(torch.zeros(n))
    assert poses.shape == (model.links.num_links, 7)


def test_forward_kinematics_unit_quaternions():
    """FK output quaternions should be unit norm."""
    model = _load_panda()
    n = model.joints.num_actuated_joints
    poses = model.forward_kinematics(torch.zeros(n))
    quat_norms = torch.norm(poses[..., 3:7], dim=-1)
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5)


def test_forward_kinematics_batch():
    """FK should support batch dimensions."""
    model = _load_panda()
    n = model.joints.num_actuated_joints
    q = torch.zeros(3, 4, n)
    poses = model.forward_kinematics(q)
    assert poses.shape == (3, 4, model.links.num_links, 7)


def test_link_index():
    """link_index should return an integer for a known link."""
    model = _load_panda()
    idx = model.link_index("panda_hand")
    assert isinstance(idx, int)
    assert 0 <= idx < model.links.num_links


def test_link_index_invalid():
    """link_index should raise ValueError for unknown link."""
    model = _load_panda()
    with pytest.raises(ValueError, match="not found"):
        model.link_index("nonexistent_link")


def test_forward_kinematics_with_base_pose_shape():
    """FK with base_pose returns same shape as without."""
    model = _load_panda()
    n = model.joints.num_actuated_joints
    q = torch.zeros(n)
    base_pose = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # translate x=1
    poses_base = model.forward_kinematics(q, base_pose=base_pose)
    poses_orig = model.forward_kinematics(q)
    assert poses_base.shape == poses_orig.shape


def test_forward_kinematics_base_pose_translates_all_links():
    """A pure-translation base_pose shifts every link position by the same offset."""
    model = _load_panda()
    cfg = model._q_default
    offset = torch.tensor([1.0, 2.0, 3.0])
    base_pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])  # identity rotation
    poses_orig = model.forward_kinematics(cfg)
    poses_base = model.forward_kinematics(cfg, base_pose=base_pose)
    # Every link should be shifted by (1, 2, 3) in position (first 3 dims)
    # Orientation should be unchanged (base rotation is identity)
    assert torch.allclose(poses_base[..., :3], poses_orig[..., :3] + offset, atol=1e-4)
    assert torch.allclose(poses_base[..., 3:], poses_orig[..., 3:], atol=1e-4)


def test_forward_kinematics_none_base_unchanged():
    """base_pose=None is identical to omitting base_pose."""
    model = _load_panda()
    cfg = model._q_default
    assert torch.allclose(
        model.forward_kinematics(cfg, base_pose=None),
        model.forward_kinematics(cfg),
        atol=1e-6,
    )


@pytest.fixture(scope="session")
def panda():
    return _load_panda()


@pytest.fixture(scope="session")
def panda_model():
    return _load_panda()


def test_get_chain_panda_hand(panda):
    """Chain from root to panda_hand has exactly 7 actuated joints."""
    hand_idx = panda.link_index("panda_hand")
    chain = panda.get_chain(hand_idx)
    assert len(chain) == 7
    # All returned indices are actuated joints
    assert all(panda._fk_cfg_indices[j] >= 0 for j in chain)


def test_get_chain_root_link(panda):
    """Chain to root link is empty."""
    chain = panda.get_chain(panda._root_link_idx)
    assert chain == []


def test_get_chain_ordering(panda):
    """Chain is in root→EE topological order (parent joint before child joint)."""
    hand_idx = panda.link_index("panda_hand")
    chain = panda.get_chain(hand_idx)
    # cfg indices should be 0,1,2,... (BFS order for Panda arm)
    cfg_indices = [panda._fk_cfg_indices[j] for j in chain]
    assert cfg_indices == sorted(cfg_indices)


def test_adjoint_se3_identity():
    """Ad(identity) = I_6."""
    T = se3_identity()
    Ad = adjoint_se3(T)
    assert Ad.shape == (6, 6)
    assert torch.allclose(Ad, torch.eye(6), atol=1e-6)


def test_adjoint_se3_pure_translation():
    """Ad([1,0,0, 0,0,0,1]) has skew(p)@R in top-right block."""
    T = torch.tensor([1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0])  # x=1, identity rotation
    Ad = adjoint_se3(T)
    # Top-left: R = I
    assert torch.allclose(Ad[:3, :3], torch.eye(3), atol=1e-6)
    # Bottom-right: R = I
    assert torch.allclose(Ad[3:, 3:], torch.eye(3), atol=1e-6)
    # Bottom-left: zeros
    assert torch.allclose(Ad[3:, :3], torch.zeros(3, 3), atol=1e-6)
    # Top-right: skew([1,0,0]) @ I = [[0,0,0],[0,0,-1],[0,1,0]]
    expected_top_right = torch.tensor([
        [ 0.,  0.,  0.],
        [ 0.,  0., -1.],
        [ 0.,  1.,  0.],
    ])
    assert torch.allclose(Ad[:3, 3:], expected_top_right, atol=1e-6)


def test_adjoint_se3_inverse_consistency():
    """Ad(T) @ Ad(T^{-1}) = I_6 for an arbitrary non-trivial pose."""
    # 90-degree rotation around Z + translation
    T = torch.tensor([0.5, -0.3, 0.1,  0.0, 0.0, 0.7071068, 0.7071068])  # ~90° around Z
    T_inv = se3_inverse(T)
    Ad = adjoint_se3(T)
    Ad_inv = adjoint_se3(T_inv)
    product = Ad @ Ad_inv
    assert torch.allclose(product, torch.eye(6), atol=1e-5), \
        f"Ad(T) @ Ad(T^-1) != I: max_err={( product - torch.eye(6)).abs().max():.6f}"


def test_robot_model_is_immutable(panda_model):
    with pytest.raises(AttributeError, match="immutable"):
        panda_model.joints = None


# --- Phase 21: RobotData tests ---

def test_create_data_default_q(panda_model):
    data = panda_model.create_data()
    assert isinstance(data, br.RobotData)
    assert data.q.shape == panda_model._q_default.shape
    assert torch.allclose(data.q, panda_model._q_default)


def test_create_data_custom_q(panda_model):
    custom_q = torch.zeros(panda_model._q_default.shape)
    data = panda_model.create_data(q=custom_q)
    assert torch.allclose(data.q, custom_q)
    # ensure it's a clone, not the same tensor
    custom_q[0] = 999.0
    assert data.q[0] != 999.0


def test_create_data_model_id(panda_model):
    data = panda_model.create_data()
    assert data._model_id == id(panda_model)


def test_robot_data_clone(panda_model):
    data = panda_model.create_data()
    data.fk_poses = torch.zeros(10, 7)  # fake cache
    cloned = data.clone()
    assert torch.allclose(cloned.q, data.q)
    # independent copy
    cloned.q[0] = 999.0
    assert data.q[0] != 999.0
    assert cloned.fk_poses is not None


def test_robot_data_invalidate_cache(panda_model):
    data = panda_model.create_data()
    data.fk_poses = torch.zeros(10, 7)
    data.invalidate_cache()
    assert data.fk_poses is None


def test_forward_kinematics_with_data(panda_model):
    data = panda_model.create_data()
    result = br.forward_kinematics(panda_model, data)
    assert data.fk_poses is not None
    assert data.fk_poses.shape[-1] == 7
    assert torch.allclose(result, data.fk_poses)
    # Cache hit: sentinel value should NOT be overwritten
    sentinel = torch.full_like(data.fk_poses, 42.0)
    data.fk_poses = sentinel
    result2 = br.forward_kinematics(panda_model, data)
    assert torch.allclose(result2, sentinel), "Cache hit should return existing fk_poses"


def test_forward_kinematics_method_with_data(panda_model):
    data = panda_model.create_data()
    result = panda_model.forward_kinematics(data)
    assert data.fk_poses is not None
    assert torch.allclose(result, data.fk_poses)


def test_forward_kinematics_tensor_backward_compat(panda_model):
    q = panda_model._q_default.clone()
    result = br.forward_kinematics(panda_model, q)
    assert result.shape[-1] == 7


def test_forward_kinematics_data_with_base_pose_raises(panda_model):
    data = panda_model.create_data()
    with pytest.raises(ValueError, match="base_pose must not be passed"):
        br.forward_kinematics(panda_model, data, base_pose=torch.zeros(7))


def test_forward_kinematics_data_wrong_model_raises(panda_model):
    data = panda_model.create_data()
    import better_robot as br2
    # Create data with wrong model_id
    from better_robot.models.data import RobotData
    bad_data = RobotData(q=data.q.clone(), _model_id=id(object()))  # wrong id
    with pytest.raises(ValueError, match="different RobotModel"):
        br.forward_kinematics(panda_model, bad_data)
