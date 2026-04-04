import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
from better_robot import Robot, solve_ik

@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return Robot.from_urdf(urdf)

def test_solve_ik_returns_correct_shape(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])  # some target pose
    result = solve_ik(panda, target_link="panda_hand", target_pose=target, max_iter=30)
    assert result.shape == (panda.joints.num_actuated_joints,)

def test_solve_ik_respects_joint_limits(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    result = solve_ik(panda, target_link="panda_hand", target_pose=target, max_iter=50)
    lo = panda.joints.lower_limits
    hi = panda.joints.upper_limits
    # Should be within limits (with some numerical tolerance)
    assert (result >= lo - 0.1).all(), f"Below lower limits: {result}"
    assert (result <= hi + 0.1).all(), f"Above upper limits: {result}"

def test_solve_ik_converges_to_reachable_pose(panda):
    """When target is reachable (FK of default config), IK should return near-zero pose error."""
    cfg_default = panda._default_cfg
    fk = panda.forward_kinematics(cfg_default)
    hand_idx = panda.get_link_index("panda_hand")
    target = fk[hand_idx].detach()

    result = solve_ik(
        panda, target_link="panda_hand", target_pose=target,
        initial_cfg=cfg_default.clone(), max_iter=5  # should need few iterations
    )

    # Check that resulting config gives close FK to target
    fk_result = panda.forward_kinematics(result)
    actual = fk_result[hand_idx]

    # Position error should be small
    pos_error = (actual[:3] - target[:3]).norm().item()
    assert pos_error < 0.05, f"Position error too large: {pos_error}"

def test_solve_ik_with_custom_weights(panda):
    target = torch.tensor([0.4, 0.0, 0.4, 0., 0., 0., 1.])
    result = solve_ik(
        panda, target_link="panda_hand", target_pose=target,
        weights={"pose": 2.0, "limits": 0.5, "rest": 0.001},
        max_iter=30
    )
    assert result.shape == (panda.joints.num_actuated_joints,)

def test_solve_ik_multi_shape(panda):
    """solve_ik_multi returns (num_actuated_joints,) tensor."""
    from better_robot.tasks._ik import solve_ik_multi
    cfg = panda._default_cfg
    fk = panda.forward_kinematics(cfg)
    targets = {
        "panda_link6": fk[panda.get_link_index("panda_link6")].detach(),
        "panda_hand":  fk[panda.get_link_index("panda_hand")].detach(),
    }
    result = solve_ik_multi(panda, targets=targets, max_iter=5)
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_converges(panda):
    """solve_ik_multi converges when targets are FK of default config."""
    from better_robot.tasks._ik import solve_ik_multi
    cfg = panda._default_cfg
    fk = panda.forward_kinematics(cfg)
    hand_idx = panda.get_link_index("panda_hand")
    link6_idx = panda.get_link_index("panda_link6")
    targets = {
        "panda_link6": fk[link6_idx].detach(),
        "panda_hand":  fk[hand_idx].detach(),
    }
    result = solve_ik_multi(panda, targets=targets, initial_cfg=cfg.clone(), max_iter=5)
    fk_result = panda.forward_kinematics(result)
    pos_err = (fk_result[hand_idx, :3] - fk[hand_idx, :3]).norm().item()
    assert pos_err < 0.05, f"Position error too large: {pos_err}"
