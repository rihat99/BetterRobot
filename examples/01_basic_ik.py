"""Interactive IK example for Franka Panda.

Usage:
    uv run python examples/01_basic_ik.py
Then open http://localhost:8080 in your browser.
Drag the red transform handle to move the target end-effector pose.
"""
import time
import torch
import numpy as np
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br


def wxyz_pos_to_se3(wxyz, pos) -> torch.Tensor:
    """Convert viser wxyz+pos to SE3 [tx, ty, tz, qx, qy, qz, qw]."""
    w, x, y, z = wxyz
    return torch.tensor([pos[0], pos[1], pos[2], x, y, z, w], dtype=torch.float32)


def main() -> None:
    # Load robot
    urdf = load_robot_description("panda_description")
    robot = br.Robot.from_urdf(urdf)
    n_joints = robot.joints.num_actuated_joints
    print(f"Loaded robot: {robot.links.num_links} links, {n_joints} actuated joints")

    # Start viser
    server = viser.ViserServer(port=8080)
    urdf_vis = viser.extras.ViserUrdf(server, urdf)

    # Add interactive transform handle for target EE pose
    target_handle = server.scene.add_transform_controls(
        "/target_ee",
        position=(0.3, 0.0, 0.6),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=0.15,
    )

    print("Open http://localhost:8080 in your browser")
    print("Drag the transform handle to set the IK target. Press Ctrl+C to quit.")

    cfg = robot._default_cfg.clone()

    # Use the robot's natural orientation at default config as the fixed target orientation.
    # This avoids a near-180° orientation error that causes LM to oscillate.
    fk0 = robot.forward_kinematics(cfg)
    hand_idx = robot.get_link_index("panda_hand")
    natural_qxyzw = fk0[hand_idx, 3:7].detach()  # [qx, qy, qz, qw]
    # Convert to viser wxyz (scalar-first) for the initial handle orientation
    natural_wxyz = (
        natural_qxyzw[3].item(),
        natural_qxyzw[0].item(),
        natural_qxyzw[1].item(),
        natural_qxyzw[2].item(),
    )
    target_handle.wxyz = natural_wxyz

    while True:
        # Read target from viser handle (position freely draggable; orientation tracks handle)
        target_pose = wxyz_pos_to_se3(target_handle.wxyz, target_handle.position)

        # Solve IK (warm-started from previous config for smoothness)
        cfg = br.solve_ik(
            robot=robot,
            target_link="panda_hand",
            target_pose=target_pose,
            initial_cfg=cfg,
            max_iter=20,
            weights={"pose": 1.0, "limits": 0.1, "rest": 0.001},
        )

        # Update robot visualization
        # ViserUrdf.update_cfg accepts a numpy array in URDF joint order,
        # or a dict {joint_name: angle}
        cfg_np = cfg.detach().cpu().numpy()

        # Build dict mapping joint name -> angle (only actuated joints, BFS order)
        actuated_joint_names = [
            name for name, jtype in zip(robot.joints.names, robot._fk_joint_types)
            if jtype in ("revolute", "continuous", "prismatic")
        ]
        cfg_dict = {name: float(val) for name, val in zip(actuated_joint_names, cfg_np)}
        urdf_vis.update_cfg(cfg_dict)

        time.sleep(1.0 / 30.0)  # ~30 Hz


if __name__ == "__main__":
    main()
