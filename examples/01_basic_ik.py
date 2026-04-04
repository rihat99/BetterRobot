"""Interactive IK example for Franka Panda.

Usage:
    uv run python examples/01_basic_ik.py
Then open http://localhost:8080 in your browser.
Drag the red transform handle to move the target end-effector pose.
"""
import time
import torch
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict


def main() -> None:
    urdf = load_robot_description("panda_description")
    robot = br.Robot.from_urdf(urdf)
    n_joints = robot.joints.num_actuated_joints
    print(f"Loaded robot: {robot.links.num_links} links, {n_joints} actuated joints")

    server = viser.ViserServer(port=8080)
    urdf_vis = viser.extras.ViserUrdf(server, urdf)

    target_handle = server.scene.add_transform_controls(
        "/target_ee",
        position=(0.3, 0.0, 0.6),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=0.15,
    )

    print("Open http://localhost:8080 in your browser")
    print("Drag the transform handle to set the IK target. Press Ctrl+C to quit.")

    cfg = robot._default_cfg.clone()

    # Use the robot's natural orientation at default config to avoid 180° singularity
    fk0 = robot.forward_kinematics(cfg)
    hand_idx = robot.get_link_index("panda_hand")
    target_handle.wxyz = qxyzw_to_wxyz(fk0[hand_idx, 3:7].detach())

    while True:
        target_pose = wxyz_pos_to_se3(target_handle.wxyz, target_handle.position)

        cfg = br.solve_ik(
            robot=robot,
            targets={"panda_hand": target_pose},
            cfg=br.IKConfig(rest_weight=0.001),
            initial_cfg=cfg,
            max_iter=20,
        )

        urdf_vis.update_cfg(build_cfg_dict(robot, cfg))
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
