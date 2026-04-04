"""Interactive whole-body IK for the Unitree G1 humanoid.

Controls both hands and both feet simultaneously with a floating base.

Usage:
    uv run python examples/02_g1_ik.py
    uv run python examples/02_g1_ik.py --profile   # profile first 10 IK calls then exit
Then open http://localhost:8080 in your browser.
Drag the coloured transform handles to move the targets.
"""
import argparse
import time
import torch
import torch.profiler
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict


# End-effector handle names and their robot link names
TARGET_SPECS = [
    ("left_hand",  "left_rubber_hand"),
    ("right_hand", "right_rubber_hand"),
    ("left_foot",  "left_ankle_roll_link"),
    ("right_foot", "right_ankle_roll_link"),
]

# G1 standing height: pelvis ~0.78 m above ground
INITIAL_BASE_POSE = torch.tensor([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Profile first 10 IK calls and exit")
    args = parser.parse_args()

    urdf = load_robot_description("g1_description")
    robot = br.Robot.from_urdf(urdf)
    print(f"Loaded G1: {robot.links.num_links} links, {robot.joints.num_actuated_joints} actuated joints")

    server = viser.ViserServer(port=8080)
    server.scene.add_grid("/ground", width=4, height=4)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")

    # Warm start: zero joint config (natural standing) + initial base pose
    base_pose = INITIAL_BASE_POSE.clone()
    cfg = torch.zeros(robot.joints.num_actuated_joints)

    # Get natural FK orientations so handles start aligned with the robot
    fk0 = robot.forward_kinematics(cfg, base_pose=base_pose)

    target_controls: list[tuple[str, viser.TransformControlsHandle]] = []
    for name, link_name in TARGET_SPECS:
        link_idx = robot.get_link_index(link_name)
        handle = server.scene.add_transform_controls(
            f"/targets/{name}",
            position=fk0[link_idx, :3].detach().numpy(),
            wxyz=qxyzw_to_wxyz(fk0[link_idx, 3:7].detach()),
            scale=0.12,
        )
        target_controls.append((link_name, handle))

    with server.gui.add_folder("IK"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        reset_button = server.gui.add_button("Reset Targets")

    @reset_button.on_click
    def _(_) -> None:
        fk_curr = robot.forward_kinematics(cfg, base_pose=base_pose)
        for link_name, handle in target_controls:
            link_idx = robot.get_link_index(link_name)
            handle.position = fk_curr[link_idx, :3].detach().numpy()
            handle.wxyz = qxyzw_to_wxyz(fk_curr[link_idx, 3:7].detach())

    print("Open http://localhost:8080 in your browser")
    print("Drag transform handles to set IK targets. Press Ctrl+C to quit.")

    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for _ in range(1):
                targets = {
                    link_name: wxyz_pos_to_se3(handle.wxyz, handle.position)
                    for link_name, handle in target_controls
                }
                base_pose, cfg = br.solve_ik(
                    robot=robot,
                    targets=targets,
                    initial_base_pose=base_pose,
                    initial_cfg=cfg,
                    max_iter=20,
                )
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        return

    while True:
        start = time.time()

        targets = {
            link_name: wxyz_pos_to_se3(handle.wxyz, handle.position)
            for link_name, handle in target_controls
        }

        base_pose, cfg = br.solve_ik(
            robot=robot,
            targets=targets,
            initial_base_pose=base_pose,
            initial_cfg=cfg,
            max_iter=20,
        )

        elapsed_ms = (time.time() - start) * 1000
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_ms

        base_frame.position = base_pose[:3].detach().numpy()
        base_frame.wxyz = qxyzw_to_wxyz(base_pose[3:7].detach())
        urdf_vis.update_cfg(build_cfg_dict(robot, cfg))

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
