"""Interactive whole-body IK for the Unitree G1 humanoid.

Controls both hands and both feet simultaneously with a floating base.

Usage:
    uv run python examples/02_g1_ik.py
Then open http://localhost:8080 in your browser.
Drag the coloured transform handles to move the targets.
"""
import time
import torch
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br


# End-effector link names and initial handle positions (world frame, metres)
TARGET_SPECS = [
    ("left_hand",  "left_rubber_hand",      ( 0.35,  0.28, 0.95)),
    ("right_hand", "right_rubber_hand",     ( 0.35, -0.28, 0.95)),
    ("left_foot",  "left_ankle_roll_link",  ( 0.00,  0.10, 0.02)),
    ("right_foot", "right_ankle_roll_link", ( 0.00, -0.10, 0.02)),
]

# G1 standing height: pelvis ~0.78 m above ground
INITIAL_BASE_POSE = torch.tensor([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])


def wxyz_pos_to_se3(wxyz, pos) -> torch.Tensor:
    """Convert viser wxyz+pos to SE3 [tx, ty, tz, qx, qy, qz, qw]."""
    w, x, y, z = wxyz
    return torch.tensor([pos[0], pos[1], pos[2], x, y, z, w], dtype=torch.float32)


def qxyzw_to_wxyz(q: torch.Tensor) -> tuple:
    """Convert [qx, qy, qz, qw] tensor to (w, x, y, z) viser tuple."""
    return (q[3].item(), q[0].item(), q[1].item(), q[2].item())


def main() -> None:
    urdf = load_robot_description("g1_description")
    robot = br.Robot.from_urdf(urdf)
    print(f"Loaded G1: {robot.links.num_links} links, {robot.joints.num_actuated_joints} actuated joints")

    server = viser.ViserServer(port=8080)
    server.scene.add_grid("/ground", width=4, height=4)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")

    # Warm start: default joint config + initial base pose
    base_pose = INITIAL_BASE_POSE.clone()
    cfg = robot._default_cfg.clone()

    # Get natural FK orientations at default config so handles start
    # aligned with the robot (avoids 180° orientation singularity)
    fk0 = robot.forward_kinematics(cfg, base_pose=base_pose)

    target_controls: list[tuple[str, viser.TransformControlsHandle]] = []
    for name, link_name, position in TARGET_SPECS:
        link_idx = robot.get_link_index(link_name)
        nat_wxyz = qxyzw_to_wxyz(fk0[link_idx, 3:7].detach())
        handle = server.scene.add_transform_controls(
            f"/targets/{name}",
            position=position,
            wxyz=nat_wxyz,
            scale=0.12,
        )
        target_controls.append((link_name, handle))

    # GUI controls
    with server.gui.add_folder("IK"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        reset_button = server.gui.add_button("Reset Targets")

    @reset_button.on_click
    def _(_) -> None:
        fk_curr = robot.forward_kinematics(cfg, base_pose=base_pose)
        for (link_name, handle), (_, _, position) in zip(target_controls, TARGET_SPECS):
            link_idx = robot.get_link_index(link_name)
            handle.position = position
            handle.wxyz = qxyzw_to_wxyz(fk_curr[link_idx, 3:7].detach())

    print("Open http://localhost:8080 in your browser")
    print("Drag transform handles to set IK targets. Press Ctrl+C to quit.")

    while True:
        start = time.time()

        targets = {
            link_name: wxyz_pos_to_se3(handle.wxyz, handle.position)
            for link_name, handle in target_controls
        }

        # Warm-started floating-base IK
        base_pose, cfg = br.solve_ik_floating_base(
            robot=robot,
            targets=targets,
            initial_base_pose=base_pose,
            initial_cfg=cfg,
            max_iter=20,
        )

        elapsed_ms = (time.time() - start) * 1000
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_ms

        # Update viser: base frame position + robot joint config
        base_frame.position = base_pose[:3].detach().numpy()
        base_frame.wxyz = qxyzw_to_wxyz(base_pose[3:7].detach())

        actuated_names = [
            name for name, jtype in zip(robot.joints.names, robot._fk_joint_types)
            if jtype in ("revolute", "continuous", "prismatic")
        ]
        cfg_np = cfg.detach().cpu().numpy()
        urdf_vis.update_cfg({name: float(v) for name, v in zip(actuated_names, cfg_np)})

        time.sleep(1.0 / 30.0)  # ~30 Hz


if __name__ == "__main__":
    main()
