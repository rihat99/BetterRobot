"""Interactive whole-body IK for the Unitree G1 humanoid.

Controls both hands and both feet simultaneously with a floating base.

Usage:
    uv run python examples/02_g1_ik.py
    uv run python examples/02_g1_ik.py --collision           # capsules from URDF
    uv run python examples/02_g1_ik.py --profile             # profile one IK call then exit

Open http://localhost:8080 in your browser.
Drag the coloured transform handles to move the targets.
Click *Restart* to reset the robot and all targets to the initial standing pose.

Solver options (pass via IKConfig):
    jacobian="analytic"   — geometric Jacobian (faster, default here)
    jacobian="autodiff"   — torch.func.jacrev (works for any custom cost)
    solver="lm"           — Levenberg-Marquardt (default)
    solver="gn"           — Gauss-Newton (no damping, faster on easy problems)
    solver="adam"         — Adam gradient descent
    solver="lbfgs"        — L-BFGS with strong Wolfe line search
"""
import argparse
import time
import torch
import torch.profiler
import better_robot as br
from better_robot.algorithms.geometry.robot_collision import RobotCollision
from robot_descriptions.loaders.yourdfpy import load_robot_description


TARGET_LINKS = [
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

# G1 standing pose: pelvis ~0.78 m above ground, identity orientation, all joints at zero.
INITIAL_BASE_POSE = torch.tensor([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Profile one IK call and exit")
    parser.add_argument(
        "--collision", action="store_true",
        help="Enable self-collision avoidance",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device to use (default: cpu, e.g. cuda, cuda:0)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    urdf = load_robot_description("g1_description")
    model = br.load_urdf(urdf)
    print(f"Loaded G1: {model.links.num_links} links, {model.joints.num_actuated_joints} actuated joints")

    q_stand = torch.zeros(model.joints.num_actuated_joints, device=device)

    robot_coll = None
    if args.collision:
        # Use standing pose as reference for rest-pose filtering so that
        # structural capsule overlaps in the upright stance are excluded.
        # Collision construction always runs on CPU.
        robot_coll = RobotCollision.from_urdf(
            urdf, model,
            filter_q=q_stand.cpu(),
            filter_base_pose=INITIAL_BASE_POSE,
            filter_below_rest_dist=0.01,
        )
        n_pairs = len(robot_coll._active_pairs_i)
        print(f"Self-collision avoidance enabled (capsule mode, {n_pairs} active pairs). Capsules shown in red.")

    vis = br.Visualizer(urdf, model, floating_base=True)
    vis.add_grid()
    for link_name in TARGET_LINKS:
        vis.add_target(link_name, scale=0.2)
    vis.add_timing_display()
    vis.add_restart_button()

    config = br.IKConfig(jacobian="analytic")

    q = q_stand.clone()
    base_pose = INITIAL_BASE_POSE.clone().to(device)
    vis.reset_targets(model, q.cpu(), base_pose.cpu())
    vis.update(q.cpu(), base_pose=base_pose.cpu())

    if robot_coll is not None:
        vis.add_collision_geometry(robot_coll, q.cpu(), base_pose=base_pose.cpu())

    print("Drag transform handles to set IK targets. Press Ctrl+C to quit.")

    if args.profile:
        prof_activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            prof_activities.append(torch.profiler.ProfilerActivity.CUDA)
        targets = {k: v.to(device) for k, v in vis.get_targets().items()}
        with torch.profiler.profile(activities=prof_activities) as prof:
            data = br.solve_ik(
                model,
                targets=targets,
                config=config,
                initial_base_pose=base_pose,
                initial_q=q,
                max_iter=10,
                robot_coll=robot_coll,
            )
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
        return

    while True:
        if vis.restart_requested:
            q = q_stand.clone()
            base_pose = INITIAL_BASE_POSE.clone().to(device)
            vis.reset_targets(model, q.cpu(), base_pose.cpu())

        t0 = time.perf_counter()
        targets = {k: v.to(device) for k, v in vis.get_targets().items()}
        data = br.solve_ik(
            model,
            targets=targets,
            config=config,
            initial_base_pose=base_pose,
            initial_q=q,
            max_iter=10,
            robot_coll=robot_coll,
        )
        vis.set_timing((time.perf_counter() - t0) * 1000)
        q = data.q
        base_pose = data.base_pose
        vis.update(q.cpu(), base_pose=base_pose.cpu())
        vis.update_collision_geometry(q.cpu(), base_pose=base_pose.cpu())
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
