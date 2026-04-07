"""Interactive IK example for Franka Panda.

Usage:
    uv run python examples/01_basic_ik.py
    uv run python examples/01_basic_ik.py --collision           # capsules from URDF

Open http://localhost:8080 in your browser.
Drag the transform handle to move the target end-effector pose.
Click *Restart* to reset the robot and target to the default configuration.

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
import better_robot as br
from better_robot.algorithms.geometry.robot_collision import RobotCollision
from robot_descriptions.loaders.yourdfpy import load_robot_description


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision", action="store_true",
        help="Enable self-collision avoidance (uses autodiff Jacobian)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device to use (default: cpu, e.g. cuda, cuda:0)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    urdf = load_robot_description("panda_description")
    model = br.load_urdf(urdf)
    print(f"Loaded Panda: {model.links.num_links} links, {model.joints.num_actuated_joints} actuated joints")

    robot_coll = None
    if args.collision:
        robot_coll = RobotCollision.from_urdf(urdf, model)
        n_pairs = len(robot_coll._active_pairs_i)
        print(f"Self-collision avoidance enabled (capsule mode, {n_pairs} active pairs). Capsules shown in red.")

    vis = br.Visualizer(urdf, model)
    vis.add_target("panda_hand", scale=0.15)
    vis.add_timing_display()
    vis.add_restart_button()

    # Use analytic Jacobian when collision is off (faster); autodiff is used automatically when collision is on.
    config = br.IKConfig(rest_weight=0.001, jacobian="analytic")

    q = model.q_default.clone().to(device)
    vis.reset_targets(model, q.cpu())
    vis.update(q.cpu())

    if robot_coll is not None:
        vis.add_collision_geometry(robot_coll, q.cpu())

    print("Drag the transform handle to set the IK target. Press Ctrl+C to quit.")

    while True:
        if vis.restart_requested:
            q = model.q_default.clone().to(device)
            vis.reset_targets(model, q.cpu())

        t0 = time.perf_counter()
        targets = {k: v.to(device) for k, v in vis.get_targets().items()}
        data = br.solve_ik(
            model,
            targets=targets,
            config=config,
            initial_q=q,
            max_iter=20,
            robot_coll=robot_coll,
        )
        vis.set_timing((time.perf_counter() - t0) * 1000)
        q = data.q
        vis.update(q.cpu())
        vis.update_collision_geometry(q.cpu())
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
