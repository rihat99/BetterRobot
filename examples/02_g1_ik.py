"""Interactive whole-body IK for the Unitree G1 humanoid.

Controls both hands and both feet simultaneously with a floating base.

Usage:
    uv run python examples/02_g1_ik.py
    uv run python examples/02_g1_ik.py --profile   # profile one IK call then exit

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
from robot_descriptions.loaders.yourdfpy import load_robot_description


TARGET_LINKS = [
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

# G1 standing pose: pelvis ~0.78 m above ground, identity orientation
INITIAL_BASE_POSE = torch.tensor([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Profile one IK call and exit")
    args = parser.parse_args()

    urdf = load_robot_description("g1_description")
    model = br.load_urdf(urdf)
    print(f"Loaded G1: {model.links.num_links} links, {model.joints.num_actuated_joints} actuated joints")

    vis = br.Visualizer(urdf, model, floating_base=True)
    vis.add_grid()
    for link_name in TARGET_LINKS:
        vis.add_target(link_name)
    vis.add_timing_display()
    vis.add_restart_button()

    config = br.IKConfig(jacobian="analytic")

    q = torch.zeros(model.joints.num_actuated_joints)
    base_pose = INITIAL_BASE_POSE.clone()
    vis.reset_targets(model, q, base_pose)
    vis.update(q, base_pose=base_pose)

    print("Drag transform handles to set IK targets. Press Ctrl+C to quit.")

    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            data = br.solve_ik(
                model,
                targets=vis.get_targets(),
                config=config,
                initial_base_pose=base_pose,
                initial_q=q,
                max_iter=20,
            )
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        return

    while True:
        if vis.restart_requested:
            q = torch.zeros(model.joints.num_actuated_joints)
            base_pose = INITIAL_BASE_POSE.clone()
            vis.reset_targets(model, q, base_pose)

        t0 = time.perf_counter()
        data = br.solve_ik(
            model,
            targets=vis.get_targets(),
            config=config,
            initial_base_pose=base_pose,
            initial_q=q,
            max_iter=20,
        )
        vis.set_timing((time.perf_counter() - t0) * 1000)
        q = data.q
        base_pose = data.base_pose
        vis.update(q, base_pose=base_pose)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
