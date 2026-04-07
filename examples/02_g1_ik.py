"""Interactive whole-body IK for the Unitree G1 humanoid.

Controls both hands and both feet simultaneously with a floating base.

Usage:
    uv run python examples/02_g1_ik.py
    uv run python examples/02_g1_ik.py --collision           # capsules from URDF (default)
    uv run python examples/02_g1_ik.py --collision --spheres # legacy sphere decomposition
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

# Sphere decomposition for the G1 humanoid (legacy / --spheres flag).
# Covers collision-prone links: torso, upper arms, forearms, and legs.
# Centers are in link-local frame; radii in metres.
# Adjacent links are automatically excluded from collision checking by RobotCollision.
G1_SPHERES = {
    "pelvis":                   {"center": [0.0,  0.0,   0.0],  "radius": 0.12},
    "torso_link":               {"centers": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.15]], "radii": [0.12, 0.10]},
    "left_shoulder_pitch_link": {"center": [0.0,  0.0,   0.0],  "radius": 0.06},
    "right_shoulder_pitch_link":{"center": [0.0,  0.0,   0.0],  "radius": 0.06},
    "left_shoulder_roll_link":  {"center": [0.0,  0.0,   0.0],  "radius": 0.06},
    "right_shoulder_roll_link": {"center": [0.0,  0.0,   0.0],  "radius": 0.06},
    "left_elbow_link":          {"center": [0.0,  0.0,   0.0],  "radius": 0.05},
    "right_elbow_link":         {"center": [0.0,  0.0,   0.0],  "radius": 0.05},
    "left_wrist_roll_link":     {"center": [0.0,  0.0,   0.0],  "radius": 0.04},
    "right_wrist_roll_link":    {"center": [0.0,  0.0,   0.0],  "radius": 0.04},
    "left_hip_pitch_link":      {"center": [0.0,  0.0,   0.0],  "radius": 0.07},
    "right_hip_pitch_link":     {"center": [0.0,  0.0,   0.0],  "radius": 0.07},
    "left_knee_link":           {"center": [0.0,  0.0,   0.0],  "radius": 0.06},
    "right_knee_link":          {"center": [0.0,  0.0,   0.0],  "radius": 0.06},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Profile one IK call and exit")
    parser.add_argument(
        "--collision", action="store_true",
        help="Enable self-collision avoidance",
    )
    parser.add_argument(
        "--spheres", action="store_true",
        help="Use legacy sphere decomposition instead of URDF capsules (requires --collision)",
    )
    args = parser.parse_args()

    urdf = load_robot_description("g1_description")
    model = br.load_urdf(urdf)
    print(f"Loaded G1: {model.links.num_links} links, {model.joints.num_actuated_joints} actuated joints")

    q_stand = torch.zeros(model.joints.num_actuated_joints)

    robot_coll = None
    if args.collision:
        if args.spheres:
            robot_coll = RobotCollision.from_sphere_decomposition(G1_SPHERES, model)
            print("Self-collision avoidance enabled (sphere mode). Spheres shown in red.")
        else:
            # Use standing pose as reference for rest-pose filtering so that
            # structural capsule overlaps in the upright stance are excluded.
            robot_coll = RobotCollision.from_urdf(
                urdf, model,
                filter_q=q_stand,
                filter_base_pose=INITIAL_BASE_POSE,
                filter_below_rest_dist=0.01,
            )
            n_pairs = len(robot_coll._active_pairs_i)
            print(f"Self-collision avoidance enabled (capsule mode, {n_pairs} active pairs). Capsules shown in red.")

    vis = br.Visualizer(urdf, model, floating_base=True)
    vis.add_grid()
    for link_name in TARGET_LINKS:
        vis.add_target(link_name)
    vis.add_timing_display()
    vis.add_restart_button()

    config = br.IKConfig(jacobian="analytic")

    q = q_stand.clone()
    base_pose = INITIAL_BASE_POSE.clone()
    vis.reset_targets(model, q, base_pose)
    vis.update(q, base_pose=base_pose)

    if robot_coll is not None:
        vis.add_collision_geometry(robot_coll, q, base_pose=base_pose)

    print("Drag transform handles to set IK targets. Press Ctrl+C to quit.")

    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            data = br.solve_ik(
                model,
                targets=vis.get_targets(),
                config=config,
                initial_base_pose=base_pose,
                initial_q=q,
                max_iter=10,
                robot_coll=robot_coll,
            )
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        return

    while True:
        if vis.restart_requested:
            q = q_stand.clone()
            base_pose = INITIAL_BASE_POSE.clone()
            vis.reset_targets(model, q, base_pose)

        t0 = time.perf_counter()
        data = br.solve_ik(
            model,
            targets=vis.get_targets(),
            config=config,
            initial_base_pose=base_pose,
            initial_q=q,
            max_iter=10,
            robot_coll=robot_coll,
        )
        vis.set_timing((time.perf_counter() - t0) * 1000)
        q = data.q
        base_pose = data.base_pose
        vis.update(q, base_pose=base_pose)
        vis.update_collision_geometry(q, base_pose=base_pose)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
