"""Interactive floating-base IK for the Unitree G1 humanoid.

Drag any hand/foot gizmo in the browser to re-solve whole-body IK.

Usage:
    uv run python examples/02_g1_ik.py [--no-viewer]
"""
import argparse
import time

import torch
import better_robot as br
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik
from robot_descriptions import g1_description

TARGET_FRAMES = [
    "body_left_rubber_hand",
    "body_right_rubber_hand",
    "body_left_ankle_roll_link",
    "body_right_ankle_roll_link",
]
BASE_HEIGHT = 0.78

COST = IKCostConfig(limit_weight=0.1, rest_weight=0.001)
OPT = OptimizerConfig(max_iter=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    model = br.load(g1_description.URDF_PATH, free_flyer=True)

    # Standing pose: pelvis at BASE_HEIGHT, identity orientation
    q0 = model.q_neutral.clone()
    q0[2] = BASE_HEIGHT
    q0[6] = 1.0
    q0[7:] = q0[7:].clamp(model.lower_pos_limit[7:], model.upper_pos_limit[7:])

    # FK to get reachable initial targets
    data = br.forward_kinematics(model, q0, compute_frames=True)
    available = [f for f in TARGET_FRAMES if f in model.frame_name_to_id]
    targets = {n: data.frame_pose_world[model.frame_id(n)].clone() for n in available}

    result = solve_ik(model, targets=targets, initial_q=q0,
                      cost_cfg=COST, optimizer_cfg=OPT)
    print(f"Initial IK: converged={result.converged}  iters={result.iters}")

    if args.no_viewer:
        return

    from better_robot.viewer import Visualizer

    viewer = Visualizer(model, port=8081)
    viewer.update(result.q)
    overlay = viewer.add_ik_targets(targets, scale=0.12)
    viewer.show(block=False)

    print("Viewer at http://localhost:8081  — drag gizmos, Ctrl-C to exit.")

    last = {k: v.clone() for k, v in overlay.live_targets().items()}
    try:
        while True:
            cur = overlay.live_targets()
            if any(not torch.allclose(cur[k], last[k], atol=1e-4)
                   for k in cur):
                r = solve_ik(model, targets=cur, initial_q=viewer.last_q,
                             cost_cfg=COST, optimizer_cfg=OPT)
                viewer.update(r.q)
                last = {k: v.clone() for k, v in cur.items()}
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nExit.")


if __name__ == "__main__":
    main()
