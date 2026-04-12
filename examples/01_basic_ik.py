"""Interactive IK for Franka Panda.

Drag the SE(3) gizmo in the browser to move the end-effector.

Usage:
    uv run python examples/01_basic_ik.py [--no-viewer]
"""
import argparse
import math
import time

import torch
import better_robot as br
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik
from robot_descriptions import panda_description

# Bent-elbow "ready" pose — away from singularities.
PANDA_READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4,
               0.0, math.pi / 2, math.pi / 4, 0.04, 0.04]

EE_FRAME = "body_panda_hand"
COST = IKCostConfig(limit_weight=0.1, rest_weight=0.001)
OPT = OptimizerConfig(max_iter=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    model = br.load(panda_description.URDF_PATH)
    q0 = torch.tensor(PANDA_READY).clamp(model.lower_pos_limit,
                                          model.upper_pos_limit)

    # FK to get a reachable initial target
    data = br.forward_kinematics(model, q0, compute_frames=True)
    T_target = data.oMf[model.frame_id(EE_FRAME)].clone()

    result = solve_ik(model, targets={EE_FRAME: T_target},
                      initial_q=q0, cost_cfg=COST, optimizer_cfg=OPT)
    print(f"Initial IK: converged={result.converged}  pos_err="
          f"{(result.fk().oMf[model.frame_id(EE_FRAME)][:3] - T_target[:3]).norm():.4f} m")

    if args.no_viewer:
        return

    from better_robot.viewer import Visualizer

    viewer = Visualizer(model, port=8080)
    viewer.update(result.q)
    overlay = viewer.add_ik_targets({EE_FRAME: T_target}, scale=0.15)
    viewer.show(block=False)

    print("Viewer at http://localhost:8080  — drag the gizmo, Ctrl-C to exit.")

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
