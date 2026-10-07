"""Interactive IK for Franka Panda.

Drag the SE(3) gizmo in the browser to move the end-effector.

Usage:
    uv run python examples/01_basic_ik.py [--no-viewer]
"""

import argparse
import math

import torch
import better_robot as br
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik
from robot_descriptions import panda_description

# Bent-elbow "ready" pose — away from singularities.
PANDA_READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]

EE_FRAME = "body_panda_hand"
COST = IKCostConfig(limit_weight=0.1, rest_weight=0.001)
OPT = OptimizerConfig(max_iter=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    model = br.load(panda_description.URDF_PATH)
    q0 = torch.tensor(PANDA_READY).clamp(model.lower_pos_limit, model.upper_pos_limit)

    # FK to get a reachable initial target
    data = br.forward_kinematics(model, q0, compute_frames=True)
    T_target = data.frame_pose_world[model.frame_id(EE_FRAME)].clone()

    result = solve_ik(model, targets={EE_FRAME: T_target}, initial_q=q0, cost_cfg=COST, optimizer_cfg=OPT)
    print(
        f"Initial IK: converged={result.converged}  pos_err="
        f"{(result.fk().frame_pose_world[model.frame_id(EE_FRAME)][:3] - T_target[:3]).norm():.4f} m"
    )

    if args.no_viewer:
        return

    from better_robot.viewer import GroundLayer, RobotView, Viewer  # noqa: PLC0415 - optional viewer extra
    from better_robot.viewer.tools import PoseGizmo  # noqa: PLC0415

    viewer = Viewer(port=8080)
    viewer.add(GroundLayer())
    robot = RobotView(model, result.q, skeleton=True)
    for layer in robot:
        viewer.add(layer, group="robot")
    live = {"q": result.q, "targets": {EE_FRAME: T_target.clone()}}

    def follow(name: str, pose) -> None:  # noqa: ANN001 - numpy pose (7,) from the gizmo
        live["targets"][name] = torch.as_tensor(pose, dtype=live["q"].dtype)
        r = solve_ik(model, targets=live["targets"], initial_q=live["q"], cost_cfg=COST, optimizer_cfg=OPT)
        live["q"] = r.q
        robot.update(r.q)
        viewer.show()

    gizmos = [
        PoseGizmo(viewer, name, pose.numpy(), scale=0.15, on_change=lambda p, name=name: follow(name, p))
        for name, pose in live["targets"].items()
    ]
    print(f"Viewer at {viewer.url} — drag the {len(gizmos)} gizmo(s), Ctrl-C to exit.")
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
