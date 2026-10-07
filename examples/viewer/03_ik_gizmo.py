"""Interactive IK: drag the gizmo, the Panda follows (LM on ``terms.solve_ik``).

    uv run python examples/viewer/03_ik_gizmo.py
    uv run python examples/viewer/03_ik_gizmo.py --check
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import terms
from better_robot.kinematics import forward_kinematics_state
from better_robot.viewer import GroundLayer, RobotView, Viewer
from better_robot.viewer.tools import PoseGizmo

FRAME = "body_panda_hand"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="solve one moved target and exit")
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)[None]
    pose = forward_kinematics_state(model, q).frame_pose[0, model.frame_id(FRAME)]

    viewer = Viewer()
    viewer.add(GroundLayer())
    robot = RobotView(model, q, skeleton=True, frames=[FRAME])
    for layer in robot:
        viewer.add(layer, group="panda")
    # PoseGizmo calls ``follow`` on its own worker thread with only the newest pose, one call at a
    # time, so ``state`` has a single writer. A short warm-started solve keeps up with the mouse.
    state = {"q": q}

    def follow(target: np.ndarray) -> None:
        result = terms.solve_ik(model, {FRAME: torch.from_numpy(target).to(q)[None]}, q0=state["q"], iters=15)
        state["q"] = result.x["q"]
        robot.update(state["q"])
        viewer.show()

    gizmo = PoseGizmo(viewer, "hand target", pose.numpy(), on_change=follow)
    if args.check:
        moved = pose.clone()
        moved[0] += 0.1
        follow(moved.numpy())
        reached = forward_kinematics_state(model, state["q"]).frame_pose[0, model.frame_id(FRAME)]
        print(f"target moved 10 cm, residual {float((reached[:3] - moved[:3]).norm()):.1e} m")
        viewer.server.stop()
        return
    print(f"drag the gizmo (now at {gizmo.pose[:3]}) at {viewer.url}; Ctrl-C to exit")
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
