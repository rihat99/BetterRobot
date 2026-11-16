"""Interactive IK: drag the gizmo, the Panda follows (LM on ``solve_ik``).

uv run python examples/viewer/03_ik_gizmo.py
uv run python examples/viewer/03_ik_gizmo.py --check
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer, forward_kinematics, solve_ik
from better_robot.testing import FakeServer
from better_robot.viewer import RobotVisual
from better_robot.viewer.tools import PoseGizmo

URDF = panda_description.URDF_PATH
FRAME = "panda_hand"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="solve one moved target on the stand-in server and exit")
    args = parser.parse_args()

    model = br.load(URDF, dtype=torch.float64)
    pose = forward_kinematics(model, model.q_neutral).frame_poses[model.frame_id(FRAME)]

    viewer = Viewer(server=FakeServer() if args.check else None)
    robot = viewer.add(RobotVisual(model, geometry=br.io.load_geometry(URDF, model), root=URDF, frames=[FRAME]))

    # PoseGizmo calls ``follow`` on its own worker thread with only the newest pose, one call at a
    # time, so ``robot.q`` has a single writer. A short warm-started solve keeps up with the mouse.
    def follow(target: np.ndarray) -> None:
        robot.update(solve_ik(model, {FRAME: torch.from_numpy(target)[None]}, q0=robot.q, iterations=15).x["q"])

    gizmo = PoseGizmo(viewer, "hand target", pose=pose.numpy(), on_change=follow)
    if args.check:
        moved = pose.clone()
        moved[0] += 0.1
        follow(moved.numpy())
        reached = forward_kinematics(model, robot.q).frame_poses[0, model.frame_id(FRAME)]
        print(f"target moved 10 cm, residual {float((reached[:3] - moved[:3]).norm()):.1e} m")
        return
    print(f"drag the gizmo (now at {gizmo.pose[:3]}) at {viewer.url}; Ctrl-C to exit")
    viewer.serve()


if __name__ == "__main__":
    main()
