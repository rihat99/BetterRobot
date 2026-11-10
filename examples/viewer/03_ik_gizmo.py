"""Interactive IK: drag the gizmo, the Panda follows (LM on ``solve_ik``).

uv run python examples/viewer/03_ik_gizmo.py
uv run python examples/viewer/03_ik_gizmo.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer, forward_kinematics, solve_ik
from better_robot.testing import FakeServer
from better_robot.viewer import FramesLayer, GroundLayer, RigidMeshLayer, robot_meshes, robot_skeleton
from better_robot.viewer.tools import PoseGizmo

FRAME = "panda_hand"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="solve one moved target on the stand-in server and exit")
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
    q = model.q_neutral[None]
    hand = model.frame_id(FRAME)
    start = forward_kinematics(model, q)
    pose = start.frame_poses[0, hand]

    viewer = Viewer(server=FakeServer() if args.check else None)
    viewer.add([GroundLayer()])
    meshes = RigidMeshLayer(
        "panda",
        robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH)),
        body_poses=start.joint_poses.numpy(),
    )
    skeleton = robot_skeleton(model, start.joint_poses.numpy())
    frames = FramesLayer("hand", start.frame_poses[:, [hand]].numpy())
    viewer.add([meshes, skeleton, frames], group="panda")
    # PoseGizmo calls ``follow`` on its own worker thread with only the newest pose, one call at a
    # time, so ``state`` has a single writer. A short warm-started solve keeps up with the mouse.
    state = {"q": q}

    def follow(target: np.ndarray) -> None:
        result = solve_ik(model, {FRAME: torch.from_numpy(target).to(q)[None]}, q0=state["q"], iterations=15)
        state["q"] = result.x["q"]
        moved = forward_kinematics(model, state["q"])
        poses = moved.joint_poses.numpy()
        meshes.body_poses[:] = poses
        skeleton.joints[:] = poses[:, 1:, :3]
        skeleton.rotations[:] = poses[:, 1:, 3:]
        frames.poses[:] = moved.frame_poses[:, [hand]].numpy()
        viewer.show()

    gizmo = PoseGizmo(viewer, "hand target", pose=pose.numpy(), on_change=follow)
    if args.check:
        moved = pose.clone()
        moved[0] += 0.1
        follow(moved.numpy())
        reached = forward_kinematics(model, state["q"]).frame_poses[0, model.frame_id(FRAME)]
        print(f"target moved 10 cm, residual {float((reached[:3] - moved[:3]).norm()):.1e} m")
        return
    print(f"drag the gizmo (now at {gizmo.pose[:3]}) at {viewer.url}; Ctrl-C to exit")
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
