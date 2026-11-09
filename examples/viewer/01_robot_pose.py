"""Show one robot pose: meshes, skeleton, a frame and the ground.

uv run python examples/viewer/01_robot_pose.py            # open http://127.0.0.1:8080
uv run python examples/viewer/01_robot_pose.py --check    # build the scene on the stand-in server and exit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import panda_description

import better_robot as br
from better_robot.testing import FakeServer
from better_robot import Viewer
from better_robot.viewer import FramesLayer, GroundLayer, RigidMeshLayer, robot_meshes, robot_skeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="build the scene on the stand-in server, push one frame and exit"
    )
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH)
    geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
    q = model.q_neutral
    state = br.forward_kinematics(model, q[None])
    poses = state.joint_poses.numpy()
    viewer = Viewer(server=FakeServer() if args.check else None)
    viewer.add([GroundLayer()])
    meshes = robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH))
    hand = FramesLayer("hand", state.frame_poses[:, [model.frame_id("panda_hand")]].numpy())
    viewer.add([RigidMeshLayer("panda", meshes, body_poses=poses), robot_skeleton(model, poses), hand], group="panda")
    viewer.show()
    print(f"{len(viewer.layers)} layers at {viewer.url}")
    if args.check:
        return
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
