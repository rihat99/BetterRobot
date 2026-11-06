"""Show one robot pose: meshes, skeleton, a frame and the ground.

uv run python examples/viewer/01_robot_pose.py            # open http://127.0.0.1:8080
uv run python examples/viewer/01_robot_pose.py --check    # build the scene and exit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import panda_description

import better_robot as br
from better_robot.viewer import FramesLayer, GroundLayer, RigidMeshLayer, Viewer, robot_meshes, robot_skeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="build the scene, push one frame and exit")
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH)
    geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
    q = model.q_neutral.clamp(model.lower_position_limit, model.upper_position_limit)
    state = br.forward_kinematics(model, q[None])
    poses = state.joint_poses.numpy()
    viewer = Viewer()
    viewer.add(GroundLayer())
    meshes = robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH))
    viewer.add(RigidMeshLayer("panda", meshes, poses), group="panda")
    viewer.add(robot_skeleton(model, poses), group="panda")
    viewer.add(FramesLayer("hand", state.frame_poses[:, [model.frame_id("body_panda_hand")]].numpy()), group="panda")
    viewer.show()
    print(f"{len(viewer.layers)} layers at {viewer.url}")
    if args.check:
        viewer.server.stop()
        return
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
