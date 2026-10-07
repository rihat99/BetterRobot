"""Show one robot pose: meshes, skeleton, a frame and the ground.

    uv run python examples/viewer/01_robot_pose.py            # open http://127.0.0.1:8080
    uv run python examples/viewer/01_robot_pose.py --check    # build the scene and exit
"""

from __future__ import annotations

import argparse

from robot_descriptions import panda_description

import better_robot as br
from better_robot.viewer import GroundLayer, Viewer, robot_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="build the scene, push one frame and exit")
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH)
    q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
    viewer = Viewer()
    viewer.add(GroundLayer())
    for layer in robot_layers(model, q, skeleton=True, frames=["body_panda_hand"]):
        viewer.add(layer, group="panda")
    viewer.show()
    print(f"{len(viewer.layers)} layers at {viewer.url}")
    if args.check:
        viewer.server.stop()
        return
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
