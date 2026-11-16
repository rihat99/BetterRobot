"""Show one robot pose: its meshes, the hand frame and the ground; the sidebar switches to the skeleton styles.

uv run python examples/viewer/01_robot_pose.py            # open http://127.0.0.1:8080
uv run python examples/viewer/01_robot_pose.py --check    # build the scene on the stand-in server and exit
"""

from __future__ import annotations

import argparse

from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer
from better_robot.testing import FakeServer
from better_robot.viewer import RobotVisual

URDF = panda_description.URDF_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="build the scene on the stand-in server, push one frame and exit"
    )
    args = parser.parse_args()

    model = br.load(URDF)
    viewer = Viewer(server=FakeServer() if args.check else None)
    robot = viewer.add(RobotVisual(model, geometry=br.io.load_geometry(URDF, model), root=URDF, frames=["panda_hand"]))
    print(f"{robot.name} as {robot.style} at {viewer.url}")
    if args.check:
        return
    viewer.serve()


if __name__ == "__main__":
    main()
