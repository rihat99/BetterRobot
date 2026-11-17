"""Show the Panda in its ready pose: its meshes, the hand frame and the ground; the sidebar switches to the skeleton
styles.

uv run python examples/viewer/01_robot_pose.py            # open http://127.0.0.1:8080
uv run python examples/viewer/01_robot_pose.py --check    # build the scene on the stand-in server and exit
"""

from __future__ import annotations

import argparse
import math

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer
from better_robot.testing import FakeServer
from better_robot.viewer import RobotVisual

URDF = panda_description.URDF_PATH
READY = torch.tensor([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="build the scene on the stand-in server, push one frame and exit"
    )
    args = parser.parse_args()

    model = br.load(URDF)
    viewer = Viewer(server=FakeServer() if args.check else None)
    geometry = br.io.load_geometry(URDF, model)
    robot = viewer.add(RobotVisual(model, READY, geometry=geometry, root=URDF, frames=["panda_hand"]))
    print(f"{robot.name} as {robot.style} at {viewer.url}")
    if args.check:
        return
    viewer.serve()


if __name__ == "__main__":
    main()
