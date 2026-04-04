"""Basic IK example (stub).

Usage:
    uv run python examples/01_basic_ik.py
"""

import better_robot as br
from robot_descriptions.loaders.yourdfpy import load_robot_description


def main() -> None:
    urdf = load_robot_description("panda_description")
    robot = br.Robot.from_urdf(urdf)
    print(f"Loaded robot with {robot.joints.num_actuated_joints} actuated joints")
    print("IK solve: not yet implemented")


if __name__ == "__main__":
    main()
