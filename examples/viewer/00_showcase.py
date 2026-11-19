"""A first look at the viewer: pick one of eight robots, show it as meshes or as the minimal, medium or maximal
skeleton, overlay its collision shapes and joint frames, switch the theme, and watch every joint swing in a calm loop.

    uv run python examples/viewer/00_showcase.py                          # open http://127.0.0.1:8080
    uv run python examples/viewer/00_showcase.py --host 0.0.0.0 --port 8139
    uv run python examples/viewer/00_showcase.py --check                  # builds the page on the stand-in server

The G1 is fetched into the robot_descriptions cache on first use, as in every example; the other robots are offered
when the cache has them, without a download. A robot is built the first time it is picked: the page starts with one.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
from robot_descriptions import g1_description

import better_robot as br
from better_robot import Viewer
from better_robot.testing import FakeServer
from better_robot.viewer import DARK, LIGHT, RobotVisual

STEPS = 240  # 8 s at 30 fps: one slow swing of every joint
CACHE = Path(g1_description.REPOSITORY_PATH).parent  # robot_descriptions' cache folder
ROBOTS = {  # name: (description file relative to the cache, floating base, ready pose: joint -> angle in rad or m)
    "G1": ("./unitree_ros/robots/g1_description/g1_29dof.urdf", True,
           {"left_shoulder_roll": 0.25, "right_shoulder_roll": -0.25, "elbow": 1.2}),
    "H1": ("./unitree_ros/robots/h1_description/urdf/h1.urdf", True,
           {"left_shoulder_roll": 0.2, "right_shoulder_roll": -0.2, "elbow": 1.0}),
    "Go2": ("./unitree_ros/robots/go2_description/urdf/go2_description.urdf", True, {"thigh": 0.7, "calf": -1.4}),
    "Spot": ("./mujoco_menagerie/boston_dynamics_spot/spot.xml", True, {"hy": 0.8, "kn": -1.6}),
    "Panda": ("./example-robot-data/robots/panda_description/urdf/panda.urdf", False,
              {"panda_joint2": -0.8, "panda_joint4": -2.4, "panda_joint6": 1.6, "panda_joint7": 0.8}),
    "UR5e": ("./mujoco_menagerie/universal_robots_ur5e/ur5e.xml", False,
             {"shoulder_lift": -1.57, "elbow": 1.57, "wrist_1": -1.57, "wrist_2": -1.57}),
    "Shadow hand": ("./mujoco_menagerie/shadow_hand/left_hand.xml", False, {".*FJ[123]": 0.3, "lh_THJ4": 0.5}),
    "Tiago": ("./example-robot-data/robots/tiago_description/robots/tiago.urdf", False,
              {"torso_lift": 0.15, "arm_2": -1.34, "arm_4": 1.94, "arm_5": -1.57, "arm_6": 1.37}),
}  # fmt: skip


def motion(model: br.Model, ready: dict[str, float]) -> torch.Tensor:
    """``q (STEPS, nq)``: each joint with limits leaves its ``ready`` angle (a name without its side, ``"knee"``, sets
    the joint of every side) for a quarter of the way to its farther limit and comes back, a little after its parent,
    so that a limb moves as one wave. It turns at most 0.3 rad and moves the far end of what it carries by at most
    10 cm: a hip less than a knee, a knee less than a finger. A joint and its twin on the other side move as mirror
    images, and no joint passes a limit."""
    q = model.q_neutral.clone()
    lag, cap, left = torch.zeros(model.nq), torch.zeros(model.nq), torch.zeros(model.nq)  # per coordinate
    rest = br.forward_kinematics(model, model.q_neutral).joint_poses[:, :3]  # (njoints, 3)
    depth, extent = [0] * model.njoints, [0.05] * model.njoints  # how far each joint's subtree reaches, in m
    for joint in reversed(range(1, model.njoints)):
        parent = model.parent_joint_ids[joint]
        extent[parent] = max(extent[parent], extent[joint] + float((rest[joint] - rest[parent]).norm()))
    for joint in range(1, model.njoints):
        depth[joint] = depth[model.parent_joint_ids[joint]] + 1
        coordinates = slice(model.idx_qs[joint], model.idx_qs[joint] + model.nqs[joint])
        lag[coordinates], cap[coordinates] = 0.6 * depth[joint], min(0.3, 0.1 / extent[joint])
        left[coordinates] = float(rest[joint, 1] > -1e-3)  # on the left half (y >= 0) or the middle
        for part, angle in ready.items():
            if re.fullmatch(f"(.*_)?{part}(_joint)?", model.joint_names[joint]):
                q[coordinates] = angle
    low, high = model.lower_position_limit, model.upper_position_limit
    room = (high - q) - (q - low)  # in the middle of its range, a joint on the right mirrors its twin on the left
    far = torch.where((room > 0) | (room == 0) & (left > 0), high, low)
    reach = torch.where(torch.isfinite(far), torch.clamp((far - q) / 4.0, -cap, cap), 0.0)  # a floating base: none
    time = torch.arange(STEPS)[:, None] * (2.0 * math.pi / STEPS)
    return q + reach * (1.0 - torch.cos(time - lag)) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves the page to other machines")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="build the page on the stand-in server and exit")
    args = parser.parse_args()

    viewer = Viewer(server=FakeServer()) if args.check else Viewer(host=args.host, port=args.port)
    models: dict[str, br.Model] = {}
    for name, (file, floating, _) in ROBOTS.items():
        if (CACHE / file).exists():  # in the cache: nothing is downloaded here
            models[name] = br.load(CACHE / file, root_joint=br.JointFreeFlyer() if floating else None)
    robots: dict[str, RobotVisual] = {}
    lifts: dict[str, float] = {}  # how far each fixed base is lifted to stand on the ground

    def pick(name: str) -> None:
        if name not in robots:  # built the first time it is picked, drawn as the robot on screen is
            (file, floating, ready), model = ROBOTS[name], models[name]
            q, geometry = motion(model, ready), br.io.load_geometry(CACHE / file, model)
            now = next((robot for robot in robots.values() if robot.visible), None)
            look = {"style": now.style, "collision": now.collision, "frames": now.frames} if now else {}
            robot = RobotVisual(model, q, name=name, geometry=geometry, root=CACHE / file, hue=LIGHT.hue(0), **look)
            lowest = torch.tensor([robot.bounds(step)[0][:, 2].min() for step in range(STEPS)])  # a sole, a palm
            if floating:  # the base stands on the ground at every step
                q[:, 2] -= lowest
                robot.update(q)
            lifts[name] = 0.0 if floating else -float(lowest.min())
            robots[name] = viewer.add(robot, tags="robot")  # one tag: the picker, not the Layers section, chooses
        for other, robot in robots.items():
            viewer.set_visible(robot, other == name)
        viewer.set_pose("robot", (0.0, 0.0, lifts[name]))
        viewer.aim()  # a three-quarter view from the front

    viewer.sidebar.dropdown("Robot", "Model", options=list(models), value="G1", on_change=pick)
    pick("G1")
    viewer.sidebar.dropdown(
        "Scene", "Theme", options=("light", "dark"), value="light",
        on_change=lambda theme: viewer.set_theme(DARK if theme == "dark" else LIGHT),
    )  # fmt: skip
    skipped = [name for name in ROBOTS if name not in models]
    print(f"{', '.join(models)} at {viewer.url}" + (f"; not in the cache: {', '.join(skipped)}" if skipped else ""))
    if not args.check:
        viewer.serve()


if __name__ == "__main__":
    main()
