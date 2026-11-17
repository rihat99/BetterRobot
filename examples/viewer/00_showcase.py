"""A first look at the viewer: pick a robot, show it as meshes or as the minimal or maximal skeleton, overlay its
collision shapes and joint frames, switch the theme, and play or pause a short loop of natural poses.

    uv run python examples/viewer/00_showcase.py                          # open http://127.0.0.1:8080
    uv run python examples/viewer/00_showcase.py --host 0.0.0.0 --port 8139
    uv run python examples/viewer/00_showcase.py --check                  # builds the page on the stand-in server
"""

from __future__ import annotations

import argparse
import math
import re

import torch
from robot_descriptions import g1_description, go2_description, panda_description

import better_robot as br
from better_robot import Viewer
from better_robot.testing import FakeServer
from better_robot.viewer import DARK, LIGHT, RobotVisual

STEPS = 240  # 8 s at 30 fps: one loop through the poses, each held for a moment
READY = {"panda_joint2": -0.8, "panda_joint4": -2.4, "panda_joint6": 1.6, "panda_joint7": 0.8, "finger_joint1": 0.04}
GRASP = {"panda_joint2": 0.3, "panda_joint4": -2.0, "panda_joint6": 2.3, "finger_joint1": 0.0}  # low, hand closed
ROBOTS = {  # name: (description, floating base, rest pose, the loop's poses over it as joint -> angle in rad or m)
    "G1": (g1_description, True, {"left_shoulder_roll": 0.15, "right_shoulder_roll": -0.15, "elbow": 1.2}, [
        {}, {"hip_pitch": -0.6, "knee": 1.3, "ankle_pitch": -0.7, "shoulder_pitch": -1.2},  # stand, squat
        {}, {"shoulder_pitch": -2.7, "left_shoulder_roll": 0.5, "right_shoulder_roll": -0.5, "elbow": 1.5},  # stretch
    ]),
    "Panda": (panda_description, False, READY, [{}, GRASP | {"panda_joint1": 0.9}, {}, GRASP | {"panda_joint1": -0.9}]),
    "Go2": (go2_description, True, {"thigh": 0.7, "calf": -1.4},  # stand, crouch, stand, give a paw
            [{}, {"thigh": 1.05, "calf": -2.1}, {}, {"FL_thigh": -0.5, "FL_calf": -1.7}]),
}  # fmt: skip


def motion(model: br.Model, rest: dict[str, float], poses: list[dict[str, float]]) -> torch.Tensor:
    """``q (STEPS, nq)`` that eases from each pose to the next and back to the first. A pose names joints over
    ``rest`` over the neutral configuration; a name without its side (``"knee"``) moves the joint of every side."""
    keys = model.q_neutral.repeat(len(poses), 1)
    for key, pose in zip(keys, poses, strict=True):
        for part, angle in (rest | pose).items():
            for joint, name in enumerate(model.joint_names):
                if re.fullmatch(f"(.*_)?{part}(_joint)?", name):
                    key[model.idx_qs[joint]] = angle
    time = torch.arange(STEPS) * len(poses) / STEPS
    index = time.long()
    ease = (1.0 - torch.cos(math.pi * (time - index)))[:, None] / 2.0
    return keys[index] + ease * (keys[(index + 1) % len(poses)] - keys[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves the page to other machines")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="build the page on the stand-in server and exit")
    args = parser.parse_args()

    viewer = Viewer(host=args.host, port=args.port, server=FakeServer() if args.check else None)
    robots: dict[str, RobotVisual] = {}

    def pick(name: str) -> None:
        for other, robot in robots.items():
            viewer.set_visible(robot, other == name)
        viewer.aim()  # a three-quarter view from the front

    viewer.sidebar.dropdown("Robot", "Model", options=list(ROBOTS), value="G1", on_change=pick)
    viewer.sidebar.dropdown(
        "Scene", "Theme", options=("light", "dark"), value="light",
        on_change=lambda theme: viewer.set_theme(DARK if theme == "dark" else LIGHT),
    )  # fmt: skip
    with torch.no_grad():
        for name, (description, floating, rest, poses) in ROBOTS.items():
            model = br.load(description.URDF_PATH, root_joint=br.JointFreeFlyer() if floating else None)
            geometry = br.io.load_geometry(description.URDF_PATH, model)
            q = motion(model, rest, poses)
            hue = LIGHT.hue(0)  # one robot on the page at a time: each in the first robot's colour
            visual = RobotVisual(model, q, name=name, geometry=geometry, root=description.URDF_PATH, hue=hue)
            if floating:  # lift the base, step by step, until the lowest point (a sole's collision sphere) is at z = 0
                q[:, 2] -= torch.tensor([visual.bounds(step)[0][:, 2].min() for step in range(STEPS)])
                visual.update(q)
            robots[name] = viewer.add(visual, tags="robot")  # one tag: the picker, not the Layers section, chooses
    pick("G1")
    print(f"{', '.join(robots)} at {viewer.url}")
    if not args.check:
        viewer.serve()


if __name__ == "__main__":
    main()
