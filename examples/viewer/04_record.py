"""Record a Panda clip into a full-HD video through a headless Chromium, and save it as a ``.viser`` file.

    uv run python examples/viewer/04_record.py --out /tmp/panda.mp4           # renders on the CPU
    uv run python examples/viewer/04_record.py --out /tmp/panda.mp4 --gpu 0   # renders on GPU 0 (nvidia-smi index)
    uv run python examples/viewer/04_record.py --check                        # builds the clip on the stand-in server

Needs Chromium (``$BETTER_ROBOT_CHROME`` or on ``PATH``) and, for ``.mp4``, ffmpeg (``.webm`` needs none); see
``better_robot.viewer.record``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer
from better_robot.testing import FakeServer
from better_robot.viewer import RobotVisual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("panda.mp4"), help=".mp4, .webm or .png")
    parser.add_argument("--gpu", type=int, default=None, help="nvidia-smi index of the GPU to render on")
    parser.add_argument("--check", action="store_true", help="build the clip on the stand-in server, no browser")
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH)
    phase = torch.linspace(0.0, 2 * torch.pi, 150)[:, None]  # 5 s at 30 fps; every joint swings about its middle
    middle = (model.lower_position_limit + model.upper_position_limit) / 2
    half = (model.upper_position_limit - model.lower_position_limit) / 2
    q = middle + 0.5 * half * torch.sin(phase + torch.arange(model.nq) * 0.7)  # (T, nq)
    geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
    viewer = Viewer(fps=30.0, server=FakeServer() if args.check else None)
    viewer.add(RobotVisual(model, q, geometry=geometry, root=panda_description.URDF_PATH, frames=["panda_hand"]))
    if args.check:
        print(f"{viewer.timeline.nsteps} steps")
        return
    print(viewer.record(args.out, gpu=args.gpu))  # 1920 x 1080, every step, the camera aimed at the clip
    print(viewer.save(args.out.with_suffix(".viser")))  # the same clip for viser's static client
    viewer.server.stop()


if __name__ == "__main__":
    main()
