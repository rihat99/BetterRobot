"""Record a clip as a GIF (screenshots from a headless Chromium) and export it as a ``.viser`` file.

    uv run python examples/viewer/04_record.py --out /tmp/panda   # writes panda.gif and panda.viser
    uv run python examples/viewer/04_record.py --check            # build the clip on the stand-in server, no files

Needs Chromium (``$BETTER_ROBOT_CHROME`` or on ``PATH``); see ``better_robot.viewer.capture``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot.testing import FakeServer
from better_robot.viewer import FramesLayer, GroundLayer, RigidMeshLayer, Viewer, robot_meshes, robot_skeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("panda"), help="output path without extension")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 360))
    parser.add_argument(
        "--check", action="store_true", help="build the clip on the stand-in server, push every frame, exit"
    )
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH)
    # A smooth sweep of every joint between its limits.
    phase = torch.linspace(0.0, 2 * torch.pi, args.frames)[:, None]
    mid = (model.lower_position_limit + model.upper_position_limit) / 2
    half = (model.upper_position_limit - model.lower_position_limit) / 2
    q = mid + 0.5 * half * torch.sin(phase + torch.arange(model.nq) * 0.7)

    geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
    state = br.forward_kinematics(model, q)
    poses = state.joint_poses.numpy()
    viewer = Viewer(fps=30.0, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer(size=3.0)])
    meshes = robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH))
    hand = FramesLayer("hand", state.frame_poses[:, [model.frame_id("panda_hand")]].numpy())
    viewer.add([RigidMeshLayer("panda", meshes, poses), robot_skeleton(model, poses), hand], group="panda")

    if args.check:  # recording needs a real server and Chromium
        for step in range(args.frames):
            viewer.show(step)
        print(f"{len(viewer.layers)} layers, {viewer.timeline.nsteps} frames")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    gif = viewer.record(args.out.with_suffix(".gif"), size=tuple(args.size))
    print(f"{gif}: {args.frames} frames rendered in {time.perf_counter() - start:.1f} s")
    start = time.perf_counter()
    exported = viewer.save(args.out.with_suffix(".viser"))
    print(
        f"{exported}: {exported.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - start:.1f} s "
        "(open with viser's static client)"
    )
    viewer.server.stop()


if __name__ == "__main__":
    main()
