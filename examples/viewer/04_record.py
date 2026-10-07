"""Record a clip as a GIF (screenshots from a headless Chromium) and export it as a ``.viser`` file.

    uv run python examples/viewer/04_record.py --out /tmp/panda   # writes panda.gif and panda.viser

Needs Chromium (``$BETTER_ROBOT_CHROME`` or on ``PATH``); see ``better_robot.viewer.capture``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot.viewer import GroundLayer, Viewer, robot_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("panda"), help="output path without extension")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 360))
    args = parser.parse_args()

    model = br.load(panda_description.URDF_PATH)
    # A smooth sweep of every joint between its limits.
    phase = torch.linspace(0.0, 2 * torch.pi, args.frames)[:, None]
    mid = (model.lower_pos_limit + model.upper_pos_limit) / 2
    half = (model.upper_pos_limit - model.lower_pos_limit) / 2
    q = mid + 0.5 * half * torch.sin(phase + torch.arange(model.nq) * 0.7)

    viewer = Viewer(fps=30.0)
    viewer.add(GroundLayer(size=3.0))
    for layer in robot_layers(model, q, skeleton=True, frames=["body_panda_hand"]):
        viewer.add(layer, group="panda")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    gif = viewer.record(args.out.with_suffix(".gif"), size=tuple(args.size))
    print(f"{gif}: {args.frames} frames rendered in {time.perf_counter() - start:.1f} s")
    start = time.perf_counter()
    exported = viewer.save(args.out.with_suffix(".viser"))
    print(f"{exported}: {exported.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - start:.1f} s "
          "(open with viser's static client)")
    viewer.server.stop()


if __name__ == "__main__":
    main()
