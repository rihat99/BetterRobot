"""What the avoidance demos 05 to 09 share: their command line, the robots they load, the scripted run and its video.

Each demo keeps its own problem, scene and controls; run as ``python examples/viewer/<demo>.py``, it finds this
module beside it.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import g1_description, panda_description

import better_robot as br
from better_robot import Viewer
from better_robot.testing import FakeServer

URDFS = {"panda": panda_description.URDF_PATH, "g1": g1_description.URDF_PATH}
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]  # the Panda's start


def arguments(robots: Collection[str] = ()) -> argparse.Namespace:
    """The demo's command line: ``--check``, ``--record`` and, given ``robots``, ``--robot`` (the Panda by default)."""
    parser = argparse.ArgumentParser()
    if robots:
        parser.add_argument("--robot", choices=robots, default="panda")
    parser.add_argument("--check", action="store_true", help="run the scripted path on the stand-in server and exit")
    parser.add_argument("--record", type=Path, help="record the scripted path to this .mp4, .webm or .png")
    return parser.parse_args()


def viewer(*, check: bool, ground: bool = True) -> Viewer:
    """The demo's viewer at 25 fps: on the stand-in server for ``--check``, else serving http://127.0.0.1:8080."""
    return Viewer(fps=25.0, ground=ground, server=FakeServer() if check else None)


def load(robot: str) -> tuple[br.Model, torch.Tensor, br.Geometry]:
    """The model of ``robot`` (the G1 on a free base), its start ``q0 (1, nq)`` (Panda ready, G1 standing on z = 0)
    and its geometry."""
    g1 = robot == "g1"
    model = br.load(URDFS[robot], root_joint=br.JointFreeFlyer() if g1 else None, dtype=torch.float64)
    q0 = model.q_neutral.clone()[None]
    if g1:
        q0[0, 2] = 0.793
    else:
        q0[0, :7] = torch.tensor(READY)
    return model, q0, br.io.load_geometry(URDFS[robot], model)


def run(tick: Callable, form: str, x: dict, path: torch.Tensor) -> tuple[torch.Tensor, float]:
    """``q (T, nq)`` after one ``tick(form, path[k][None], x, warm)`` per row of ``path`` from ``x``, and the median
    tick in seconds."""
    warm, q, seconds = None, [], []
    for target in path:
        began = time.perf_counter()
        x, warm = tick(form, target[None], x, warm)
        seconds.append(time.perf_counter() - began)
        q.append(x["q"][0])
    return torch.stack(q), float(np.median(seconds))


def record(viewer: Viewer, path: Path) -> None:
    """Record the scripted path to ``path``, aimed at every tenth step: ``record``'s own fit reads every vertex."""
    camera = viewer.aim(steps=range(0, viewer.timeline.nsteps, 10))
    print(f"recorded {viewer.record(path, size=(960, 540), camera=camera)}")
