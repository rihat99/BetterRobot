"""Joint limits while tracking: turn the target about the hand's axis; ``optim.track`` keeps every joint in its range.

    uv run python examples/viewer/05_avoid_joint_limits.py                         # open http://127.0.0.1:8080
    uv run python examples/viewer/05_avoid_joint_limits.py --check                 # the scripted path, stand-in server
    uv run python examples/viewer/05_avoid_joint_limits.py --record limits.webm    # the scripted path, recorded

``terms.ik_problem`` passes the position limits as the problem's bounds; ``track`` makes them a box on every step,
which holds exactly at any step size since it is linear in the configuration. The GUI switches the limits between
that box ("rows"), a cost term (``terms.joint_limits``, which gives way a little) and off. A joint at its limit gets
an amber marker, a joint past it a red one. The scripted path turns the target 4.5 rad about the hand's axis and
back, 2.4 rad more than the last joint allows, and shows the Panda without limits (left) beside the Panda with the
box (right).
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer, forward_kinematics, lie, optim, terms
from better_robot.testing import FakeServer
from better_robot.viewer import RobotVisual, SpheresLayer
from better_robot.viewer.tools import PoseGizmo

URDF = panda_description.URDF_PATH
FRAME, READY = "panda_hand", [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
STEP, TURN, TICKS = 0.02, 4.5, 800  # rad per tick (the box needs no small step), rad, ticks of the scripted path
FORMS = ("rows", "cost", "off")  # the GUI's choices: the limits as the box, as a cost term, or no limits
AMBER, RED = (0.95, 0.65, 0.1), (0.9, 0.1, 0.1)


def run(tick: Callable, form: str, x: dict, path: torch.Tensor) -> tuple[torch.Tensor, float]:
    """``q (T, nq)`` after one ``tick`` per target of ``path (T, 7)`` from ``x``, and the median tick in seconds."""
    warm, q, seconds = None, [], []
    for target in path:
        began = time.perf_counter()
        x, warm = tick(form, target[None], x, warm)
        seconds.append(time.perf_counter() - began)
        q.append(x["q"][0])
    return torch.stack(q), float(np.median(seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="run the scripted path on the stand-in server and exit")
    parser.add_argument("--record", type=Path, help="record the scripted path to this .webm, .gif or .png")
    args = parser.parse_args()

    model = br.load(URDF, dtype=torch.float64)
    q0 = model.q_neutral.clone()[None]  # (1, nq)
    q0[0, :7] = torch.tensor(READY)
    start = forward_kinematics(model, q0).frame_poses[:, model.frame_id(FRAME)]  # (1, 7)
    lower, upper = model.lower_position_limit, model.upper_position_limit

    def unbounded() -> optim.Problem:
        """``terms.ik_problem`` without its bounds."""
        problem = optim.Problem({"q": q0}, manifolds={"q": terms.RobotManifold(model)}, data={f"target/{FRAME}": start})
        state = problem.cached(terms.kinematic_state(model), name="state")
        problem.residual(f"pose/{FRAME}", terms.frame_pose(model, frame=FRAME, target=f"target/{FRAME}", state=state))
        return problem

    problems = {"rows": terms.ik_problem(model, {FRAME: start}, q0=q0), "cost": unbounded(), "off": unbounded()}
    problems["cost"].residual("limits", terms.joint_limits(model, margin=0.05), weight=1e3)

    def tick(form: str, target: torch.Tensor, x: dict, warm: optim.QPSolution | None) -> tuple:
        """One ``track`` step toward ``target (1, 7)``."""
        return optim.track(problems[form].with_(data={f"target/{FRAME}": target}), x, warm=warm, step=STEP)

    def look(q: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """The arm's joint positions ``(T, 7, 3)`` and how far each joint is inside its range ``(T, 7)`` (negative
        past a limit) of ``q (T, nq)``."""
        poses = forward_kinematics(model, q).joint_poses
        return poses[:, 1:8, :3].numpy(), torch.minimum(q - lower, upper - q)[:, :7].numpy()

    geometry = br.io.load_geometry(URDF, model)
    viewer = Viewer(fps=25.0, server=FakeServer() if args.check else None)

    def draw(group: str, q: torch.Tensor, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list:
        """The robot at ``q (T, nq)`` and a marker on each joint at (amber) or past (red) its limit, under ``group``."""
        joints, inside = look(q)
        radii = np.full(7, 0.07)
        layers = [
            RobotVisual(model, q, name="robot", geometry=geometry, root=URDF),
            SpheresLayer("at the limit", joints, radii=radii, colors=AMBER, opacity=0.8),
            SpheresLayer("past the limit", joints, radii=radii, colors=RED, opacity=0.8),
        ]
        viewer.add(layers[:1], tags=group, pose=offset)
        viewer.add(layers[1:2], tags=(group, "markers"), pose=offset, valid=np.abs(inside) <= 1e-3)
        viewer.add(layers[2:], tags=(group, "markers"), pose=offset, valid=inside < -1e-3)
        return layers

    if not (args.check or args.record):
        follow(viewer, tick, look, draw("robot", q0), x=dict(problems["rows"].variables), start=start[0])
        return
    alpha = 0.5 - 0.5 * torch.cos(torch.linspace(0, 2 * math.pi, TICKS, dtype=torch.float64))
    turns = lie.so3.exp(torch.stack([0 * alpha, 0 * alpha, TURN * alpha], dim=-1))  # (TICKS, 4) about hand z
    path = torch.cat([start[:, :3].expand(TICKS, 3), lie.so3.compose(start[:, 3:], turns)], dim=-1)  # (TICKS, 7)
    for side, form in enumerate(("off", "rows")):
        q, seconds = run(tick, form, dict(problems[form].variables), path)
        inside = look(q)[1]
        print(
            f"limits {form:4s}: deepest joint {max(0.0, -inside.min()):.5f} rad past its limit "
            f"(panda_joint{int(inside.min(0).argmin()) + 1}), tick {1e3 * seconds:.1f} ms (median)"
        )
        draw(form, q[::8], (0.0, 1.0 * side - 0.5, 0.0))  # every 8th tick: 100 Hz ticks at twice real time
    if args.record:
        camera = viewer.aim(steps=range(0, viewer.timeline.nsteps, 10))  # record's own fit reads every vertex
        print(f"recorded {viewer.record(args.record, size=(960, 540), camera=camera)}")


def follow(viewer: Viewer, tick: Callable, look: Callable, robot: list, *, x: dict, start: torch.Tensor) -> None:
    """Track a gizmo that starts at ``start (7,)`` until Ctrl-C, with the form the GUI selects, and redraw
    ``robot``'s layers after every tick."""
    with viewer.sidebar.section("Joint limits"):  # after the Robot and Layers sections
        form = viewer.sidebar.gui.add_dropdown("Form", FORMS)
        readout = viewer.sidebar.gui.add_markdown("")
    gizmo = PoseGizmo(viewer, "target", pose=start.numpy(), scale=0.2)
    warm, last = None, form.value
    print(f"turn the target at {viewer.url}; Ctrl-C to exit")
    while True:
        began = time.perf_counter()
        if form.value != last:  # another program: the warm start no longer fits
            warm, last = None, form.value
        x, warm = tick(form.value, torch.from_numpy(gizmo.pose).to(start)[None], x, warm)
        seconds = time.perf_counter() - began
        joints, inside = look(x["q"])
        robot[0].update(x["q"])
        robot[1].centers[:] = robot[2].centers[:] = joints
        robot[1].valid[:], robot[2].valid[:] = np.abs(inside) <= 1e-3, inside < -1e-3
        viewer.redraw("markers")
        joint = int(inside[0].argmin())
        readout.content = (
            f"panda_joint{joint + 1} is closest: **{inside[0, joint]:+.3f} rad** inside its range  \n"
            f"tick {1e3 * seconds:.1f} ms"
        )
        time.sleep(max(0.0, 0.01 - (time.perf_counter() - began)))


if __name__ == "__main__":
    main()
