"""Singularity avoidance while tracking: pull the target out of reach; ``optim.track`` keeps the arm from going straight.

    uv run python examples/viewer/06_avoid_singularity.py                          # open http://127.0.0.1:8080
    uv run python examples/viewer/06_avoid_singularity.py --check                  # the scripted path, stand-in server
    uv run python examples/viewer/06_avoid_singularity.py --record singular.webm   # the scripted path, recorded

The hand's manipulability ``w = sqrt(det(J Jᵀ))`` (position rows of its Jacobian) falls to zero as the arm
stretches out straight, where the hand can no longer move along the arm. ``terms.manipulability(..., hinge=False)``
is the row ``threshold - w <= 0`` of the program ``track`` solves every tick. The GUI switches it between that row,
a cost term (a weighted hinge, which gives way a little) and off; the hand's marker turns red below the threshold.
The scripted path pulls the target 80 cm out and down, tilting it outward, and back (the hand lines up with the
shoulder and the elbow there), and shows the Panda without the row
(left) beside the Panda with it (right).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch

from better_robot import Viewer, forward_kinematics, lie, optim, terms
from better_robot.viewer import RobotVisual, SpheresLayer
from better_robot.viewer.tools import PoseGizmo

import avoidance  # beside this file: the command line, robots and scripted run the avoidance demos share

FRAME = "panda_hand"
THRESHOLD, STEP, TICKS = 0.03, 0.005, 1200  # manipulability (m³ for three position rows), rad per tick, ticks
REACH, TILT = (0.75, 0.0, -0.3), 0.8  # the scripted target's far point from its start (m), its tilt there (rad)
FORMS = ("rows", "cost", "off")  # the GUI's choices: the singular poses as a row, as a cost term, or not avoided
RED, GREY = (0.9, 0.1, 0.1), (0.6, 0.6, 0.65)


def main() -> None:
    args = avoidance.arguments()
    model, q0, geometry = avoidance.load("panda")  # q0 (1, nq): the ready pose
    hand = model.frame_id(FRAME)
    start = forward_kinematics(model, q0).frame_poses[:, hand]  # (1, 7)
    singular = terms.manipulability(model, frame=FRAME, threshold=THRESHOLD, hinge=False)
    problems = {form: terms.ik_problem(model, {FRAME: start}, q0=q0) for form in FORMS}
    problems["rows"].inequality("singular", singular, upper=0.0)
    problems["cost"].residual("singular", terms.manipulability(model, frame=FRAME, threshold=THRESHOLD), weight=1e3)

    def tick(form: str, target: torch.Tensor, x: dict, warm: optim.QPSolution | None) -> tuple:
        """One ``track`` step toward ``target (1, 7)``."""
        return optim.track(problems[form].with_(data={f"target/{FRAME}": target}), x, warm=warm, step=STEP)

    def look(q: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """The hand's position ``(T, 1, 3)`` and manipulability ``(T, 1)`` of ``q (T, nq)``."""
        state = forward_kinematics(model, q)
        w = THRESHOLD - singular.fn(state)
        return state.frame_poses[:, [hand], :3].numpy(), w.numpy()

    viewer = avoidance.viewer(check=args.check)

    def draw(group: str, q: torch.Tensor, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list:
        """The robot at ``q (T, nq)`` and its hand's marker, red below the threshold, under ``group``."""
        point, w = look(q)
        layers = [
            RobotVisual(model, q, name="robot", geometry=geometry, root=avoidance.URDFS["panda"]),
            SpheresLayer("hand", point, radii=np.array([0.05]), colors=GREY, opacity=0.4),
            SpheresLayer("below the threshold", point, radii=np.array([0.05]), colors=RED, opacity=0.8),
        ]
        viewer.add(layers[:1], tags=group, pose=offset)
        viewer.add(layers[1:2], tags=(group, "markers"), pose=offset)
        viewer.add(layers[2:], tags=(group, "markers"), pose=offset, valid=w < THRESHOLD - 1e-4)
        return layers

    if not (args.check or args.record):
        follow(viewer, tick, look, draw("robot", q0), x=dict(problems["rows"].variables), start=start[0])
        return
    alpha = 0.5 - 0.5 * torch.cos(torch.linspace(0, 2 * math.pi, TICKS, dtype=torch.float64))[:, None]
    tilt = lie.so3.exp(alpha * torch.tensor([0.0, -TILT, 0.0], dtype=torch.float64))  # (TICKS, 4) about world y
    position = start[:, :3] + alpha * torch.tensor(REACH, dtype=torch.float64)
    path = torch.cat([position, lie.so3.compose(tilt, start[:, 3:].expand(TICKS, 4))], dim=-1)  # (TICKS, 7)
    for side, form in enumerate(("off", "rows")):
        q, seconds = avoidance.run(tick, form, dict(problems[form].variables), path)
        print(
            f"singular {form:4s}: lowest manipulability {float(look(q)[1].min()):.5f} (threshold {THRESHOLD}), "
            f"tick {1e3 * seconds:.1f} ms (median)"
        )
        offset = (0.0, 1.0 * side - 0.5, 0.0)
        draw(form, q[::12], offset)  # every 12th tick: 100 Hz ticks at three times real time
        target = SpheresLayer("target", path[::12, None, :3].numpy(), radii=np.array([0.02]), colors=(0.1, 0.3, 0.9))
        viewer.add([target], tags=form, pose=offset)
    if args.record:
        avoidance.record(viewer, args.record)


def follow(viewer: Viewer, tick: Callable, look: Callable, robot: list, *, x: dict, start: torch.Tensor) -> None:
    """Track a gizmo that starts at ``start (7,)`` until Ctrl-C, with the form the GUI selects, and redraw
    ``robot``'s layers after every tick."""
    with viewer.sidebar.section("Singular poses"):  # after the Robot and Layers sections
        form = viewer.sidebar.gui.add_dropdown("Form", FORMS)
        readout = viewer.sidebar.gui.add_markdown("")
    gizmo = PoseGizmo(viewer, "target", pose=start.numpy(), scale=0.2)
    warm, last = None, form.value
    print(f"pull the target out of reach at {viewer.url}; Ctrl-C to exit")
    while True:
        began = time.perf_counter()
        if form.value != last:  # another row count: the warm start no longer fits
            warm, last = None, form.value
        x, warm = tick(form.value, torch.from_numpy(gizmo.pose).to(start)[None], x, warm)
        seconds = time.perf_counter() - began
        point, w = look(x["q"])
        robot[0].update(x["q"])
        robot[1].centers[:] = robot[2].centers[:] = point
        robot[2].valid[:] = w < THRESHOLD - 1e-4
        viewer.redraw("markers")
        readout.content = (
            f"manipulability **{float(w[0, 0]):.4f}** (threshold {THRESHOLD})  \ntick {1e3 * seconds:.1f} ms"
        )
        time.sleep(max(0.0, 0.01 - (time.perf_counter() - began)))


if __name__ == "__main__":
    main()
