"""Floor avoidance while tracking: drag the target below the floor; ``optim.track`` keeps the robot's spheres 2 cm above.

    uv run python examples/viewer/07_avoid_floor.py                      # open http://127.0.0.1:8080, drag the target
    uv run python examples/viewer/07_avoid_floor.py --robot g1
    uv run python examples/viewer/07_avoid_floor.py --check              # the scripted path on the stand-in server
    uv run python examples/viewer/07_avoid_floor.py --record floor.webm  # the scripted path, recorded (Chromium)

Each sphere of the robot's sphere set is a row of the program ``track`` solves every tick (``hinge=False``,
``upper=0``). The GUI switches the floor between those rows, a cost term (the same clearance as a weighted hinge,
which gives way a little) and off; spheres more than 1 mm inside the margin turn red. The scripted path takes the
target 65 cm (Panda) or 105 cm (G1) down and back, and shows the robot without the floor (left) beside the robot
with the rows (right).
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import g1_description, panda_description

import better_robot as br
from better_robot import Viewer, collision, forward_kinematics, optim, terms
from better_robot.testing import FakeServer
from better_robot.viewer import GroundLayer, RobotVisual, SpheresLayer
from better_robot.viewer.tools import PoseGizmo

READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
ROBOTS = {  # description, IK frames (the first follows the target), step per tick, closest spheres as rows
    "panda": (panda_description, ("panda_hand",), 0.005, 8),
    "g1": (
        g1_description,
        ("left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link"),
        0.0063,
        None,
    ),
}
STANDING = {"panda": ("panda_link0", "panda_link1"), "g1": ("left_ankle_roll_link", "right_ankle_roll_link")}
PATH = {"panda": (0.25, 0.0, -0.65), "g1": (0.3, 0.0, -1.05)}  # the scripted target's lowest point, from its start
MARGIN, TICKS, RED = 0.02, 800, (0.9, 0.1, 0.1)
FORMS = ("rows", "cost", "off")  # the GUI's choices: the floor as rows, as a cost term, or no floor


def load(robot: str) -> tuple[br.Model, torch.Tensor, br.Geometry]:
    """The model of ``robot`` (the G1 on a free base), its start ``q0 (1, nq)`` (Panda ready, G1 standing on z = 0)
    and its geometry."""
    description, g1 = ROBOTS[robot][0], robot == "g1"
    model = br.load(description.URDF_PATH, root_joint=br.JointFreeFlyer() if g1 else None, dtype=torch.float64)
    q0 = model.q_neutral.clone()[None]
    if g1:
        q0[0, 2] = 0.793
    else:
        q0[0, :7] = torch.tensor(READY)
    return model, q0, br.io.load_geometry(description.URDF_PATH, model)


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
    parser.add_argument("--robot", choices=ROBOTS, default="panda")
    parser.add_argument("--check", action="store_true", help="run the scripted path on the stand-in server and exit")
    parser.add_argument("--record", type=Path, help="record the scripted path to this .webm, .gif or .png")
    args = parser.parse_args()

    model, q0, geometry = load(args.robot)
    description, frames, step, closest = ROBOTS[args.robot]
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / f"{args.robot}.json", model)
    standing = torch.tensor([model.frame_names[int(f)] in STANDING[args.robot] for f in spheres.frame_ids])
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    def to_floor(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:  # (B, S, 3), (S,) -> (B, S)
        return collision.sphere_halfspace(centers, radii, normal=up, offset=0.0) + standing  # feet, base: 1 m

    start = forward_kinematics(model, q0).frame_poses[:, [model.frame_id(f) for f in frames]]  # (1, F, 7)
    problems = {form: terms.ik_problem(model, dict(zip(frames, start.unbind(1))), q0=q0) for form in FORMS}
    row = terms.sphere_clearance(spheres, to_floor, margin=MARGIN, hinge=False, closest=closest)
    problems["rows"].inequality("floor", row, upper=0.0)
    problems["cost"].residual("floor", terms.sphere_clearance(spheres, to_floor, margin=MARGIN), weight=1e3)

    def tick(form: str, target: torch.Tensor, x: dict, warm: optim.QPSolution | None) -> tuple:
        """One ``track`` step toward ``target (1, 7)`` for the first frame; the others hold their start poses."""
        return optim.track(problems[form].with_(data={f"target/{frames[0]}": target}), x, warm=warm, step=step)

    def look(q: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Sphere centres ``(T, S, 3)`` and clearances ``(T, S)`` of ``q (T, nq)``."""
        centers = spheres.centers(forward_kinematics(model, q))
        return centers.numpy(), to_floor(centers, spheres.radii).numpy()

    viewer = Viewer(fps=25.0, ground=False, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer(size=4.0, fill=0.5)])  # a solid floor: what sinks below it fades

    def draw(group: str, q: torch.Tensor, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list:
        """The robot at ``q (T, nq)``, its spheres and the ones inside the margin (red), under ``group``."""
        centers, clearance = look(q)
        radii = spheres.radii.numpy()
        layers = [
            RobotVisual(model, q, name="robot", geometry=geometry, root=description.URDF_PATH),
            SpheresLayer("spheres", centers, radii=radii, opacity=0.15),
            SpheresLayer("inside the margin", centers, radii=radii, colors=RED, opacity=0.8),
        ]
        viewer.add(layers[:1], tags=group, offset=offset)
        viewer.add(layers[1:2], tags=(group, "spheres"), offset=offset)  # the Layers section switches the tag
        viewer.add(layers[2:], tags=(group, "spheres"), offset=offset, valid=clearance < MARGIN - 1e-3)
        return layers

    if not (args.check or args.record):
        follow(viewer, tick, look, draw("robot", q0), x=dict(problems["rows"].variables), start=start[0, 0])
        return
    alpha = 0.5 - 0.5 * torch.cos(torch.linspace(0, 2 * math.pi, TICKS, dtype=torch.float64))[:, None]
    path = start[:, 0].repeat(TICKS, 1)  # (TICKS, 7): down and back up
    path[:, :3] += alpha * torch.tensor(PATH[args.robot], dtype=torch.float64)
    for side, form in enumerate(("off", "rows")):
        q, seconds = run(tick, form, dict(problems[form].variables), path)
        print(
            f"floor {form:4s}: lowest sphere {1e3 * look(q)[1].min():+8.3f} mm above the floor "
            f"(margin {1e3 * MARGIN:.0f}), tick {1e3 * seconds:.1f} ms (median)"
        )
        offset = (0.0, 1.2 * side - 0.6, 0.0)
        draw(form, q[::8], offset)  # every 8th tick: 100 Hz ticks at twice real time
        target = SpheresLayer("target", path[::8, None, :3].numpy(), radii=np.array([0.02]), colors=(0.1, 0.3, 0.9))
        viewer.add([target], tags=form, offset=offset)
    if args.record:
        camera = viewer.aim(steps=range(0, viewer.timeline.nsteps, 10))  # record's own fit reads every vertex
        print(f"recorded {viewer.record(args.record, size=(960, 540), camera=camera)}")


def follow(viewer: Viewer, tick: Callable, look: Callable, robot: list, *, x: dict, start: torch.Tensor) -> None:
    """Track a gizmo that starts at ``start (7,)`` until Ctrl-C, with the form the GUI selects, and redraw
    ``robot``'s layers after every tick."""
    with viewer.sidebar.section("Floor"):  # after the Robot and Layers sections
        form = viewer.sidebar.gui.add_dropdown("Form", FORMS)
        readout = viewer.sidebar.gui.add_markdown("")
    gizmo = PoseGizmo(viewer, "target", pose=start.numpy(), scale=0.2)
    warm, last = None, form.value
    print(f"drag the target at {viewer.url}; Ctrl-C to exit")
    while True:
        began = time.perf_counter()
        if form.value != last:  # another row count: the warm start no longer fits
            warm, last = None, form.value
        x, warm = tick(form.value, torch.from_numpy(gizmo.pose).to(start)[None], x, warm)
        seconds = time.perf_counter() - began
        centers, clearance = look(x["q"])
        robot[0].update(x["q"])
        robot[1].centers[:] = robot[2].centers[:] = centers
        robot[2].valid[:] = clearance < MARGIN - 1e-3
        viewer.redraw("spheres")
        readout.content = (
            f"lowest sphere **{1e3 * clearance.min():.1f} mm** above the floor (margin {1e3 * MARGIN:.0f})  \n"
            f"tick {1e3 * seconds:.1f} ms"
        )
        time.sleep(max(0.0, 0.01 - (time.perf_counter() - began)))


if __name__ == "__main__":
    main()
