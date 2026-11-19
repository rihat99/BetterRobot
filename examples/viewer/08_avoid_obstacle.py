"""Obstacle avoidance while tracking: drag the obstacle into the robot; ``optim.track`` keeps its spheres 2 cm away.

    uv run python examples/viewer/08_avoid_obstacle.py                          # open http://127.0.0.1:8080
    uv run python examples/viewer/08_avoid_obstacle.py --robot g1
    uv run python examples/viewer/08_avoid_obstacle.py --check                  # the scripted path, stand-in server
    uv run python examples/viewer/08_avoid_obstacle.py --record obstacle.webm   # the scripted path, recorded

The obstacle is a sphere ``[x, y, z, radius]`` in a tensor that the distance function reads; the loop moves it in
place (``copy_``) before each tick, so the problem, and a compiled tick, stay the same. The rows see the obstacle
where it is when the tick starts: one that closes in by ``v`` per tick would end a tick ``v / gain`` inside the
margin, so its radius is padded by that much. The GUI switches the obstacle between its rows, a cost term (a
weighted hinge, which gives way a little) and off; robot spheres more than 1 mm inside the margin turn red. The
scripted path holds the hand still while the obstacle sweeps through it at 1 mm per tick, and shows the robot
without the rows (left) beside the robot with them (right).
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
from better_robot.viewer import RobotVisual, SpheresLayer
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
RADIUS, MARGIN, GAIN, TICKS, SPEED = 0.06, 0.02, 0.5, 1000, 0.001  # m, m, track's gain, ticks, m per tick
FORMS = ("rows", "cost", "off")  # the GUI's choices: the obstacle as rows, as a cost term, or ignored
RED, GREY = (0.9, 0.1, 0.1), (0.45, 0.45, 0.5)


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


def run(tick: Callable, form: str, x: dict, sweep: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
    """``q (T, nq)`` after one ``tick`` toward ``target (1, 7)`` per obstacle centre of ``sweep (T, 3)`` from ``x``, and
    the median tick in seconds."""
    warm, q, seconds = None, [], []
    for center in sweep:
        began = time.perf_counter()
        x, warm = tick(form, target, center, x, warm)
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
    start = forward_kinematics(model, q0).frame_poses[:, [model.frame_id(f) for f in frames]]  # (1, F, 7)
    hand = start[0, 0, :3]
    obstacle = torch.cat([hand + torch.tensor([0.03, -0.5, 0.0], dtype=torch.float64), torch.tensor([RADIUS])])

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:  # (B, S, 3), (S,) -> (B, S)
        return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]

    problems = {form: terms.ik_problem(model, dict(zip(frames, start.unbind(1))), q0=q0) for form in FORMS}
    row = terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN, hinge=False, closest=closest)
    problems["rows"].inequality("obstacle", row, upper=0.0)
    problems["cost"].residual("obstacle", terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN), weight=1e3)

    def tick(form: str, target: torch.Tensor, center: torch.Tensor, x: dict, warm: optim.QPSolution | None) -> tuple:
        """Move the obstacle to ``center (3,)``, its radius padded by its move over ``GAIN``, and take one ``track``
        step toward ``target (1, 7)`` for the first frame; the others hold their start poses."""
        obstacle.copy_(torch.cat([center, RADIUS + (center - obstacle[:3]).norm(dim=0, keepdim=True) / GAIN]))
        problem = problems[form].with_(data={f"target/{frames[0]}": target})
        return optim.track(problem, x, warm=warm, gain=GAIN, step=step)

    def look(q: torch.Tensor, centers: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Sphere centres ``(T, S, 3)`` and clearances ``(T, S)`` of ``q (T, nq)`` to the obstacle of radius
        ``RADIUS`` at ``centers (T, 3)``."""
        robot = spheres.centers(forward_kinematics(model, q))
        clearance = (robot - centers[:, None]).norm(dim=-1) - spheres.radii - RADIUS
        return robot.numpy(), clearance.numpy()

    radii = spheres.radii.numpy()
    viewer = Viewer(fps=25.0, server=FakeServer() if args.check else None)

    def draw(
        group: str, q: torch.Tensor, centers: torch.Tensor, offset: tuple[float, float, float] = (0, 0, 0)
    ) -> list:
        """The robot at ``q (T, nq)``, its spheres, the ones inside the margin (red) and the obstacle at ``centers
        (T, 3)``, under ``group``."""
        robot, clearance = look(q, centers)
        layers = [
            RobotVisual(model, q, name="robot", geometry=geometry, root=description.URDF_PATH, spheres=spheres,
                        collision=True),  # the Robot section's Collision box shows the spheres
            SpheresLayer("obstacle", centers[:, None].numpy(), radii=np.array([RADIUS]), colors=GREY, opacity=0.9),
            SpheresLayer("inside the margin", robot, radii=radii, colors=RED, opacity=0.8),
        ]  # fmt: skip
        viewer.add(layers[:2], tags=group, pose=offset)
        viewer.add(layers[2:], tags=group, pose=offset, valid=clearance < MARGIN - 1e-3)
        return layers

    if not (args.check or args.record):
        center = obstacle[:3].clone()  # where the scripted path starts: outside every margin on both robots
        robot = draw("robot", q0, center[None])
        follow(viewer, tick, look, robot, x=dict(problems["rows"].variables), start=start[0, 0], center=center)
        return
    sweep = obstacle[:3] + torch.linspace(0, SPEED * TICKS, TICKS, dtype=torch.float64)[:, None] * torch.tensor(
        [0.0, 1.0, 0.0], dtype=torch.float64
    )  # (TICKS, 3): through the hand
    for side, form in enumerate(("off", "rows")):
        obstacle[:3] = sweep[0]
        q, seconds = run(tick, form, dict(problems[form].variables), sweep, start[:, 0])
        gave_way = (forward_kinematics(model, q).frame_poses[:, model.frame_id(frames[0]), :3] - hand).norm(dim=-1)
        print(
            f"obstacle {form:4s}: closest sphere {1e3 * float(look(q, sweep)[1].min()):+8.3f} mm from it "
            f"(margin {1e3 * MARGIN:.0f}), the hand gave way up to {1e3 * float(gave_way.max()):.0f} mm, "
            f"tick {1e3 * seconds:.1f} ms (median)"
        )
        draw(form, q[::10], sweep[::10], (1.2 * side - 0.6, 0.0, 0.0))  # every 10th tick: 2.5 times real time
    if args.record:
        camera = viewer.aim(steps=range(0, viewer.timeline.nsteps, 10))  # record's own fit reads every vertex
        print(f"recorded {viewer.record(args.record, size=(960, 540), camera=camera)}")


def follow(
    viewer: Viewer, tick: Callable, look: Callable, robot: list, *, x: dict, start: torch.Tensor, center: torch.Tensor
) -> None:
    """Track a target gizmo that starts at ``start (7,)`` and an obstacle gizmo at ``center (3,)`` until Ctrl-C, with
    the form the GUI selects, and redraw ``robot``'s layers after every tick."""
    with viewer.sidebar.section("Obstacle"):  # after the Robot and Layers sections
        form = viewer.sidebar.gui.add_dropdown("Form", FORMS)
        readout = viewer.sidebar.gui.add_markdown("")
    gizmo = PoseGizmo(viewer, "target", pose=start.numpy(), scale=0.2)
    mover = PoseGizmo(viewer, "obstacle", pose=np.concatenate([center.numpy(), [0.0, 0.0, 0.0, 1.0]]), scale=0.2)
    warm, last = None, form.value
    print(f"drag the obstacle (or the target) at {viewer.url}; Ctrl-C to exit")
    while True:
        began = time.perf_counter()
        if form.value != last:  # another row count: the warm start no longer fits
            warm, last = None, form.value
        center = torch.from_numpy(mover.pose[:3]).to(start)
        x, warm = tick(form.value, torch.from_numpy(gizmo.pose).to(start)[None], center, x, warm)
        seconds = time.perf_counter() - began
        centers, clearance = look(x["q"], center[None])
        robot[0].update(x["q"])
        robot[1].centers[:] = center.numpy()
        robot[2].centers[:] = centers
        robot[2].valid[:] = clearance < MARGIN - 1e-3
        robot[1].redraw()
        robot[2].redraw()
        readout.content = (
            f"closest sphere **{1e3 * clearance.min():.1f} mm** from the obstacle (margin {1e3 * MARGIN:.0f})  \n"
            f"tick {1e3 * seconds:.1f} ms"
        )
        time.sleep(max(0.0, 0.01 - (time.perf_counter() - began)))


if __name__ == "__main__":
    main()
