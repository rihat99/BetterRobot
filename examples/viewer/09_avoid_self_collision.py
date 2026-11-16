"""Self-collision avoidance while tracking: drag the target through the robot; ``optim.track`` keeps it out of itself.

    uv run python examples/viewer/09_avoid_self_collision.py                      # open http://127.0.0.1:8080
    uv run python examples/viewer/09_avoid_self_collision.py --robot g1
    uv run python examples/viewer/09_avoid_self_collision.py --check              # the scripted path, stand-in server
    uv run python examples/viewer/09_avoid_self_collision.py --record self.webm   # the scripted path, recorded

Every pair of the sphere set that can collide (``collision.self_collision_pairs`` without the file's ignore list) is
kept apart by a margin of 2 cm, or by its distance at the start less 1 mm where that is smaller (60 of the standing
G1's 3890 pairs), so that the start holds every row; the ``closest`` pairs of each tick are the rows of the program
``track`` solves (``terms.self_clearance(..., hinge=False)``). The GUI switches between those rows, a cost term (a
weighted hinge, which gives way a little) and off; spheres of a pair more than 1 mm inside its margin turn red. The
ignore list holds the pairs whose shapes touch at the neutral pose, such as the Panda's hand and fifth link: those
are never checked. The scripted path drags the hand target through the robot's own body and back, and shows the
robot without the rows (left) beside the robot with them (right).
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
from better_robot.viewer import GroundLayer, MeshSpec, RigidMeshLayer, SpheresLayer, robot_meshes
from better_robot.viewer.tools import PoseGizmo

READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
ROBOTS = {  # description, IK frames (the first follows the target), step per tick, closest pairs as rows, QP iterations
    "panda": (panda_description, ("panda_hand",), 0.005, 8, 20),
    "g1": (
        g1_description,
        ("left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link"),
        0.0063,
        64,
        100,  # many pairs press at once: 20 iterations leave more steps inexact
    ),
}
PATH = {"panda": (-0.3, 0.15, -0.35), "g1": (-0.15, -0.35, 0.05)}  # the scripted target's far point, from its start
MARGIN, TICKS, RED = 0.02, 800, (0.9, 0.1, 0.1)
FORMS = ("rows", "cost", "off")  # the GUI's choices: the pairs as rows, as a cost term, or not checked


def load(robot: str) -> tuple[br.Model, torch.Tensor, list[MeshSpec]]:
    """The model of ``robot`` (the G1 on a free base), its start ``q0 (1, nq)`` (Panda ready, G1 standing on z = 0)
    and its display meshes."""
    description, g1 = ROBOTS[robot][0], robot == "g1"
    model = br.load(description.URDF_PATH, root_joint=br.JointFreeFlyer() if g1 else None, dtype=torch.float64)
    q0 = model.q_neutral.clone()[None]
    if g1:
        q0[0, 2] = 0.793
    else:
        q0[0, :7] = torch.tensor(READY)
    geometry = br.io.load_geometry(description.URDF_PATH, model)
    root = Path(description.URDF_PATH).parent if g1 else Path(description.REPOSITORY_PATH)
    return model, q0, robot_meshes(geometry, root=root)


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

    model, q0, meshes = load(args.robot)
    _, frames, step, closest, iterations = ROBOTS[args.robot]
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / f"{args.robot}.json", model)
    pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)  # (P, 2) sphere indices
    apart = collision.self_distances(spheres, forward_kinematics(model, q0), pairs=pairs)[0]  # (P,) at the start
    margin = (apart - 1e-3).clamp(max=MARGIN)  # (P,): a pair that starts closer keeps its start distance less 1 mm
    start = forward_kinematics(model, q0).frame_poses[:, [model.frame_id(f) for f in frames]]  # (1, F, 7)
    problems = {form: terms.ik_problem(model, dict(zip(frames, start.unbind(1))), q0=q0) for form in FORMS}
    row = terms.self_clearance(spheres, pairs=pairs, margin=margin, hinge=False, closest=closest)
    problems["rows"].inequality("self", row, upper=0.0)
    cost = terms.self_clearance(spheres, pairs=pairs, margin=margin, closest=closest)
    problems["cost"].residual("self", cost, weight=1e3)

    def tick(form: str, target: torch.Tensor, x: dict, warm: optim.QPSolution | None) -> tuple:
        """One ``track`` step toward ``target (1, 7)`` for the first frame; the others hold their start poses."""
        problem = problems[form].with_(data={f"target/{frames[0]}": target})
        return optim.track(problem, x, warm=warm, step=step, iterations=iterations)

    def look(q: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Joint poses ``(T, J, 7)``, sphere centres ``(T, S, 3)``, which spheres are in a pair inside its margin
        ``(T, S)`` and each pair's distance less its margin ``(T, P)`` of ``q (T, nq)``."""
        state = forward_kinematics(model, q)
        clearance = collision.self_distances(spheres, state, pairs=pairs) - margin
        inside = (clearance < -1e-3).double()
        hits = torch.zeros(q.shape[0], len(spheres.radii), dtype=torch.float64)
        hits = hits.index_add(1, pairs[:, 0], inside).index_add(1, pairs[:, 1], inside)
        return state.joint_poses.numpy(), spheres.centers(state).numpy(), hits.numpy() > 0, clearance.numpy()

    viewer = Viewer(fps=25.0, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer(size=4.0)])

    def draw(group: str, q: torch.Tensor, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list:
        """The robot at ``q (T, nq)``, its spheres and the ones of a pair inside its margin (red), under ``group``."""
        poses, centers, red, _ = look(q)
        radii = spheres.radii.numpy()
        layers = [
            RigidMeshLayer("robot", meshes, body_poses=poses),
            SpheresLayer("spheres", centers, radii=radii, opacity=0.15),
            SpheresLayer("inside the margin", centers, radii=radii, colors=RED, opacity=0.8),
        ]
        viewer.add(layers[:2], group=group, offset=offset)
        viewer.add(layers[2:], group=group, offset=offset, valid=red)
        return layers

    if not (args.check or args.record):
        follow(viewer, tick, look, draw("robot", q0), x=dict(problems["rows"].variables), start=start[0, 0])
        return
    alpha = 0.5 - 0.5 * torch.cos(torch.linspace(0, 2 * math.pi, TICKS, dtype=torch.float64))[:, None]
    path = start[:, 0].repeat(TICKS, 1)  # (TICKS, 7): into the body and back
    path[:, :3] += alpha * torch.tensor(PATH[args.robot], dtype=torch.float64)
    for side, form in enumerate(("off", "rows")):
        q, seconds = run(tick, form, dict(problems[form].variables), path)
        print(
            f"self {form:4s}: deepest pair {1e3 * look(q)[3].min():+8.3f} mm outside its margin ({len(pairs)} pairs, "
            f"margins {1e3 * margin.min():.2f} to {1e3 * MARGIN:.0f} mm), tick {1e3 * seconds:.1f} ms (median)"
        )
        offset = (0.0, 1.2 * side - 0.6, 0.0)
        draw(form, q[::8], offset)  # every 8th tick: 100 Hz ticks at twice real time
        target = SpheresLayer("target", path[::8, None, :3].numpy(), radii=np.array([0.02]), colors=(0.1, 0.3, 0.9))
        viewer.add([target], group=form, offset=offset)
    if args.record:
        camera = viewer.aim(steps=range(0, viewer.timeline.nsteps, 10))  # record's own fit reads every vertex
        print(f"recorded {viewer.record(args.record, size=(960, 540), camera=camera)}")


def follow(viewer: Viewer, tick: Callable, look: Callable, robot: list, *, x: dict, start: torch.Tensor) -> None:
    """Track a gizmo that starts at ``start (7,)`` until Ctrl-C, with the form the GUI selects, and redraw
    ``robot``'s layers after every tick."""
    form = viewer.server.gui.add_dropdown("self-collision", FORMS)
    readout = viewer.server.gui.add_markdown("")
    gizmo = PoseGizmo(viewer, "target", pose=start.numpy(), scale=0.2)
    warm, last = None, form.value
    print(f"drag the target through the robot at {viewer.url}; Ctrl-C to exit")
    while True:
        began = time.perf_counter()
        if form.value != last:  # another row count: the warm start no longer fits
            warm, last = None, form.value
        x, warm = tick(form.value, torch.from_numpy(gizmo.pose).to(start)[None], x, warm)
        seconds = time.perf_counter() - began
        poses, centers, red, clearance = look(x["q"])
        robot[0].body_poses[:] = poses
        robot[1].centers[:] = robot[2].centers[:] = centers
        robot[2].valid[:] = red
        viewer.show()
        readout.content = (
            f"deepest pair **{1e3 * clearance.min():.1f} mm** outside its margin  \ntick {1e3 * seconds:.1f} ms"
        )
        time.sleep(max(0.0, 0.01 - (time.perf_counter() - began)))


if __name__ == "__main__":
    main()
