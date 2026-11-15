"""Fit spheres to the collision shapes of a robot description and write the file ``collision.load_spheres`` reads.

Per body, in the body's frame, at most ``--count`` spheres hold the body's shapes (``fit``): a greedy cover whose
overshoot past the surface is the smallest a bisection finds, then each radius shrinks to the points it holds with
the most room, and a sample ten times denser widens it again where a point fell between the fitted ones. Centres
are rounded to 0.1 mm and radii rounded up with 0.1 mm added. With ``--mirror`` a ``left_`` body is fitted with its
mirrored ``right_`` twin and the right body gets the mirror image, and every other body is fitted symmetric about its
plane y = 0 (the sagittal plane), so mirrored pairs behave alike. ``--sole BODY`` keeps BODY's spheres within 5 mm
below its lowest point along its z axis, so a floor constraint holds a foot at its sole.

The ignore list follows MoveIt's rules, checked on the shapes themselves with coal (from Pinocchio, in the
``test-oracles`` group): a frame pair that ``self_collision_pairs`` keeps is ignored when its shapes collide in none of
100000 random configurations within the joint limits (torch generator seed 0) nor at the neutral configuration
("never"), in all of them ("always"), or at the neutral configuration ("neutral", MoveIt's "default"). Each entry
carries its reason. Every other pair whose spheres overlap at the neutral configuration has both bodies fitted again
with each sphere reaching at most halfway across the shapes' gap there: near the other body the spheres give up
holding the shapes, and the report's gap says by how much.

Measured per body on a third sample the fit never saw: the gap, how far a surface point lies outside every sphere,
and the overshoot, how far the surface of the union of spheres lies from the nearest surface point.

    uv run --extra meshes --extra demos --group test-oracles python tools/fit_spheres.py DESCRIPTION OUTPUT \\
        --default 2 --count panda_link0=3 --visual left_rubber_hand --mirror --sole left_ankle_roll_link

DESCRIPTION is a URDF or MJCF file whose meshes resolve beside it, or the name of a ``robot_descriptions`` module
(``g1_description``), which fetches the description into its cache on first use. The file's ``command`` records the
call; run it from the repository root.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shlex
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import coal
import numpy as np
import torch
import trimesh

from better_robot import collision, forward_kinematics, io
from better_robot.model import Box, Capsule, Cylinder, Geometry, Mesh, Model, Sphere

CONFIGURATIONS, SEED = 100000, 0
SOLE, REACH = 0.005, 0.5
MIRROR = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)


def body_mesh(geometry: Geometry, body: int, root: Path, *, visual: bool) -> trimesh.Trimesh | None:
    """The collision (or, with ``visual``, display) shapes of ``body`` as one mesh in the body's frame; mesh URIs
    resolve under ``root``."""
    meshes = []
    for attached in geometry.shapes:
        if attached.body != body or attached.collision == visual:
            continue
        shape = attached.shape
        if isinstance(shape, Mesh):
            uri = (
                shape.uri.removeprefix("package://").partition("/")[2]
                if shape.uri.startswith("package://")
                else shape.uri
            )
            mesh = trimesh.load(root / uri, force="mesh")
            mesh.apply_scale(shape.scale)
        elif isinstance(shape, Box):
            mesh = trimesh.creation.box(extents=shape.size)
        elif isinstance(shape, Cylinder):
            mesh = trimesh.creation.cylinder(radius=shape.radius, height=shape.length, sections=64)
        elif isinstance(shape, Capsule):
            mesh = trimesh.creation.capsule(radius=shape.radius, height=shape.length)
        elif isinstance(shape, Sphere):
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=shape.radius)
        x, y, z, qx, qy, qz, qw = shape.placement
        transform = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
        transform[:3, 3] = (x, y, z)
        meshes.append(mesh.apply_transform(transform))
    return trimesh.util.concatenate(meshes) if meshes else None


def points_of(mesh: trimesh.Trimesh, samples: int, seed: int) -> torch.Tensor:
    """``(N, 3)``: the vertices of ``mesh`` and ``samples`` area-weighted points of its surface."""
    surface, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    return torch.from_numpy(np.concatenate([mesh.vertices, surface]))


def fit(
    mesh: trimesh.Trimesh,
    count: int,
    samples: int,
    *,
    symmetric: bool = False,
    cap: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centres ``(K, 3)`` and radii ``(K,)`` of at most ``count`` spheres that hold ``mesh``.

    A candidate centre ``c`` lies in the convex hull or on the surface, at depth ``d(c)`` from the nearest surface
    point; with an overshoot ``delta`` its sphere has radius ``d(c) + delta``, so it reaches at most ``delta`` past the
    surface of a solid that holds ``c``. The targets are surface points and, for a closed mesh, points inside it. A
    bisection finds the smallest ``delta`` for which the greedy cover (the sphere holding the most uncovered targets,
    in turn) holds every target with ``count`` spheres. Each radius then shrinks to the targets it holds with the most
    room, and grows again to hold a sample ten times denser.

    ``symmetric``: the spheres hold ``mesh`` and its mirror image in the plane y = 0 and come in mirror pairs (a sphere
    on the plane is its own pair); the fit runs on the half y >= 0. ``cap(centres (M, 3)) -> (M,)`` bounds each
    radius; the targets that ``count`` capped spheres do not hold are given up.
    """
    surface = points_of(mesh, samples, seed=0)[-samples:]
    dense = points_of(mesh, 10 * samples, seed=1)
    box = torch.from_numpy(np.random.default_rng(0).uniform(*mesh.bounds, size=(samples, 3)))
    hull = box[inside(mesh.convex_hull, box)]
    solid = hull[inside(mesh, hull)] if mesh.is_watertight else hull[:0]
    candidates = torch.cat([hull, surface[::4]])
    if symmetric:  # a point and its mirror image fold onto the same point of the half y >= 0

        def fold(points: torch.Tensor) -> torch.Tensor:
            return points * torch.where(points[:, 1:2] < 0, MIRROR, 1.0)

        surface, dense, solid = fold(surface), fold(dense), fold(solid)
        candidates = torch.cat([fold(candidates), hull * MIRROR.clamp(min=0)])
    targets = torch.cat([surface, solid])
    cost = 1 + (candidates[:, 1] > 0).long() if symmetric else torch.ones(len(candidates), dtype=torch.long)
    depth = torch.cat([torch.cdist(chunk, surface).amin(1) for chunk in candidates.split(2048)])
    room = torch.full_like(depth, math.inf)
    if cap is not None:
        room = cap(candidates) - depth
        if symmetric:
            room = room.minimum(cap(candidates * MIRROR) - depth)
    reach = torch.cdist(candidates, targets) - depth[:, None]
    if cap is not None:
        # Under the caps, the targets that count spheres can hold at any overshoot; the others are given up.
        _, uncovered = greedy_cover(reach <= room[:, None], count, cost)
        targets, reach = targets[~uncovered], reach[:, ~uncovered]
    low, high = 0.0, float(reach.max())
    for _ in range(30):
        middle = (low + high) / 2
        _, uncovered = greedy_cover(reach <= room.clamp(max=middle)[:, None], count, cost)
        low, high = (middle, high) if uncovered.any() else (low, middle)
    chosen, _ = greedy_cover(reach <= room.clamp(max=high)[:, None], count, cost)
    centres, radii = candidates[chosen], depth[chosen] + room[chosen].clamp(max=high)
    for _ in range(20):
        # Each target to the sphere that holds it with the most room; each radius shrinks to its targets.
        distance = torch.cdist(targets, centres)
        member = (distance - radii).argmin(1)[:, None] == torch.arange(len(radii))
        radii = distance.where(member, 0).amax(0)
    # Points between the fitted samples can lie just outside: a sample ten times denser widens the radii.
    distance = torch.cdist(dense, centres)
    widened = distance.where((distance - radii).argmin(1)[:, None] == torch.arange(len(radii)), 0).amax(0)
    radii = widened.maximum(radii)
    if cap is not None:
        radii = radii.minimum(cap(centres))
        if symmetric:
            radii = radii.minimum(cap(centres * MIRROR))
    if symmetric:
        twins = centres[:, 1] > 0
        centres, radii = torch.cat([centres, centres[twins] * MIRROR]), torch.cat([radii, radii[twins]])
    return centres, radii


def inside(mesh: trimesh.Trimesh, points: torch.Tensor) -> torch.Tensor:
    """``(N,)``: whether each of ``points (N, 3)`` lies inside the closed ``mesh``, by its winding number (the solid
    angle its triangles span seen from the point, over 4 pi): 1 inside, 2 inside two overlapping parts, 0 outside.
    Trimesh's own test casts rays with Embree, and its answer changes from one call to the next."""
    triangles = torch.tensor(mesh.triangles)
    turns = []
    for chunk in points.split(64):
        a, b, c = (triangles[None, :, k] - chunk[:, None] for k in range(3))
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        numerator = (a * torch.cross(b, c, dim=-1)).sum(-1)
        denominator = la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
        turns.append(torch.atan2(numerator, denominator).sum(-1) / (2 * math.pi))
    return torch.cat(turns).abs() > 0.5


def greedy_cover(holds: torch.Tensor, count: int, cost: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    """Rows of ``holds (M, T)`` (candidate ``m`` holds target ``t``) picked greedily, each costing ``cost[m]`` of the
    ``count`` spheres, until every target is held or no budget is left; and the targets ``(T,)`` left uncovered."""
    uncovered, chosen, spent = torch.ones(holds.shape[1], dtype=torch.bool), [], 0
    while uncovered.any() and spent < count:
        gain = (holds & uncovered).sum(1).masked_fill(cost > count - spent, -1)
        best = int(gain.argmax())
        if int(gain[best]) <= 0:
            break
        chosen.append(best)
        spent += int(cost[best])
        uncovered &= ~holds[best]
    return chosen, uncovered


def rounded(centres: torch.Tensor, radii: torch.Tensor) -> list[list[float]]:
    """``[x, y, z, radius]`` rows at 0.1 mm; a radius is rounded up and widened by 0.1 mm for the centre's rounding."""
    return [
        [round(float(value), 4) + 0.0 for value in centre] + [round(math.ceil(float(radius) * 1e4) / 1e4 + 1e-4, 4)]
        for centre, radius in zip(centres, radii, strict=True)
    ]


def measure(mesh: trimesh.Trimesh, spheres: torch.Tensor, samples: int) -> tuple[float, float, float]:
    """Gap (max), overshoot (mean, max) in metres of the spheres ``(K, 4)`` against ``mesh``, on a sample the fit
    never saw."""
    points = points_of(mesh, 10 * samples, seed=2)
    centres, radii = spheres[:, :3], spheres[:, 3]
    gap = (torch.cdist(points, centres) - radii).min(1).values.clamp(min=0).max()
    # The union's surface: points of each sphere that lie outside every other sphere.
    golden = torch.arange(500, dtype=points.dtype) + 0.5
    polar, azimuth = torch.acos(1 - 2 * golden / 500), math.pi * (1 + 5**0.5) * golden
    unit = torch.stack([polar.sin() * azimuth.cos(), polar.sin() * azimuth.sin(), polar.cos()], dim=1)
    shell = (centres[:, None] + radii[:, None, None] * unit).reshape(-1, 3)
    inside = torch.cdist(shell, centres) - radii < -1e-9
    surface = shell[~inside.any(1)]
    overshoot = torch.cat([torch.cdist(chunk, points).min(1).values for chunk in surface.split(4096)])
    return float(gap), float(overshoot.mean()), float(overshoot.max())


def shape_model(mesh: trimesh.Trimesh) -> coal.BVHModelOBBRSS:
    """``mesh`` as a coal collision model."""
    model = coal.BVHModelOBBRSS()
    model.beginModel(len(mesh.faces), len(mesh.vertices))
    model.addVertices(np.asarray(mesh.vertices, dtype=np.float64))
    model.addTriangles(np.asarray(mesh.faces, dtype=np.int64))
    model.endModel()
    return model


def shape_check(
    model: Model, shapes: dict[str, coal.BVHModelOBBRSS], robot: collision.RobotSpheres, pairs: torch.Tensor
) -> dict[tuple[str, str], str]:
    """The ignore reason of every frame pair of ``pairs`` (sphere pairs ``(P, 2)`` of ``robot``): ``"never"``,
    ``"always"``, ``"neutral"`` or ``""`` (kept), from the collisions of ``shapes`` (per frame name) at the neutral
    configuration and at ``CONFIGURATIONS`` random ones. ``robot`` holds the shapes, so a pair whose spheres are 5 mm
    apart is not checked."""
    names = [model.frame_names[frame] for frame in robot.frame_ids.tolist()]
    keys = sorted({tuple(sorted((names[i], names[j]))) for i, j in pairs.tolist()})
    column = torch.tensor([keys.index(tuple(sorted((names[i], names[j])))) for i, j in pairs.tolist()])
    generator = torch.Generator().manual_seed(SEED)
    q = torch.stack(
        [model.q_neutral, *(model.random_configuration(generator=generator) for _ in range(CONFIGURATIONS))]
    )
    hits = torch.zeros(len(q), len(keys), dtype=torch.bool)
    request = coal.CollisionRequest()
    for start in range(0, len(q), 1000):
        state = forward_kinematics(model, q[start : start + 1000])
        distance = collision.self_distances(robot, state, pairs=pairs)
        nearest = torch.full((len(distance), len(keys)), math.inf, dtype=distance.dtype)
        nearest = nearest.scatter_reduce(1, column.expand_as(distance), distance, "amin")
        rotations, translations = state.frame_rotations.numpy(), state.frame_translations.numpy()
        for row, key in (nearest < 0.005).nonzero().tolist():
            a, b = (model.frame_id(name) for name in keys[key])
            result = coal.CollisionResult()
            coal.collide(
                shapes[keys[key][0]],
                coal.Transform3s(rotations[row, a], translations[row, a]),
                shapes[keys[key][1]],
                coal.Transform3s(rotations[row, b], translations[row, b]),
                request,
                result,
            )
            hits[start + row, key] = result.isCollision()
    collided = hits[1:].sum(0).tolist()
    return {
        key: "always" if count == CONFIGURATIONS else "neutral" if hits[0, k] else "never" if count == 0 else ""
        for k, (key, count) in enumerate(zip(keys, collided, strict=True))
    }


def surface_distance(shape: coal.BVHModelOBBRSS, points: torch.Tensor) -> torch.Tensor:
    """``(M,)``: how far each of ``points (M, 3)`` lies from the surface of ``shape``."""
    probe, request, origin, distances = coal.Sphere(1e-9), coal.DistanceRequest(), coal.Transform3s(), []
    for point in points.numpy():
        result = coal.DistanceResult()
        coal.distance(probe, coal.Transform3s(np.eye(3), point), shape, origin, request, result)
        distances.append(result.min_distance)
    return torch.tensor(distances, dtype=points.dtype)


def swap(body: str) -> str:
    """``body`` with its ``left_`` and ``right_`` prefix exchanged."""
    side = {"left_": "right_", "right_": "left_"}
    return next((side[prefix] + body[len(prefix) :] for prefix in side if body.startswith(prefix)), body)


def fit_body(
    body: str,
    partners: set[str],
    rows: dict[str, list],
    model: Model,
    meshes: dict,
    arguments: argparse.Namespace,
    *,
    sole: bool,
) -> list[list[float]]:
    """The rounded rows of ``body``; its spheres stop short of the ``partners`` and, with ``sole``, of the floor.
    With ``--mirror`` a right body mirrors the left one in ``rows``."""
    if arguments.mirror and body.startswith("right_"):
        return [[x, -y, z, radius] for x, y, z, radius in rows[swap(body)]]
    mesh = meshes[body]
    if arguments.mirror and body.startswith("left_"):
        mesh = trimesh.util.concatenate([mesh, meshes[swap(body)].copy().apply_transform(np.diag([1.0, -1, 1, 1]))])
    caps = []
    if sole:
        low = float(mesh.bounds[0][2])
        caps.append(lambda centres: centres[:, 2] - low + SOLE)
    neutral = forward_kinematics(model, model.q_neutral)
    for partner in partners:
        # The partner's shapes at the neutral configuration, in this body's frame; a sphere reaches REACH of the gap.
        pose = np.eye(4), np.eye(4)
        for transform, frame in zip(pose, (body, partner), strict=True):
            transform[:3, :3] = neutral.frame_rotations[model.frame_id(frame)].numpy()
            transform[:3, 3] = neutral.frame_translations[model.frame_id(frame)].numpy()
        shape = shape_model(meshes[partner].copy().apply_transform(np.linalg.solve(*pose)))
        gap = coal.DistanceResult()
        coal.distance(
            shape_model(meshes[body]), coal.Transform3s(), shape, coal.Transform3s(), coal.DistanceRequest(), gap
        )
        keep = (1 - REACH) * gap.min_distance
        caps.append(lambda centres, shape=shape, keep=keep: surface_distance(shape, centres) - keep)
    cap = (lambda centres: torch.stack([each(centres) for each in caps]).amin(0)) if caps else None
    symmetric = arguments.mirror and not body.startswith(("left_", "right_"))
    return rounded(
        *fit(mesh, arguments.count.get(body, arguments.default), arguments.samples, symmetric=symmetric, cap=cap)
    )


def separate(
    rows: dict[str, list], ignore: list[tuple[str, str]], model: Model, meshes: dict, arguments: argparse.Namespace
) -> tuple[collision.RobotSpheres, torch.Tensor, dict[str, set[str]]]:
    """Refit ``rows`` in place until every kept pair is clear at the neutral configuration: the spheres, the kept sphere
    pairs ``(P, 2)`` and each fitted body's partners."""
    neutral = forward_kinematics(model, model.q_neutral)
    partners: dict[str, set[str]] = defaultdict(set)
    refit = set(arguments.sole)
    while True:
        # Refit the bodies with new partners (their sides swapped where --mirror fits the left body), then check again.
        for body in meshes:
            if body in refit or swap(body) in refit:
                rows[body] = fit_body(body, partners[body], rows, model, meshes, arguments, sole=body in arguments.sole)
        robot = spheres_of(model, rows)
        kept = collision.self_collision_pairs(model, robot, ignore=ignore)
        names = [model.frame_names[frame] for frame in robot.frame_ids.tolist()]
        distance = collision.self_distances(robot, neutral, pairs=kept).tolist()
        refit = set()
        for (i, j), value in zip(kept.tolist(), distance, strict=True):
            for body, partner in ((names[i], names[j]), (names[j], names[i])) if value < 0 else ():
                fitted, other = (
                    (swap(body), swap(partner)) if arguments.mirror and body.startswith("right_") else (body, partner)
                )
                if other not in partners[fitted]:
                    partners[fitted].add(other)
                    refit.add(fitted)
        if not refit:
            break
    # What the caps leave (a sphere bulging beside the gap, the radius rounding): both radii give up half the overlap.
    owners = [
        (swap(body) if arguments.mirror and body.startswith("right_") else body, k)
        for body, entries in rows.items()
        for k in range(len(entries))
    ]
    while min(distance, default=0) < 0:
        for (i, j), overlap in zip(kept.tolist(), distance, strict=True):
            for body, k in (owners[i], owners[j]) if overlap < 0 else ():
                rows[body][k][3] = min(rows[body][k][3], math.floor((rows[body][k][3] + overlap / 2) * 1e4) / 1e4)
        for body in meshes:
            if arguments.mirror and body.startswith("right_"):
                rows[body] = fit_body(body, set(), rows, model, meshes, arguments, sole=False)
        robot = spheres_of(model, rows)
        distance = collision.self_distances(robot, neutral, pairs=kept).tolist()
    return robot, kept, partners


def spheres_of(model: Model, rows: dict[str, list]) -> collision.RobotSpheres:
    """The spheres of ``rows`` (``{frame: [[x, y, z, radius], ...]}``) on ``model``."""
    return collision.RobotSpheres.from_frames(
        model, {frame: [(row[:3], row[3]) for row in entries] for frame, entries in rows.items()}
    )


def main() -> int:
    """Fit, check, report and write; the exit code is 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("description", help="a URDF or MJCF file, or a robot_descriptions module name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--default", type=int, default=2, help="spheres per body not named by --count")
    parser.add_argument("--count", action="append", default=[], metavar="BODY=N", help="spheres of one body; 0 skips")
    parser.add_argument("--visual", action="append", default=[], metavar="BODY", help="fit BODY's display shapes")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="fit each left_ body with its right_ twin mirrored (y to -y), the right body gets the mirror image, and "
        "every other body is symmetric about y = 0",
    )
    parser.add_argument("--sole", action="append", default=[], metavar="BODY", help="BODY stands on the floor")
    parser.add_argument("--samples", type=int, default=4000, help="surface points per body for the fit")
    parser.add_argument("--source", default="", help="where the description comes from, stored in the file")
    arguments = parser.parse_args()
    arguments.count = {body: int(count) for body, count in (item.split("=") for item in arguments.count)}
    description = Path(arguments.description)
    if not description.suffix:
        description = Path(importlib.import_module(f"robot_descriptions.{arguments.description}").URDF_PATH)
    model = io.load(description, dtype=torch.float64)
    geometry = io.load_geometry(description, model)
    meshes = {
        body: mesh
        for body in model.body_names[1:]
        if arguments.count.get(body, arguments.default) > 0
        and (mesh := body_mesh(geometry, model.body_id(body), description.parent, visual=body in arguments.visual))
    }
    rows: dict[str, list] = {}
    for body in meshes:  # spheres that hold the shapes, so that shape_check may skip pairs whose spheres are apart
        rows[body] = fit_body(body, set(), rows, model, meshes, arguments, sole=False)
    robot = spheres_of(model, rows)
    pairs = collision.self_collision_pairs(model, robot)
    reasons = shape_check(model, {body: shape_model(mesh) for body, mesh in meshes.items()}, robot, pairs)
    if (
        arguments.mirror
    ):  # a pair and its mirror image share a status: neutral if either is, else kept unless both agree
        twins = {key: reasons[tuple(sorted(swap(name) for name in key))] for key in reasons}
        reasons = {
            key: "neutral" if "neutral" in (reason, twins[key]) else reason if reason == twins[key] else ""
            for key, reason in reasons.items()
        }
    robot, kept, partners = separate(rows, [key for key, reason in reasons.items() if reason], model, meshes, arguments)

    out = sys.stdout
    out.write(f"{'body':28s} {'spheres':>7s} {'gap mm':>7s} {'over mean':>9s} {'over max':>8s}  stops short of\n")
    for body, entries in rows.items():
        gap, mean, worst = measure(meshes[body], torch.tensor(entries, dtype=torch.float64), arguments.samples)
        stops = ", ".join(
            sorted(
                partners[body] | {swap(partner) for partner in partners[swap(body)]}
                if arguments.mirror
                else partners[body]
            )
        )
        out.write(f"{body:28s} {len(entries):7d} {1e3 * gap:7.2f} {1e3 * mean:9.1f} {1e3 * worst:8.1f}  {stops}\n")
    tally = {reason: sum(value == reason for value in reasons.values()) for reason in ("never", "always", "neutral")}
    distance = collision.self_distances(robot, forward_kinematics(model, model.q_neutral), pairs=kept)
    out.write(
        f"{len(rows)} bodies, {robot.frame_ids.shape[0]} spheres, {kept.shape[0]} of {pairs.shape[0]} sphere pairs kept; "
        f"ignored frame pairs {tally}; closest kept pair at neutral {1e3 * float(distance.min()):.2f} mm\n"
    )
    command = "uv run --extra meshes --extra demos --group test-oracles python tools/fit_spheres.py"
    ignore = [[*key, reason] for key, reason in sorted(reasons.items()) if reason]
    write(
        arguments.output,
        source=arguments.source,
        command=f"{command} {shlex.join(sys.argv[1:])}",
        spheres=rows,
        ignore=ignore,
    )
    return 0


def write(path: Path, *, source: str, command: str, spheres: dict[str, list], ignore: list[list[str]]) -> None:
    """The sphere file: one sphere per line, so a change reads as a line diff."""
    frames = ",\n".join(
        f"    {json.dumps(frame)}: [\n" + ",\n".join(f"      {json.dumps(row)}" for row in rows) + "\n    ]"
        for frame, rows in spheres.items()
    )
    pairs = ",\n".join(f"    {json.dumps(entry)}" for entry in ignore)
    check = (
        f"coal on the fitted shapes at the neutral configuration and at {CONFIGURATIONS} random configurations within "
        f"the joint limits (torch generator seed {SEED}): an ignored pair collides in none of them (never), in all the "
        "random ones (always) or at the neutral one (neutral)"
    )
    path.write_text(
        f'{{\n  "source": {json.dumps(source)},\n  "command": {json.dumps(command)},\n'
        f'  "shape_check": {json.dumps(check)},\n  "spheres": {{\n{frames}\n  }},\n  "ignore": [\n{pairs}\n  ]\n}}\n'
    )


if __name__ == "__main__":
    sys.exit(main())
