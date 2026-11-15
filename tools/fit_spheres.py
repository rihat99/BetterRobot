"""Fit spheres to the collision shapes of a robot description and write the file ``collision.load_spheres`` reads.

Per body, in the body's frame, at most ``--count`` spheres hold the body's shapes (``fit``): a greedy cover whose
overshoot past the surface is the smallest a bisection finds, then each radius shrinks to the points it holds with
the most room, and a sample ten times denser widens it again where a point fell between the fitted ones. Centres
are rounded to 0.1 mm and radii rounded up with 0.1 mm added, which keeps every fitted point inside. The ignore list holds the
frame pairs that ``self_collision_pairs`` keeps and whose spheres overlap at the model's neutral configuration; the
report gives the share of random configurations in which each of them overlaps.

Measured per body on a third sample the fit never saw: the gap, how far a surface point lies outside every sphere,
and the overshoot, how far the surface of the union of spheres lies from the nearest surface point.

    uv run --extra meshes python tools/fit_spheres.py DESCRIPTION OUTPUT --default 2 --count panda_link0=3 \\
        --visual left_rubber_hand --mirror --source "where the description comes from"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

from better_robot import collision, forward_kinematics, io
from better_robot.model import Box, Capsule, Cylinder, Geometry, Mesh, Sphere


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


def fit(mesh: trimesh.Trimesh, count: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Centres ``(K, 3)`` and radii ``(K,)`` of at most ``count`` spheres that hold ``mesh``.

    A candidate centre ``c`` lies in the convex hull or on the surface, at depth ``d(c)`` from the nearest surface
    point; with an overshoot ``delta`` its sphere has radius ``d(c) + delta``, so it reaches at most ``delta`` past the
    surface of a solid that holds ``c``. The targets are surface points and, for a closed mesh, points inside it. A
    bisection finds the smallest ``delta`` for which the greedy cover (the sphere holding the most uncovered targets,
    in turn) holds every target with ``count`` spheres.
    """
    surface = points_of(mesh, samples, seed=0)[-samples:]
    box = np.random.default_rng(0).uniform(*mesh.bounds, size=(samples, 3))
    hull = box[mesh.convex_hull.contains(box)]
    solid = hull[mesh.contains(hull)] if mesh.is_watertight else hull[:0]
    targets = torch.cat([surface, torch.from_numpy(solid)])
    candidates = torch.cat([torch.from_numpy(hull), surface[::4]])
    depth = torch.cdist(candidates, surface).min(1).values
    reach = torch.cdist(candidates, targets) - depth[:, None]
    low, high = 0.0, float(reach.max())
    for _ in range(30):
        middle = (low + high) / 2
        low, high = (low, middle) if greedy_cover(reach <= middle, count) else (middle, high)
    chosen = torch.tensor(greedy_cover(reach <= high, count))
    centres, radii = candidates[chosen], depth[chosen] + high
    for _ in range(20):
        # Each target to the sphere that holds it with the most room; each radius shrinks to its targets.
        distance = torch.cdist(targets, centres)
        member = (distance - radii).argmin(1)[:, None] == torch.arange(len(radii))
        radii = distance.where(member, 0).amax(0)
    return centres, radii


def greedy_cover(holds: torch.Tensor, count: int) -> list[int]:
    """Rows of ``holds (M, T)`` (candidate ``m`` holds target ``t``) picked greedily until every target is held; empty
    when ``count`` rows do not suffice."""
    uncovered, chosen = torch.ones(holds.shape[1], dtype=torch.bool), []
    while uncovered.any() and len(chosen) < count:
        best = int((holds & uncovered).sum(1).argmax())
        chosen.append(best)
        uncovered &= ~holds[best]
    return [] if uncovered.any() else chosen


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


def write(path: Path, *, source: str, command: str, spheres: dict[str, list], ignore: list[tuple[str, str]]) -> None:
    """The sphere file: one sphere per line, so a change reads as a line diff."""
    frames = ",\n".join(
        f"    {json.dumps(frame)}: [\n" + ",\n".join(f"      {json.dumps(row)}" for row in rows) + "\n    ]"
        for frame, rows in spheres.items()
    )
    pairs = ",\n".join(f"    {json.dumps(list(pair))}" for pair in ignore)
    path.write_text(
        f'{{\n  "source": {json.dumps(source)},\n  "command": {json.dumps(command)},\n  "spheres": {{\n{frames}\n  }},\n'
        f'  "ignore": [\n{pairs}\n  ]\n}}\n'
    )


def main() -> int:
    """Fit, report and write; the exit code is 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("description", type=Path, help="a URDF or MJCF file whose meshes resolve beside it")
    parser.add_argument("output", type=Path)
    parser.add_argument("--default", type=int, default=2, help="spheres per body not named by --count")
    parser.add_argument("--count", action="append", default=[], metavar="BODY=N", help="spheres of one body; 0 skips")
    parser.add_argument("--visual", action="append", default=[], metavar="BODY", help="fit BODY's display shapes")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="fit each left_ body with its right_ twin mirrored (y to -y); the right body gets the mirror image",
    )
    parser.add_argument("--samples", type=int, default=4000, help="surface points per body for the fit")
    parser.add_argument("--source", default="", help="where the description comes from, stored in the file")
    arguments = parser.parse_args()
    counts = {body: int(count) for body, count in (item.split("=") for item in arguments.count)}
    model = io.load(arguments.description, dtype=torch.float64)
    geometry = io.load_geometry(arguments.description, model)

    def shapes(body: str) -> trimesh.Trimesh | None:
        return body_mesh(geometry, model.body_id(body), arguments.description.parent, visual=body in arguments.visual)

    spheres: dict[str, list] = {}
    out = sys.stdout
    out.write(f"{'body':28s} {'spheres':>7s} {'gap mm':>7s} {'over mean':>9s} {'over max':>8s}\n")
    for body in model.body_names[1:]:
        count, mesh = counts.get(body, arguments.default), shapes(body)
        if mesh is None or count == 0:
            continue
        if arguments.mirror and body.startswith("right_"):
            rows = [[x, -y, z, radius] for x, y, z, radius in spheres[body.replace("right_", "left_", 1)]]
        else:
            if arguments.mirror and body.startswith("left_"):
                twin = shapes(body.replace("left_", "right_", 1))
                mesh = trimesh.util.concatenate([mesh, twin.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))])
            centres, radii = fit(mesh, count, arguments.samples)
            # Points between the fitted samples can lie just outside: a sample ten times denser widens the radii.
            dense = torch.cdist(points_of(mesh, 10 * arguments.samples, seed=1), centres)
            radii = (
                dense.where((dense - radii).argmin(1)[:, None] == torch.arange(len(radii)), 0).amax(0).maximum(radii)
            )
            rows = rounded(centres, radii)
        gap, mean, worst = measure(shapes(body), torch.tensor(rows, dtype=torch.float64), arguments.samples)
        out.write(f"{body:28s} {len(rows):7d} {1e3 * gap:7.2f} {1e3 * mean:9.1f} {1e3 * worst:8.1f}\n")
        spheres[body] = rows

    robot = collision.RobotSpheres.from_frames(
        model, {frame: [(row[:3], row[3]) for row in rows] for frame, rows in spheres.items()}
    )
    pairs = collision.self_collision_pairs(model, robot)
    neutral = collision.self_distances(robot, forward_kinematics(model, model.q_neutral), pairs=pairs)
    ignore = sorted(
        {
            tuple(sorted((model.frame_names[robot.frame_ids[i]], model.frame_names[robot.frame_ids[j]])))
            for (i, j), distance in zip(pairs.tolist(), neutral.tolist(), strict=True)
            if distance < 0
        }
    )
    generator = torch.Generator().manual_seed(0)
    q = torch.stack([model.random_configuration(generator=generator) for _ in range(2000)])
    sampled = collision.self_distances(robot, forward_kinematics(model, q), pairs=pairs) < 0
    out.write(f"{len(spheres)} bodies, {robot.frame_ids.shape[0]} spheres, {pairs.shape[0]} pairs\n")
    out.write("ignored (overlap at neutral), share of 2000 random configurations where they overlap:\n")
    names = [model.frame_names[frame] for frame in robot.frame_ids.tolist()]
    for pair in ignore:
        mask = torch.tensor([tuple(sorted((names[i], names[j]))) == pair for i, j in pairs.tolist()])
        out.write(f"  {pair[0]} - {pair[1]}: {float(sampled[:, mask].any(1).double().mean()):.3f}\n")

    command = " ".join(
        [
            "uv run --extra meshes python tools/fit_spheres.py",
            arguments.description.name,
            arguments.output.name,
            f"--default {arguments.default}",
            *(f"--count {item}" for item in arguments.count),
            *(f"--visual {body}" for body in arguments.visual),
            *(["--mirror"] if arguments.mirror else []),
            f"--samples {arguments.samples}",
        ]
    )
    write(arguments.output, source=arguments.source, command=command, spheres=spheres, ignore=ignore)
    return 0


if __name__ == "__main__":
    sys.exit(main())
