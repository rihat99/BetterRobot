"""Inverse kinematics under clearance rows: ``SequentialQP`` on a Panda that avoids a sphere and itself."""

import dataclasses
import math

import pytest
import torch
from torch._dynamo.utils import counters

import better_robot as br
from better_robot import collision, optim, terms
from better_robot.kinematics import forward_kinematics
from tests.support.robots import panda_urdf

FRAME = "panda_hand"
MARGIN = 0.02
OBSTACLE = (0.45, 0.25, 0.45, 0.10)  # centre and radius of a sphere beside the arm
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]
# Torch's CUDA-graph tree captures an empty graph on purpose to keep its memory pool alive, a warning it records but
# the "error" filter raises (as in tests/kinematics/test_compilation.py).
CUDA_GRAPH_KEEPS_ITS_POOL = "ignore:The CUDA Graph is empty:UserWarning"


def _scene(device: torch.device):
    """``(build, worst, reached)`` on ``device``: ``build(batch, seed)`` is the IK problem of ``batch`` random targets
    from the ready pose with the clearances as inequality rows, the target poses and their configurations;
    ``worst(q): (B,)`` is the largest violation over every row of the nonlinear problem (all sphere pairs, not the
    closest eight; the joint limits); ``reached(q, goal): (B,)`` is the benchmark's 1 mm and 0.01 rad."""
    panda = br.load(panda_urdf(), dtype=torch.float64, device=device)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", panda)
    pairs = collision.self_collision_pairs(panda, spheres, ignore=ignore)
    obstacle = torch.tensor(OBSTACLE, dtype=torch.float64, device=device)

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]

    def build(batch: int, seed: int) -> tuple[optim.Problem, torch.Tensor, torch.Tensor]:
        q0 = torch.tensor(READY, dtype=torch.float64, device=device).expand(batch, -1)
        generator = torch.Generator(device).manual_seed(seed)
        q_target = terms.random_start(panda, panda.q_neutral.expand(batch, -1), generator=generator)
        goal = forward_kinematics(panda, q_target).frame_poses[:, panda.frame_id(FRAME)]
        problem = terms.ik_problem(panda, {FRAME: goal}, q0=q0)
        avoid = terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN, hinge=False)
        apart = terms.self_clearance(spheres, pairs=pairs, margin=MARGIN, hinge=False, closest=8)
        problem.inequality("obstacle", avoid, upper=0.0)
        problem.inequality("self", apart, upper=0.0)
        return problem, goal, q_target

    def worst(q: torch.Tensor) -> torch.Tensor:
        state = forward_kinematics(panda, q)
        distances = [to_obstacle(spheres.centers(state), spheres.radii)]
        distances.append(collision.self_distances(spheres, state, pairs=pairs))
        limits = torch.maximum(panda.lower_position_limit - q, q - panda.upper_position_limit).amax(-1)
        return torch.maximum((MARGIN - torch.cat(distances, dim=-1)).amax(-1), limits).clamp_min(0.0)

    def reached(q: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        pose = forward_kinematics(panda, q).frame_poses[:, panda.frame_id(FRAME)]
        turn = 2.0 * torch.acos((pose[:, 3:] * goal[:, 3:]).sum(-1).abs().clamp(max=1.0))
        return ((pose[:, :3] - goal[:, :3]).norm(dim=-1) < 1e-3) & (turn < 1e-2)

    return build, worst, reached


@pytest.fixture(scope="module")
def scene():
    return _scene(torch.device("cpu"))


def test_every_row_holds_at_the_end_of_a_constrained_ik_run(scene) -> None:
    """A run of a fixed number of steps from a clear start ends within 1e-4 of every row, also for the targets whose
    configuration breaks one (12 of these 64), and reaches most poses."""
    build, worst, reached = scene
    problem, goal, q_target = build(64, 0)
    assert int((worst(q_target) > 0).sum()) == 12
    assert float(worst(problem.variables["q"]).max()) == 0.0
    result = optim.run(problem, [optim.Phase(optim.SequentialQP(), 50, stop_every=0)])
    q = result.x["q"]
    assert float(worst(q).max()) <= 1e-4
    assert float(result.diagnostics["phase"]["violation"].max()) <= 1e-4
    assert int(reached(q, goal).sum()) >= 45  # 46 measured, 44 of them among the 52 clear targets


def _tensors(value: object, path: str = "state") -> dict[str, torch.Tensor]:
    """``{path: tensor}`` of every tensor a state holds, through its dataclasses, dicts and tuples."""
    if isinstance(value, torch.Tensor):
        return {path: value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}
    elif isinstance(value, tuple):
        value = dict(enumerate(value))
    if not isinstance(value, dict):
        return {}
    return {name: found for key, item in value.items() for name, found in _tensors(item, f"{path}.{key}").items()}


def _check_compiled_step(scene, mode: str) -> None:
    """Over the whole state: the reported point, the iterate, its rows and normal equations, the quadratic program and
    the step control."""
    build, _, _ = scene
    solver = optim.SequentialQP()
    step = br.kinematics.compiled(solver.step, mode=mode)
    problem, _, _ = build(4, 1)
    compiled, eager = (optim.run(problem, [optim.Phase(solver, 6, stop_every=0, step=hook)]) for hook in (step, None))
    found, expected = _tensors(compiled.state), _tensors(eager.state)
    assert found.keys() == expected.keys() and {"state.extra.radius", "state.extra.qp.dual"} <= found.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(
            found[name], tensor, rtol=1e-9, atol=1e-9, msg=lambda text, name=name: f"{name}: {text}"
        )


@pytest.mark.slow
def test_a_compiled_step_equals_the_eager_one(scene) -> None:
    _check_compiled_step(scene, "default")


@pytest.mark.cuda
@pytest.mark.filterwarnings(CUDA_GRAPH_KEEPS_ITS_POOL)
def test_a_step_compiled_with_cuda_graphs_equals_the_eager_one() -> None:
    """``mode="reduce-overhead"`` adds CUDA graphs, so its twin of the step above runs on a GPU (release tier)."""
    _check_compiled_step(_scene(torch.device("cuda")), "reduce-overhead")


def _tracking(device: torch.device):
    """``(problem, path, worst)`` on ``device``: 24 Panda targets whose paths from the ready pose cross a joint limit, a floor 5 cm
    up, the obstacle sphere, the arm itself or a singular pose (each crossed by some), and the five behaviours as rows
    of ``optim.track``; ``path(t)`` is ``{frame: poses (24, 7)}`` and its configurations at tick ``t`` of 100;
    ``worst(q)`` is ``{behaviour: (24,) largest violation over every row}``."""
    panda = br.load(panda_urdf(), dtype=torch.float64, device=device)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", panda)
    pairs = collision.self_collision_pairs(panda, spheres, ignore=ignore)
    start = torch.tensor(READY, dtype=torch.float64, device=device)
    obstacle = torch.tensor(OBSTACLE, dtype=torch.float64, device=device)
    base = torch.tensor(
        [int(f) in {panda.frame_id("panda_link0"), panda.frame_id("panda_link1")} for f in spheres.frame_ids],
        device=device,
    )
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)

    def to_floor(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:  # the base spheres never reach it
        return collision.sphere_halfspace(centers, radii, normal=up, offset=0.05) + base

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]

    rows = {
        "floor": terms.sphere_clearance(spheres, to_floor, margin=MARGIN, hinge=False, closest=8),
        "obstacle": terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN, hinge=False, closest=8),
        "self": terms.self_clearance(spheres, pairs=pairs, margin=MARGIN, hinge=False, closest=8),
        "singular": terms.manipulability(panda, frame=FRAME, threshold=0.03, hinge=False),
    }

    def worst(q: torch.Tensor) -> dict[str, torch.Tensor]:
        state = forward_kinematics(panda, q)
        out = {"limits": torch.maximum(panda.lower_position_limit - q, q - panda.upper_position_limit).amax(-1)}
        out["floor"] = (MARGIN - to_floor(spheres.centers(state), spheres.radii)).amax(-1)
        out["obstacle"] = (MARGIN - to_obstacle(spheres.centers(state), spheres.radii)).amax(-1)
        out["self"] = (MARGIN - collision.self_distances(spheres, state, pairs=pairs)).amax(-1)
        out["singular"] = rows["singular"].fn(state)[:, 0]
        return {name: value.clamp_min(0.0) for name, value in out.items()}

    mid = (panda.lower_position_limit + panda.upper_position_limit) / 2
    draws = terms.random_start(panda, start.expand(512, -1), generator=torch.Generator(device).manual_seed(3))
    goal = mid + 1.2 * (draws - mid)  # some beyond a limit
    crossed = torch.stack(
        [torch.stack(list(worst(start + s * (goal - start)).values())) for s in (0.25, 0.5, 0.75, 1.0)]
    )
    crossed = crossed.amax(0) > 0  # (5, 512)
    keep = torch.cat([torch.nonzero(crossed[b]).flatten()[:5] for b in range(5)])[:24]
    goal = goal[keep]
    hand = panda.frame_id(FRAME)

    def path(tick: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        share = min(tick / 80, 1.0)
        q = start + (3 - 2 * share) * share * share * (goal - start)
        return {FRAME: forward_kinematics(panda, q).frame_poses[:, hand]}, q

    problem = terms.ik_problem(panda, path(0)[0], q0=start.expand(24, -1).contiguous())
    for name, row in rows.items():
        problem.inequality(name, row, upper=0.0)
    return problem, path, worst


@pytest.fixture(scope="module")
def tracking():
    return _tracking(torch.device("cpu"))


def test_each_behaviour_row_holds_while_track_follows_targets_through_the_forbidden_regions(tracking) -> None:
    """Steps of at most 5 mrad keep every row of the five behaviours within 1e-4 over the whole trajectory (the
    measured region), while the targets' own configurations break each behaviour; most clear targets are reached."""
    problem, path, worst = tracking
    x, warm, seen, broken = dict(problem.variables), None, {}, {}
    for tick in range(1, 101):
        poses, q_target = path(tick)
        moved = problem.with_(data={f"target/{FRAME}": poses[FRAME]})
        x, warm = optim.track(moved, x, warm=warm, step=0.005)
        for name, value in worst(x["q"]).items():
            seen[name] = max(seen.get(name, 0.0), float(value.max()))
        for name, value in worst(q_target).items():
            broken[name] = broken.get(name, 0) + int((value > 1e-3).sum())
    assert all(count > 0 for count in broken.values()), broken
    assert max(seen.values()) <= 1e-4, seen


def test_a_tick_reads_nothing_back(tracking) -> None:
    """A control loop of ``track`` calls with the five behaviours copies no tensor value to the host."""
    problem, path, _ = tracking
    x, warm = optim.track(problem, dict(problem.variables), step=0.005)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for tick in range(1, 4):
            moved = problem.with_(data={f"target/{FRAME}": path(tick)[0][FRAME]})
            x, warm = optim.track(moved, x, warm=warm, step=0.005)
    assert sum(event.name in ("aten::_local_scalar_dense", "aten::nonzero") for event in profile.events()) == 0


def _check_compiled_tick(tracking, mode: str) -> None:
    """Over six ticks: the configuration and the whole quadratic program's solution."""
    problem, path, _ = tracking
    tick = br.kinematics.compiled(lambda problem, x, warm: optim.track(problem, x, warm=warm, step=0.005), mode=mode)
    runs = []
    for step in (tick, lambda problem, x, warm: optim.track(problem, x, warm=warm, step=0.005)):
        x, warm = dict(problem.variables), None
        for t in range(1, 7):
            x, warm = step(problem.with_(data={f"target/{FRAME}": path(t)[0][FRAME]}), x, warm)
        runs.append(_tensors((x, warm)))
    assert runs[0].keys() == runs[1].keys() and "state.1.dual" in runs[0]
    for name, tensor in runs[1].items():
        torch.testing.assert_close(
            runs[0][name], tensor, rtol=1e-9, atol=1e-9, msg=lambda text, name=name: f"{name}: {text}"
        )


@pytest.mark.slow
def test_a_compiled_tick_equals_the_eager_one(tracking) -> None:
    _check_compiled_tick(tracking, "default")


@pytest.mark.cuda
@pytest.mark.filterwarnings(CUDA_GRAPH_KEEPS_ITS_POOL)
def test_a_tick_compiled_with_cuda_graphs_equals_the_eager_one() -> None:
    """The twin of the tick above with ``mode="reduce-overhead"``, on a GPU (release tier)."""
    _check_compiled_tick(_tracking(torch.device("cuda")), "reduce-overhead")


def test_a_start_deep_inside_a_margin_comes_out_without_a_stall() -> None:
    """A Panda that starts with three forearm spheres 26 to 30 mm inside an obstacle's margin, more than one step of
    6.3 mrad can repair, comes out a little on every call and is clear within 40 (28 measured), while its hand holds
    its pose. (Scaled back alone, the step was given up whole from the second call on: clamped to the box, it took
    one of the violated rows further out.)"""
    panda = br.load(panda_urdf(), dtype=torch.float64)
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "panda.json", panda)
    start = torch.tensor([READY], dtype=torch.float64)
    state = forward_kinematics(panda, start)
    forearm = int(torch.nonzero(spheres.frame_ids == panda.frame_id("panda_link4"))[3])
    lift = spheres.radii[forearm] + 0.05 + MARGIN - 0.03
    obstacle = torch.cat([spheres.centers(state)[0, forearm], torch.tensor([0.05], dtype=torch.float64)])
    obstacle[2] += lift

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]

    def inside(q: torch.Tensor) -> float:
        return float((MARGIN - to_obstacle(spheres.centers(forward_kinematics(panda, q)), spheres.radii)).max())

    problem = terms.ik_problem(panda, {FRAME: state.frame_poses[:, panda.frame_id(FRAME)]}, q0=start)
    clearance = terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN, hinge=False, closest=8)
    problem.inequality("obstacle", clearance, upper=0.0)
    x, warm, depth = dict(problem.variables), None, [inside(start)]
    for _ in range(40):
        x, warm = optim.track(problem, x, warm=warm, step=0.0063)
        depth.append(inside(x["q"]))
    assert depth[0] > 0.029 and depth[-1] <= 0.0, depth
    assert all(after < before - 2e-5 for before, after in zip(depth, depth[1:], strict=False) if before > 0), depth


@pytest.mark.slow
def test_an_obstacle_moved_in_place_needs_no_new_problem_and_compiles_once() -> None:
    """The obstacle's distance function reads a tensor that the loop moves in place before each tick, 1 mm per tick
    through the hand, which holds its pose (a new problem and target tensor every tick). The compiled tick compiles no
    graph after its first ones and equals the eager tick; the hand gives way, and a tick ends at most ``v / gain`` (2 mm) inside the margin of where the obstacle
    has moved, or within 1e-4 of it when the loop pads the radius by that much."""
    panda = br.load(panda_urdf(), dtype=torch.float64)
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "panda.json", panda)
    start = torch.tensor(READY, dtype=torch.float64).expand(2, -1).contiguous()
    hand = panda.frame_id(FRAME)
    goal = forward_kinematics(panda, start).frame_poses[:, hand]
    obstacle = torch.zeros(4, dtype=torch.float64)

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]

    def place(tick: int, pad: float = 0.0) -> None:
        sweep = torch.tensor([0.0, tick / 1000 - 0.3, 0.0, 0.0], dtype=torch.float64)
        obstacle.copy_(torch.cat([goal[0, :3], torch.tensor([0.04 + pad], dtype=torch.float64)]) + sweep)

    problem = terms.ik_problem(panda, {FRAME: goal}, q0=start)
    clearance = terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN, hinge=False, closest=8)
    problem.inequality("obstacle", clearance, upper=0.0)

    def eager(problem: optim.Problem, x: dict, warm: optim.QPSolution | None) -> tuple:
        return optim.track(problem, x, warm=warm, step=0.005)

    def run(tick, pad: float = 0.0) -> tuple[torch.Tensor, float, float]:
        x, warm, deepest, moved = dict(problem.variables), None, 0.0, 0.0
        for t in range(1, 601):
            place(t, pad)
            x, warm = tick(problem.with_(data={f"target/{FRAME}": goal.clone()}), x, warm)
            if t == 3:  # the eager calls of both keys and the compile are behind
                counters.clear()
            place(t + 1)
            state = forward_kinematics(panda, x["q"])
            deepest = max(deepest, float((MARGIN - to_obstacle(spheres.centers(state), spheres.radii)).max()))
            moved = max(moved, float((state.frame_poses[:, hand, :3] - goal[:, :3]).norm(dim=-1).max()))
        return x["q"], deepest, moved

    found, _, _ = run(br.kinematics.compiled(eager))
    assert counters["stats"]["unique_graphs"] == 0
    expected, deepest, moved = run(eager)
    torch.testing.assert_close(found, expected, rtol=1e-9, atol=1e-9)
    assert moved > 0.05 and 1e-3 < deepest <= 2e-3 + 1e-4, (moved, deepest)
    assert run(eager, pad=0.002)[1] <= 1e-4
