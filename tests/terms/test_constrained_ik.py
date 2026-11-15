"""Inverse kinematics under clearance rows: ``SequentialQP`` on a Panda that avoids a sphere and itself."""

import math

import pytest
import torch

import better_robot as br
from better_robot import collision, optim, terms
from better_robot.kinematics import forward_kinematics
from tests.support.robots import panda_urdf

FRAME = "panda_hand"
MARGIN = 0.02
OBSTACLE = (0.45, 0.25, 0.45, 0.10)  # centre and radius of a sphere beside the arm
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]


@pytest.fixture(scope="module")
def scene():
    """``(build, worst, reached)``: ``build(batch, seed)`` is the IK problem of ``batch`` random targets from the
    ready pose with the clearances as inequality rows, the target poses and their configurations;
    ``worst(q): (B,)`` is the largest violation over every row of the nonlinear problem (all sphere pairs, not the
    closest eight; the joint limits); ``reached(q, goal): (B,)`` is the benchmark's 1 mm and 0.01 rad."""
    panda = br.load(panda_urdf(), dtype=torch.float64)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", panda)
    pairs = collision.self_collision_pairs(panda, spheres, ignore=ignore)
    obstacle = torch.tensor(OBSTACLE, dtype=torch.float64)

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]

    def build(batch: int, seed: int) -> tuple[optim.Problem, torch.Tensor, torch.Tensor]:
        q0 = torch.tensor(READY, dtype=torch.float64).expand(batch, -1)
        generator = torch.Generator().manual_seed(seed)
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
    assert int(reached(q, goal).sum()) >= 45  # 47 measured, 44 of them among the 52 clear targets


@pytest.mark.slow
def test_a_compiled_step_equals_the_eager_one(scene) -> None:
    build, _, _ = scene
    solver = optim.SequentialQP()
    step = br.kinematics.compiled(solver.step)
    problem, _, _ = build(4, 1)
    compiled, eager = (optim.run(problem, [optim.Phase(solver, 6, stop_every=0, step=hook)]) for hook in (step, None))
    torch.testing.assert_close(compiled.x["q"], eager.x["q"], rtol=0.0, atol=1e-9)
    torch.testing.assert_close(compiled.loss, eager.loss, rtol=0.0, atol=1e-12)
