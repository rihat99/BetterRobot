"""Trajectory optimization around an obstacle: the Panda is a set of spheres, the obstacle is a sphere.

The straight joint-space path between the two end configurations sweeps the hand through the
obstacle. A hinge clearance term (zero when clear) pushes the path out as a soft penalty; the same
clearance as the inequality ``margin - d <= 0`` gives the hard-constrained solve.

    uv run python examples/06_obstacle_trajectory.py
"""

from __future__ import annotations

import math
import time

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import collision, optim, terms
from better_robot.kinematics import forward_kinematics_state

KNOTS, DT, MARGIN = 30, 0.05, 0.02
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]
SPHERES = {
    "body_panda_link3": [((0.0, 0.0, -0.05), 0.08)],
    "body_panda_link4": [((0.0, 0.0, 0.0), 0.08)],
    "body_panda_link5": [((0.0, 0.05, -0.12), 0.07), ((0.0, 0.0, -0.22), 0.07)],
    "body_panda_link7": [((0.0, 0.0, 0.05), 0.07)],
    "body_panda_hand": [((0.0, 0.0, 0.06), 0.07), ((0.0, 0.0, 0.11), 0.05)],
}


def main() -> None:
    torch.set_num_threads(4)
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    spheres = collision.RobotSpheres.from_frames(model, SPHERES)
    q_a, q_b = torch.tensor(READY, dtype=torch.float64), torch.tensor(READY, dtype=torch.float64)
    q_a[0], q_b[0] = -0.9, 0.9  # swing the base; halfway the hand is where the obstacle sits
    hand = forward_kinematics_state(model, torch.tensor(READY, dtype=torch.float64)).frame_translation
    obstacle, radius = hand[model.frame_id("body_panda_hand")].clone(), torch.tensor(0.08, dtype=torch.float64)

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return collision.sphere_sphere(centers, radii, obstacle, radius)

    def clearance(q: torch.Tensor) -> float:
        return float(to_obstacle(spheres.centers(forward_kinematics_state(model, q)), spheres.radii).min())

    alpha = torch.linspace(0, 1, KNOTS, dtype=torch.float64)[:, None]
    straight = (1 - alpha) * q_a + alpha * q_b
    ends = straight.clone()[None]
    end_weight = torch.zeros(KNOTS, 1, dtype=torch.float64)
    end_weight[0] = end_weight[-1] = 1e3

    problem = optim.Problem(
        {"q": straight[None].clone()},
        manifolds={"q": terms.RobotManifold(model)},
        bounds={"q": (model.lower_pos_limit, model.upper_pos_limit)},
        data={"ends": ends},
        time=("q",),
    )
    problem.cached(terms.kinematic_state(model), name="state")
    problem.residual("ends", lambda q, ends: q - ends, weight=end_weight, window=1)
    problem.residual("velocity", terms.finite_difference(model, order=1, dt=DT), weight=1e-2)
    problem.residual("acceleration", terms.finite_difference(model, order=2, dt=DT), weight=1e-4)
    problem.residual("clearance", terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN), weight=1e2, window=1)
    problem.residual("keep_out", terms.sphere_clearance(spheres, to_obstacle, margin=MARGIN, hinge=False), window=1)

    print(f"straight path: minimum clearance {clearance(straight):+.3f} m (negative = penetration)")
    runs = {
        "soft penalty (LM)": lambda: optim.solve(
            problem.with_(weights={"keep_out": 0.0}), optim.LevenbergMarquardt(), iters=200
        ),
        "hard inequality": lambda: optim.solve_constrained(
            problem.with_(weights={"clearance": 0.0}), optim.LevenbergMarquardt(), inequalities=("keep_out",)
        ),
    }
    for label, run in runs.items():
        start = time.perf_counter()
        result = run()
        seconds = time.perf_counter() - start
        q = result.x["q"][0]
        length = float((q[1:] - q[:-1]).norm(dim=-1).sum())
        end_error = float(torch.maximum((q[0] - q_a).abs().max(), (q[-1] - q_b).abs().max()))
        print(f"{label:18s}: minimum clearance {clearance(q):+.3f} m, joint path length {length:.3f} rad "
              f"(straight {float((q_b - q_a).norm()):.3f}), end error {end_error:.1e} rad, "
              f"{result.iters} LM steps in {seconds * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
