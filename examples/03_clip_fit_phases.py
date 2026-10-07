"""Fit a floating-base humanoid to noisy marker tracks over a clip, in phases.

This is the shape of a human-motion fit: a root-only warm-up with Adam, a first-order
polish with batched L-BFGS, then Levenberg-Marquardt on the block-tridiagonal system.
The terms are the same in every phase; a phase only changes weights, the free
coordinates and the solver.

    uv run python examples/03_clip_fit_phases.py
"""

from __future__ import annotations

import time

import torch
from robot_descriptions import g1_description

import better_robot as br
from better_robot import optim, terms
from better_robot.kinematics import forward_kinematics_state

KNOTS, FPS = 120, 30.0


def main() -> None:
    torch.set_num_threads(8)
    model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
    markers = [n for n in model.frame_names if any(k in n for k in ("pelvis", "ankle_roll", "wrist_yaw", "head"))][:6]
    marker_ids = [model.frame_id(n) for n in markers]

    # Synthetic clip: a walking-like reference, markers with noise and a few gross outliers.
    generator = torch.Generator().manual_seed(0)
    phase = torch.linspace(0, 6.28, KNOTS, dtype=torch.float64)[:, None]
    q_true = model.q_neutral.repeat(KNOTS, 1)
    q_true[:, 7:] += 0.4 * torch.sin(phase + torch.arange(model.nq - 7, dtype=torch.float64))
    q_true[:, 0] += 0.5 * phase[:, 0] / 6.28
    true_positions = forward_kinematics_state(model, q_true[None]).frame_pose[:, :, marker_ids, :3]  # (1, T, M, 3)
    observed = true_positions + 0.005 * torch.randn(true_positions.shape, generator=generator, dtype=torch.float64)
    outliers = torch.rand(true_positions.shape[:-1], generator=generator) < 0.02
    observed[outliers] += 0.5

    problem = optim.Problem(
        {"q": model.q_neutral.repeat(1, KNOTS, 1)},
        manifolds={"q": terms.RobotManifold(model)},
        data={"observed": observed},
        time=("q",),
    )
    problem.cached(terms.kinematic_state(model), name="state")
    # One custom term for all markers; the last axis (xyz of one marker) is one robust group.
    problem.residual(
        "markers",
        lambda state, observed: state.frame_pose[:, :, marker_ids, :3] - observed,
        kernel=optim.Huber(delta=0.03),
        window=1,
    )
    problem.residual("smooth", terms.finite_difference(model, order=2, dt=1 / FPS), weight=1e-5)
    problem.residual("rest", lambda q: q[..., 7:] - model.q_neutral[7:], weight=1e-3, window=1)

    root_only = torch.zeros(KNOTS, model.nv, dtype=torch.bool)
    root_only[:, :6] = True
    phases = [
        optim.Phase(optim.LevenbergMarquardt(), 5, name="root", free={"q": root_only.flatten()}, weights={"smooth": 0.0}),
        optim.Phase(optim.LBFGS(), 30, name="lbfgs"),
        optim.Phase(optim.LevenbergMarquardt(), 15, name="lm"),
    ]
    start = time.perf_counter()
    result = optim.run(problem, phases)
    seconds = time.perf_counter() - start

    fitted = forward_kinematics_state(model, result.x["q"]).frame_pose[:, :, marker_ids, :3]
    error = (fitted - true_positions).norm(dim=-1)
    for name, losses in result.history.items():
        print(f"phase {name:6s}: loss {float(losses[0].mean()):.4f} -> {float(losses[-1].mean()):.4f} in {len(losses)} steps")
    print(f"{KNOTS} frames, {KNOTS * model.nv} unknowns, {seconds:.1f} s")
    print(f"marker error against the clean reference: mean {float(error.mean()) * 1e3:.1f} mm, "
          f"max {float(error.max()) * 1e3:.1f} mm (noise 5 mm, 2% outliers of 0.5 m)")


if __name__ == "__main__":
    main()
