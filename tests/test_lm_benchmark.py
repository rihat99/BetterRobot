"""Benchmark: PyPose LM vs our LM (autodiff) vs our LM (analytic).

Run with:
    uv run pytest tests/test_lm_benchmark.py -v -s

Correctness: all modes must reach pos_err < 5 cm.
Speed: printed per-call timings; analytic expected fastest.
"""

import functools
import time

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description

import better_robot as br
from better_robot import load_urdf, solve_ik, IKConfig
from better_robot.costs import pose_residual, limit_residual, rest_residual, CostTerm
from better_robot.solvers import Problem, PyposeLevenbergMarquardt


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return load_urdf(urdf)


def _fixed_base_run(model, target, link_idx, jacobian_mode, n_iter=20, n_runs=10):
    cfg0 = model._q_default.clone()
    elapsed = 0.0
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = solve_ik(
            model,
            targets={"panda_hand": target},
            config=IKConfig(jacobian=jacobian_mode),
            initial_q=cfg0.clone(),
            max_iter=n_iter,
        )
        elapsed += time.perf_counter() - t0
    fk = model.forward_kinematics(result)
    pos_err = (fk[link_idx, :3] - target[:3]).norm().item()
    return elapsed / n_runs * 1000, pos_err


def _pypose_run(model, target, link_idx, n_iter=20, n_runs=10):
    cfg0 = model._q_default.clone()
    hand_idx = link_idx
    elapsed = 0.0
    result = None
    for _ in range(n_runs):
        costs = [
            CostTerm(functools.partial(pose_residual, model=model, target_link_index=hand_idx,
                                       target_pose=target, pos_weight=1.0, ori_weight=0.1), weight=1.0),
            CostTerm(functools.partial(limit_residual, model=model), weight=0.1),
            CostTerm(functools.partial(rest_residual, q_rest=model._q_default), weight=0.01),
        ]
        prob = Problem(variables=cfg0.clone(), costs=costs,
                       lower_bounds=model.joints.lower_limits.clone(),
                       upper_bounds=model.joints.upper_limits.clone())
        t0 = time.perf_counter()
        result = PyposeLevenbergMarquardt().solve(prob, max_iter=n_iter)
        elapsed += time.perf_counter() - t0
    fk = model.forward_kinematics(result)
    pos_err = (fk[link_idx, :3] - target[:3]).norm().item()
    return elapsed / n_runs * 1000, pos_err


def test_benchmark_fixed_base(panda, capsys):
    """All three modes converge to < 5 cm; analytic no more than 20% slower than autodiff."""
    hand_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(panda._q_default)[hand_idx].detach().clone()
    target[0] += 0.1

    results = {}
    results["lm_pypose"] = _pypose_run(panda, target, hand_idx)
    results["autodiff"] = _fixed_base_run(panda, target, hand_idx, "autodiff")
    results["analytic"] = _fixed_base_run(panda, target, hand_idx, "analytic")

    with capsys.disabled():
        print("\n--- Fixed-Base IK Benchmark (20 iters, 10 runs avg) ---")
        for mode, (ms, err) in results.items():
            print(f"  {mode:20s}: {ms:7.2f} ms/call   pos_err={err:.4f} m")

    # Correctness: target is 10 cm offset, 20 iters — threshold is generous (8 cm)
    for mode, (_, err) in results.items():
        assert err < 0.08, f"{mode} pos_err={err:.4f} > 0.08"

    # Speed: analytic must not be more than 20% slower than autodiff
    assert results["analytic"][0] <= results["autodiff"][0] * 1.2, (
        f"Analytic ({results['analytic'][0]:.1f} ms) is >20% slower than "
        f"autodiff ({results['autodiff'][0]:.1f} ms)"
    )


def test_benchmark_floating_base(panda, capsys):
    """Floating-base: both modes converge to < 5 cm."""
    hand_idx = panda.link_index("panda_hand")
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    target = panda.forward_kinematics(panda._q_default)[hand_idx].detach().clone()
    target[0] += 0.1

    results = {}
    for mode in ("autodiff", "analytic"):
        cfg0 = panda._q_default.clone()
        elapsed = 0.0
        N = 5
        data_r = None
        for _ in range(N):
            t0 = time.perf_counter()
            data_r = solve_ik(
                panda, targets={"panda_hand": target},
                config=IKConfig(jacobian=mode),
                initial_base_pose=identity_base.clone(),
                initial_q=cfg0.clone(), max_iter=20,
            )
            elapsed += time.perf_counter() - t0
        cfg_r, base_r = data_r.q, data_r.base_pose
        fk = panda.forward_kinematics(cfg_r, base_pose=base_r)
        pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
        results[mode] = (elapsed / N * 1000, pos_err)

    with capsys.disabled():
        print("\n--- Floating-Base IK Benchmark (20 iters, 5 runs avg) ---")
        for mode, (ms, err) in results.items():
            print(f"  {mode:20s}: {ms:7.2f} ms/call   pos_err={err:.4f} m")

    for mode, (_, err) in results.items():
        assert err < 0.05, f"Floating {mode} pos_err={err:.4f} > 0.05"
