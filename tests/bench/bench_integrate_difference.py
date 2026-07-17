"""Advisory CPU benchmark for grouped versus per-joint manifold operations."""

from __future__ import annotations

import platform

import pytest
import torch

pytest.importorskip("pytest_benchmark")

from better_robot.io.builders.smpl_like import make_smpl_like_model


pytestmark = pytest.mark.bench

WARMUP_ROUNDS = 10
MEASURED_ROUNDS = 50
TRAJECTORY_LENGTH = 200
SEED = 20250717


def _loop_integrate(model, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for joint_id, joint in enumerate(model.joint_models):
        nq_joint = model.nqs[joint_id]
        if nq_joint == 0:
            continue
        iq = model.idx_qs[joint_id]
        iv = model.idx_vs[joint_id]
        parts.append(
            joint.integrate(
                q[..., iq : iq + nq_joint],
                v[..., iv : iv + model.nvs[joint_id]],
            )
        )
    return torch.cat(parts, dim=-1) if parts else q.clone()


def _loop_difference(model, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for joint_id, joint in enumerate(model.joint_models):
        nq_joint = model.nqs[joint_id]
        if nq_joint == 0:
            continue
        iq = model.idx_qs[joint_id]
        parts.append(
            joint.difference(
                q0[..., iq : iq + nq_joint],
                q1[..., iq : iq + nq_joint],
            )
        )
    return torch.cat(parts, dim=-1) if parts else q0.new_zeros(*q0.shape[:-1], model.nv)


@pytest.fixture(scope="module")
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def smpl_manifold_case(single_thread_cpu):
    del single_thread_cpu
    model = make_smpl_like_model(dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    tangent = (
        torch.randn(
            TRAJECTORY_LENGTH,
            model.nv,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.05
    )
    neutral = model.q_neutral.expand(TRAJECTORY_LENGTH, -1).clone()
    q0 = _loop_integrate(model, neutral, tangent * 0.25)
    q1 = _loop_integrate(model, q0, tangent)
    return model, q0, q1, tangent


@pytest.mark.parametrize(
    ("operation", "implementation"),
    (
        pytest.param("integrate", "loop", id="integrate-loop"),
        pytest.param("integrate", "grouped", id="integrate-grouped"),
        pytest.param("difference", "loop", id="difference-loop"),
        pytest.param("difference", "grouped", id="difference-grouped"),
    ),
)
def test_smpl_manifold_throughput(
    benchmark,
    smpl_manifold_case,
    operation: str,
    implementation: str,
) -> None:
    """Record all four distributions; speedup is advisory, never ratio-gated."""

    model, q0, q1, tangent = smpl_manifold_case
    if operation == "integrate":
        function = (
            (lambda: _loop_integrate(model, q0, tangent))
            if implementation == "loop"
            else (lambda: model.integrate(q0, tangent))
        )
        expected_shape = (TRAJECTORY_LENGTH, model.nq)
    else:
        function = (
            (lambda: _loop_difference(model, q0, q1))
            if implementation == "loop"
            else (lambda: model.difference(q0, q1))
        )
        expected_shape = (TRAJECTORY_LENGTH, model.nv)

    result = benchmark.pedantic(
        function,
        iterations=1,
        rounds=MEASURED_ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
    )
    benchmark.extra_info.update(
        {
            "operation": operation,
            "implementation": implementation,
            "device": "cpu",
            "dtype": "torch.float32",
            "model": "SMPL-like free-flyer + 23 spherical joints",
            "q_shape": [TRAJECTORY_LENGTH, model.nq],
            "v_shape": [TRAJECTORY_LENGTH, model.nv],
            "torch_threads": 1,
            "warmup_rounds": WARMUP_ROUNDS,
            "measured_rounds": MEASURED_ROUNDS,
            "iterations_per_round": 1,
            "statistic": "pytest-benchmark median over measured rounds",
            "platform_machine": platform.machine(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
        }
    )
    assert result.shape == expected_shape
