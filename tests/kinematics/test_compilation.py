"""`kinematics.compiled`: one graph from a model's first call, results equal to eager, CUDA-graph results kept."""

from __future__ import annotations

import pytest
import torch
from torch._dynamo.utils import counters

from better_robot import load
from better_robot.dynamics import forward_dynamics
from better_robot.exceptions import ArgumentError
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import compiled, forward_kinematics
from tests.support.robots import panda_urdf


def _free_flyer():
    builder = ModelBuilder("free_flyer_entry_point")
    builder.add_body("base", mass=1.0)
    builder.add_body("arm", mass=0.5)
    builder.add_free_flyer_root("floating", child="base", origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    builder.add_revolute("elbow", parent="base", child="arm", axis=(0.0, 0.0, 1.0), lower=-2.0, upper=2.0)
    return build_model(builder.finalize())


def test_a_cold_model_compiles_one_graph_equal_to_eager():
    """The first call builds the plans and memos eagerly, so the trace reads them: one graph, no recompile (a plain
    ``torch.compile`` of a cold model traces the plan building, then recompiles)."""
    model = _free_flyer()
    generator = torch.Generator().manual_seed(0)
    q = torch.stack([model.random_configuration(generator=generator) for _ in range(3)])
    fk = compiled(lambda q: forward_kinematics(model, q).joint_transforms)
    torch._dynamo.reset()
    counters.clear()
    with torch.no_grad():
        first, second = fk(q), fk(q)
        expected = forward_kinematics(model, q).joint_transforms
    assert counters["stats"]["unique_graphs"] == 1
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


def test_an_unknown_mode_raises():
    with pytest.raises(ArgumentError, match="mode"):
        compiled(lambda q: q, mode="max-autotune")


@pytest.mark.cuda
def test_reduce_overhead_captures_forward_dynamics_and_keeps_each_result():
    """CUDA graphs capture forward dynamics (no host sync), and a result stays valid after the next replay, which
    overwrites the graph's own outputs without an error: plain tensors and the tensors of a dataclass result, whose
    model stays the caller's."""
    model = load(panda_urdf(), dtype=torch.float32).to(device=torch.device("cuda"))
    generator = torch.Generator().manual_seed(0)
    q = torch.stack([model.random_configuration(generator=generator) for _ in range(64)]).cuda()
    v, tau = (torch.randn(64, model.nv, generator=generator).cuda() for _ in range(2))
    aba = compiled(
        lambda q, v, tau: forward_dynamics(model, forward_kinematics(model, q), v=v, tau=tau), mode="reduce-overhead"
    )
    fk = compiled(lambda q: forward_kinematics(model, q), mode="reduce-overhead")
    torch._dynamo.reset()
    counters.clear()
    scales = (1.0, 2.0, 3.0)
    with torch.no_grad():
        accelerations = [aba(q, v, scale * tau) for scale in scales]
        states = [fk(scale * q) for scale in scales]
        expected = [forward_dynamics(model, forward_kinematics(model, q), v=v, tau=scale * tau) for scale in scales]
        expected_transforms = [forward_kinematics(model, scale * q).joint_transforms for scale in scales]
    assert counters["inductor"]["cudagraph_skips"] == 0
    for result, reference in zip(accelerations, expected, strict=True):
        # float32 ABA: compiled and eager differ by up to 1e-4 of the largest acceleration; an overwritten result by 1.
        torch.testing.assert_close(result, reference, rtol=0.0, atol=1e-3 * float(reference.abs().max()))
    for state, reference in zip(states, expected_transforms, strict=True):
        assert state.model is model
        torch.testing.assert_close(state.joint_transforms, reference, rtol=0.0, atol=1e-5)
