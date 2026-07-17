"""CUDA-only validation of the opt-in Warp FK bridge."""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("warp")

from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.kinematics._warp_bridge import (
    _warp_fk_forward,
    try_warp_forward_kinematics,
)
from better_robot.kinematics.forward import (
    forward_kinematics_raw,
    frame_placements_raw,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _model_and_q():
    device = torch.device("cuda:0")
    model = make_smpl_like_model(dtype=torch.float32).to(device=device)
    return model, model.q_neutral.unsqueeze(0).contiguous()


def _torch_outputs(model, q):
    world, local = forward_kinematics_raw(model.structure, model.values, q)
    frames = frame_placements_raw(model.structure, model.values, world)
    return world, local, frames


def _assert_close(actual, expected) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-5, atol=2e-6)


def _direct_inputs(model, q):
    structure = model.structure
    values = model.values
    device = q.device
    q_map = torch.zeros(q.shape[0], dtype=torch.int32, device=device)
    value_map = torch.zeros_like(q_map)
    return (
        q,
        values.joint_placements.reshape(1, model.njoints, 7),
        values.frame_placements.reshape(1, model.nframes, 7),
        q_map,
        value_map,
        structure.parents_tensor,
        structure.topo_order_tensor,
        structure.joint_kind_tensor,
        structure.nqs_tensor,
        structure.idx_qs_tensor,
        structure.joint_axes,
        structure.joint_pitches,
        structure.frame_parent_joints,
    )


def test_cuda_forward_and_q_gradient_match_torch() -> None:
    model, q_data = _model_and_q()
    q = q_data.requires_grad_(True)
    result = try_warp_forward_kinematics(model.structure, model.values, q)
    assert result is not None
    actual = (result.world, result.local, result.frames)
    expected = _torch_outputs(model, q)
    _assert_close(actual, expected)

    actual_loss = sum(tensor.square().sum() for tensor in actual)
    actual_grad = torch.autograd.grad(actual_loss, q)[0]
    q_ref = q_data.detach().clone().requires_grad_(True)
    expected_outputs = _torch_outputs(model, q_ref)
    expected_loss = sum(tensor.square().sum() for tensor in expected_outputs)
    expected_grad = torch.autograd.grad(expected_loss, q_ref)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)


def test_cuda_current_stream_and_graph_replay() -> None:
    model, q_neutral = _model_and_q()
    structure, values = model.structure, model.values

    # Capture the direct functional op after its one-time Warp compilation.
    q_static = q_neutral.clone()
    inputs = _direct_inputs(model, q_static)
    _warp_fk_forward(*inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _warp_fk_forward(*inputs)

    q_static[:, 7:10].add_(0.01)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(captured, _torch_outputs(model, q_static))

    # A delayed update on a non-default stream makes a default-stream Warp
    # launch race and read the old configuration.  Correct stream interop
    # serializes the update and kernel without a host synchronization.
    sleep = getattr(torch.cuda, "_sleep", None)
    if sleep is None:
        pytest.skip("torch.cuda._sleep is unavailable")
    stream = torch.cuda.Stream()
    q_stream = q_neutral.clone()
    target = q_neutral.clone()
    target[:, 7:10].add_(0.02)
    with torch.cuda.stream(stream):
        sleep(5_000_000)
        q_stream.copy_(target)
        result = try_warp_forward_kinematics(structure, values, q_stream)
        assert result is not None
    stream.synchronize()
    _assert_close((result.world, result.local, result.frames), _torch_outputs(model, target))
