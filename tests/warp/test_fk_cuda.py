"""CUDA-only validation of the opt-in Warp FK bridge."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch


pytest.importorskip("warp")

from better_robot.io.build_model import build_model
from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.io.parsers.programmatic import ModelBuilder
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


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _make_branched_model(dtype: torch.dtype):
    builder = ModelBuilder("warp_cuda_branched")
    for name in ("root", "left", "right", "tip"):
        builder.add_body(name)
    builder.add_revolute_z(
        "left_joint",
        parent="root",
        child="left",
        origin=_pose(0.2, 0.0, 0.1),
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_prismatic_x(
        "right_joint",
        parent="root",
        child="right",
        origin=_pose(0.0, 0.3, 0.1),
        lower=-0.5,
        upper=0.5,
    )
    builder.add_revolute_y(
        "tip_joint",
        parent="left",
        child="tip",
        origin=_pose(0.4, 0.0, 0.0),
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_frame("tool", parent_body="tip", placement=_pose(0.1, 0.0, 0.0))
    return build_model(builder.finalize(), dtype=dtype)


def _make_deep_model(dtype: torch.dtype, *, depth: int = 18):
    builder = ModelBuilder("warp_cuda_deep")
    builder.add_body("root")
    parent = "root"
    for index in range(depth):
        child = f"link_{index}"
        builder.add_body(child)
        builder.add_revolute_z(
            f"joint_{index}",
            parent=parent,
            child=child,
            origin=_pose(0.05, 0.0, 0.01),
            lower=-math.pi,
            upper=math.pi,
        )
        parent = child
    builder.add_frame("deep_tip", parent_body=parent, placement=_pose(0.03, 0.0, 0.0))
    return build_model(builder.finalize(), dtype=dtype)


def _model_and_q(kind: str = "smpl", dtype: torch.dtype = torch.float32):
    device = torch.device("cuda:0")
    if kind == "smpl":
        model = make_smpl_like_model(dtype=dtype)
    elif kind == "branched":
        model = _make_branched_model(dtype)
    elif kind == "deep":
        model = _make_deep_model(dtype)
        assert max(len(support) for support in model.supports) > 16
    else:  # pragma: no cover - tests below enumerate every supported case
        raise AssertionError(f"unknown CUDA test model {kind!r}")
    model = model.to(device=device)
    tangent = torch.linspace(-0.03, 0.03, model.nv, dtype=dtype, device=device)
    q = model.integrate(model.q_neutral, tangent).unsqueeze(0).contiguous()
    return model, q


def _torch_outputs(model, q, values=None):
    values = model.values if values is None else values
    world, local = forward_kinematics_raw(model.structure, values, q)
    frames = frame_placements_raw(model.structure, values, world)
    return world, local, frames


def _assert_close(actual, expected) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-5, atol=2e-6)


def _assert_close_dtype(actual, expected, dtype: torch.dtype) -> None:
    rtol, atol = (5e-5, 5e-6) if dtype == torch.float32 else (2e-9, 2e-10)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


def _weighted_loss(outputs) -> torch.Tensor:
    result = outputs[0].new_zeros(())
    for scale, tensor in zip((0.7, -0.4, 0.2), outputs, strict=True):
        weight = torch.linspace(
            0.1,
            0.9,
            tensor.numel(),
            dtype=tensor.dtype,
            device=tensor.device,
        ).reshape(tensor.shape)
        result = result + scale * (weight * tensor).sum()
    return result


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


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
@pytest.mark.parametrize("model_kind", ("branched", "deep"))
def test_cuda_model_dtype_q_and_placement_gradient_matrix(
    model_kind: str,
    dtype: torch.dtype,
) -> None:
    """Exercise generic branching/depth with every differentiable FK input."""
    model, q_data = _model_and_q(model_kind, dtype)

    q = q_data.detach().clone().requires_grad_()
    placements = model.values.joint_placements.detach().clone().requires_grad_()
    frames = model.values.frame_placements.detach().clone().requires_grad_()
    values = dataclasses.replace(
        model.values,
        joint_placements=placements,
        frame_placements=frames,
    )
    result = try_warp_forward_kinematics(model.structure, values, q)
    assert result is not None
    actual = (result.world, result.local, result.frames)
    actual_gradients = torch.autograd.grad(_weighted_loss(actual), (q, placements, frames))

    q_ref = q_data.detach().clone().requires_grad_()
    placements_ref = model.values.joint_placements.detach().clone().requires_grad_()
    frames_ref = model.values.frame_placements.detach().clone().requires_grad_()
    values_ref = dataclasses.replace(
        model.values,
        joint_placements=placements_ref,
        frame_placements=frames_ref,
    )
    expected = _torch_outputs(model, q_ref, values_ref)
    expected_gradients = torch.autograd.grad(
        _weighted_loss(expected),
        (q_ref, placements_ref, frames_ref),
    )

    _assert_close_dtype(actual, expected, dtype)
    _assert_close_dtype(actual_gradients, expected_gradients, dtype)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
def test_cuda_multi_axis_shared_value_gradient_reduction(dtype: torch.dtype) -> None:
    """Broadcast ``q=(2,1)`` with values ``=(1,3)`` and reduce shared VJPs."""
    model, q_seed = _model_and_q("branched", dtype)
    tangent = torch.linspace(-0.06, 0.06, model.nv, dtype=dtype, device=q_seed.device)
    q_other = model.integrate(model.q_neutral, tangent).unsqueeze(0)
    q_data = torch.cat((q_seed, q_other), dim=0).unsqueeze(1).contiguous()

    placement_data = model.values.joint_placements[None, None].repeat(1, 3, 1, 1)
    placement_data[:, 1, :, 0].add_(0.01)
    placement_data[:, 2, :, 1].sub_(0.02)
    frame_data = model.values.frame_placements[None, None].repeat(1, 3, 1, 1)
    frame_data[:, 1, :, 2].add_(0.02)
    frame_data[:, 2, :, 0].sub_(0.01)

    q = q_data.detach().clone().requires_grad_()
    placements = placement_data.detach().clone().requires_grad_()
    frames = frame_data.detach().clone().requires_grad_()
    values = dataclasses.replace(
        model.values,
        joint_placements=placements,
        frame_placements=frames,
    )
    result = try_warp_forward_kinematics(model.structure, values, q)
    assert result is not None
    actual = (result.world, result.local, result.frames)
    actual_gradients = torch.autograd.grad(_weighted_loss(actual), (q, placements, frames))

    q_ref = q_data.detach().clone().requires_grad_()
    placements_ref = placement_data.detach().clone().requires_grad_()
    frames_ref = frame_data.detach().clone().requires_grad_()
    values_ref = dataclasses.replace(
        model.values,
        joint_placements=placements_ref,
        frame_placements=frames_ref,
    )
    world_ref, local_ref = forward_kinematics_raw(model.structure, values_ref, q_ref)
    expected = (
        world_ref,
        local_ref,
        frame_placements_raw(model.structure, values_ref, world_ref),
    )
    expected_gradients = torch.autograd.grad(
        _weighted_loss(expected),
        (q_ref, placements_ref, frames_ref),
    )

    assert actual[0].shape[:2] == (2, 3)
    assert actual_gradients[0].shape == q_data.shape
    assert actual_gradients[1].shape == placement_data.shape
    assert actual_gradients[2].shape == frame_data.shape
    _assert_close_dtype(actual, expected, dtype)
    _assert_close_dtype(actual_gradients, expected_gradients, dtype)


def test_cuda_gradcheck_at_zero_for_q_and_placements() -> None:
    """Numerically check the CUDA VJP at the zero-angle singular seam."""
    model, _ = _model_and_q("branched", torch.float64)
    q = torch.zeros((1, model.nq), dtype=torch.float64, device="cuda:0", requires_grad=True)
    placements = model.values.joint_placements.detach().clone().requires_grad_()
    frames = model.values.frame_placements.detach().clone().requires_grad_()

    def function(q_input, placement_input, frame_input):
        values = dataclasses.replace(
            model.values,
            joint_placements=placement_input,
            frame_placements=frame_input,
        )
        result = try_warp_forward_kinematics(model.structure, values, q_input)
        assert result is not None
        return _weighted_loss((result.world, result.local, result.frames))

    assert torch.autograd.gradcheck(
        function,
        (q, placements, frames),
        eps=1e-6,
        atol=5e-5,
        rtol=5e-4,
        fast_mode=True,
    )


@pytest.mark.parametrize(
    ("model_kind", "dtype"),
    (
        pytest.param("smpl", torch.float32, id="smpl-fp32"),
        pytest.param("branched", torch.float32, id="branched-fp32"),
        pytest.param("branched", torch.float64, id="branched-fp64"),
        pytest.param("deep", torch.float32, id="deep-fp32"),
        pytest.param("deep", torch.float64, id="deep-fp64"),
    ),
)
def test_cuda_current_stream_and_graph_replay(model_kind: str, dtype: torch.dtype) -> None:
    model, q_neutral = _model_and_q(model_kind, dtype)
    structure, values = model.structure, model.values

    # Capture the direct functional op after its one-time Warp compilation.
    q_static = q_neutral.clone()
    inputs = _direct_inputs(model, q_static)
    _warp_fk_forward(*inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _warp_fk_forward(*inputs)

    replay_delta = torch.linspace(
        -0.01,
        0.01,
        model.nv,
        dtype=dtype,
        device=q_neutral.device,
    )
    replay_target = model.integrate(q_neutral, replay_delta)
    q_static.copy_(replay_target)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close_dtype(captured, _torch_outputs(model, q_static), dtype)

    # A delayed update on a non-default stream makes a default-stream Warp
    # launch race and read the old configuration.  Correct stream interop
    # serializes the update and kernel without a host synchronization.
    sleep = getattr(torch.cuda, "_sleep", None)
    if sleep is None:
        pytest.skip("torch.cuda._sleep is unavailable")
    stream = torch.cuda.Stream()
    q_stream = q_neutral.clone()
    stream_delta = torch.linspace(
        0.02,
        -0.02,
        model.nv,
        dtype=dtype,
        device=q_neutral.device,
    )
    target = model.integrate(q_neutral, stream_delta)
    with torch.cuda.stream(stream):
        sleep(5_000_000)
        q_stream.copy_(target)
        result = try_warp_forward_kinematics(structure, values, q_stream)
        assert result is not None
    stream.synchronize()
    _assert_close_dtype(
        (result.world, result.local, result.frames),
        _torch_outputs(model, target),
        dtype,
    )
