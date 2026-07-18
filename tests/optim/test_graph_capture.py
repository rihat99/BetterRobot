"""Reusable GraphExecutor lifecycle and named-block LM CUDA certification."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import gc
import importlib
from typing import Any

import pytest
import torch
from torch.utils._pytree import tree_flatten

from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.kinematics.forward import forward_kinematics_raw, frame_placements_raw
from better_robot.optim import LevenbergMarquardt, Problem, ResidualItem, VarSpec
from better_robot.optim._graph_executor import GraphCaptureError, GraphExecutor


_CUDA_AVAILABLE = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def test_cpu_and_disabled_execution_are_eager_and_preserve_autograd() -> None:
    calls = 0

    def function(value: torch.Tensor, *, bias: torch.Tensor) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"value": value.square() + bias, "summary": (value.sum(),)}

    value = torch.tensor([1.0, -2.0], requires_grad=True)
    bias = torch.tensor([0.2, 0.3])
    enabled = GraphExecutor(function)
    first = enabled(value, bias=bias)
    disabled = GraphExecutor(function, enabled=False)
    second = disabled(value, bias=bias)

    assert calls == 2
    assert not enabled.is_captured
    assert enabled.record_count == enabled.replay_count == 0
    assert not disabled.is_captured
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        torch.autograd.grad(first["value"].sum(), value)[0],
        2.0 * value,
    )


@pytest.mark.cuda
@_CUDA_AVAILABLE
@pytest.mark.parametrize(
    ("unsafe_kind", "message"),
    (
        ("storage_offset", "storage_offset"),
        ("zero_stride", "stride-zero"),
        ("internal_overlap", "internally overlapping"),
        ("leaf_alias", "overlapping storage"),
    ),
)
def test_capture_rejects_unsafe_input_views_before_allocating_static_buffers(
    unsafe_kind: str,
    message: str,
) -> None:
    base = torch.arange(16, dtype=torch.float32, device="cuda")
    if unsafe_kind == "storage_offset":
        inputs = (base[1:9],)
    elif unsafe_kind == "zero_stride":
        inputs = (base[:3].reshape(1, 3).expand(4, 3),)
    elif unsafe_kind == "internal_overlap":
        inputs = (base.as_strided((2, 2), (1, 1)),)
    else:
        inputs = (base, base.view(4, 4))

    executor = GraphExecutor(lambda *values: values[0] + 1.0, warmup_runs=1)
    with pytest.raises(ValueError, match=message):
        executor(*inputs)

    assert executor.record_count == 0
    assert executor.input_buffer_ptrs == ()


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_capture_rejects_outputs_requiring_grad_from_a_closure_parameter() -> None:
    parameter = torch.tensor(2.0, device="cuda", requires_grad=True)
    executor = GraphExecutor(lambda value: value * parameter, warmup_runs=1)

    with pytest.raises(GraphCaptureError, match="outputs must not require gradients"):
        executor(torch.ones(4, device="cuda"))

    assert not executor.is_captured
    assert executor.record_count == 0


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_cuda_lazy_record_copy_replay_clone_contract_and_reset() -> None:
    device = torch.device("cuda")

    def function(value: torch.Tensor, bias: torch.Tensor) -> dict[str, Any]:
        return {
            "value": torch.sin(value) + bias,
            "summary": (value.square().sum(dim=-1),),
        }

    executor = GraphExecutor(function, warmup_runs=2)
    first_input = torch.linspace(-0.5, 0.8, 12, device=device).reshape(3, 4)
    first_bias = torch.full_like(first_input, 0.2)
    disabled = GraphExecutor(function, enabled=False)
    disabled_output = disabled(first_input, first_bias)
    assert not disabled.is_captured
    assert disabled.record_count == 0
    torch.testing.assert_close(disabled_output, function(first_input, first_bias))

    first = executor(first_input, first_bias)
    first_snapshot = {"value": first["value"].clone(), "summary": (first["summary"][0].clone(),)}
    pointers = executor.input_buffer_ptrs

    second_input = torch.linspace(0.7, -0.3, 12, device=device).reshape(3, 4)
    second_bias = torch.full_like(second_input, -0.1)
    second = executor(second_input, second_bias)

    assert executor.is_captured
    assert executor.record_count == 1
    assert executor.replay_count == 1
    assert executor.last_record_ms is not None and executor.last_record_ms > 0.0
    assert executor.record_times_ms == (executor.last_record_ms,)
    assert executor.total_record_ms == executor.last_record_ms
    assert executor.input_buffer_ptrs == pointers
    torch.testing.assert_close(first, first_snapshot)
    torch.testing.assert_close(second["value"], function(second_input, second_bias)["value"])
    assert first["value"].data_ptr() != second["value"].data_ptr()

    executor.reset()
    assert not executor.is_captured
    assert executor.input_buffer_ptrs == ()
    third = executor(second_input, second_bias)
    assert executor.record_count == 2
    assert len(executor.record_times_ms) == 2
    assert executor.total_record_ms >= executor.last_record_ms > 0.0
    torch.testing.assert_close(third, function(second_input, second_bias))


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_shape_dtype_and_device_changes_have_controlled_lifecycle() -> None:
    executor = GraphExecutor(lambda value: value.cos() + 0.25, warmup_runs=1)

    output_2 = executor(torch.zeros(2, 3, device="cuda", dtype=torch.float32))
    output_5 = executor(torch.zeros(5, 3, device="cuda", dtype=torch.float32))
    output_64 = executor(torch.zeros(5, 3, device="cuda", dtype=torch.float64))

    assert output_2.shape == (2, 3)
    assert output_5.shape == (5, 3)
    assert output_64.dtype == torch.float64
    assert executor.record_count == 3
    assert executor.replay_count == 0

    cpu = executor(torch.zeros(5, 3, dtype=torch.float64))
    assert not executor.is_captured
    assert executor.record_count == 3
    torch.testing.assert_close(cpu, torch.full((5, 3), 1.25, dtype=torch.float64))

    replayable = executor(torch.ones(5, 3, device="cuda", dtype=torch.float64))
    assert executor.record_count == 4
    torch.testing.assert_close(replayable, torch.ones_like(replayable).cos() + 0.25)


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_sequential_caller_stream_change_synchronizes_and_rerecords() -> None:
    executor = GraphExecutor(lambda value: value.square() + 0.5, warmup_runs=1)
    first_input = torch.linspace(-0.4, 0.6, 8, device="cuda")
    first = executor(first_input)
    assert executor.record_count == 1

    side_stream = torch.cuda.Stream()
    second_input = torch.linspace(0.7, -0.2, 8, device="cuda")
    with torch.cuda.stream(side_stream):
        second = executor(second_input)
    side_stream.synchronize()

    torch.testing.assert_close(first, first_input.square() + 0.5)
    torch.testing.assert_close(second, second_input.square() + 0.5)
    assert executor.record_count == 2
    assert executor.replay_count == 0

    # Returning to the default caller stream is another controlled lifecycle
    # change, never an unsynchronized replay into the side stream's buffers.
    third_input = torch.linspace(-0.1, 0.9, 8, device="cuda")
    third = executor(third_input)
    torch.testing.assert_close(third, third_input.square() + 0.5)
    assert executor.record_count == 3


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_graph_executor_replays_mixed_warp_fk_and_torch_ops_after_q_copy() -> None:
    """Prove one external CUDA graph can contain Warp and Torch launches."""
    pytest.importorskip("warp")
    warp_bridge = importlib.import_module("better_robot.kinematics._warp_bridge")
    warp_fk_forward = warp_bridge._warp_fk_forward

    model = make_smpl_like_model(dtype=torch.float32).to(device="cuda")
    structure, values = model.structure, model.values
    q_map = torch.zeros(1, dtype=torch.int32, device="cuda")
    value_map = torch.zeros_like(q_map)
    static_inputs = (
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

    def mixed_warp_torch(q: torch.Tensor) -> dict[str, torch.Tensor]:
        world, local, frames = warp_fk_forward(q, *static_inputs)
        # These reductions are ordinary Torch kernels recorded after Warp's
        # current-stream launch in the same external CUDA graph.
        score = world[..., :3].square().sum(dim=(-2, -1)) + local[..., :3].sin().sum(dim=(-2, -1))
        return {"world": world, "local": local, "frames": frames, "score": score}

    def torch_reference(q: torch.Tensor) -> dict[str, torch.Tensor]:
        world, local = forward_kinematics_raw(structure, values, q)
        frames = frame_placements_raw(structure, values, world)
        score = world[..., :3].square().sum(dim=(-2, -1)) + local[..., :3].sin().sum(dim=(-2, -1))
        return {"world": world, "local": local, "frames": frames, "score": score}

    executor = GraphExecutor(mixed_warp_torch, warmup_runs=3)
    first_q = model.q_neutral.unsqueeze(0).contiguous()
    first = executor(first_q)
    first_snapshot = {name: tensor.clone() for name, tensor in first.items()}
    input_pointer = executor.input_buffer_ptrs

    tangent = torch.linspace(-0.03, 0.03, model.nv, device="cuda")
    second_q = model.integrate(model.q_neutral, tangent).unsqueeze(0).contiguous()
    second = executor(second_q)

    torch.testing.assert_close(first, torch_reference(first_q), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(second, torch_reference(second_q), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(first, first_snapshot)
    assert executor.record_count == 1
    assert executor.replay_count == 1
    assert executor.input_buffer_ptrs == input_pointer
    assert executor.last_record_ms is not None and executor.last_record_ms > 0.0


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_public_warp_probe_is_capture_safe_and_replays_changed_q() -> None:
    """Exercise the selector itself, including its execution-batch plumbing."""
    pytest.importorskip("warp")
    warp_bridge = importlib.import_module("better_robot.kinematics._warp_bridge")
    try_warp_forward_kinematics = warp_bridge.try_warp_forward_kinematics

    model = make_smpl_like_model(dtype=torch.float32).to(device="cuda")
    structure, values = model.structure, model.values

    def public_warp_torch(q: torch.Tensor) -> dict[str, torch.Tensor]:
        result = try_warp_forward_kinematics(structure, values, q)
        if result is None:
            raise AssertionError("the supported public Warp FK probe unexpectedly fell back")
        score = result.world[..., :3].square().sum(dim=(-2, -1))
        return {
            "world": result.world,
            "local": result.local,
            "frames": result.frames,
            "score": score,
        }

    def torch_reference(q: torch.Tensor) -> dict[str, torch.Tensor]:
        world, local = forward_kinematics_raw(structure, values, q)
        frames = frame_placements_raw(structure, values, world)
        score = world[..., :3].square().sum(dim=(-2, -1))
        return {"world": world, "local": local, "frames": frames, "score": score}

    executor = GraphExecutor(public_warp_torch, warmup_runs=3)
    first_q = model.q_neutral.unsqueeze(0).contiguous()
    first = executor(first_q)

    tangent = torch.linspace(0.04, -0.03, model.nv, device="cuda")
    second_q = model.integrate(model.q_neutral, tangent).unsqueeze(0).contiguous()
    second = executor(second_q)

    torch.testing.assert_close(first, torch_reference(first_q), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(second, torch_reference(second_q), rtol=2e-5, atol=2e-6)
    assert not torch.equal(first["score"], second["score"])
    assert executor.record_count == 1
    assert executor.replay_count == 1


@pytest.mark.cuda
@_CUDA_AVAILABLE
@pytest.mark.parametrize("input_name", ("q", "joint_placements"))
def test_warp_layout_fallback_is_a_hard_error_only_during_capture(input_name: str) -> None:
    pytest.importorskip("warp")
    warp_bridge = importlib.import_module("better_robot.kinematics._warp_bridge")
    try_warp_forward_kinematics = warp_bridge.try_warp_forward_kinematics

    model = make_smpl_like_model(dtype=torch.float32).to(device="cuda")
    q = model.q_neutral.unsqueeze(0).contiguous()
    values = model.values
    if input_name == "q":
        storage = torch.empty(q.shape[0], q.shape[1] * 2, dtype=q.dtype, device=q.device)
        q = storage[..., ::2]
        q.copy_(model.q_neutral)
        assert q.stride(-1) == 2
    else:
        placements = values.joint_placements
        storage = torch.empty(
            *placements.shape[:-1],
            placements.shape[-1] * 2,
            dtype=placements.dtype,
            device=placements.device,
        )
        placement_view = storage[..., ::2]
        placement_view.copy_(placements)
        assert placement_view.stride(-1) == 2
        values = dataclasses.replace(values, joint_placements=placement_view)

    # Outside capture the opt-in probe keeps its established silent fallback.
    assert try_warp_forward_kinematics(model.structure, values, q) is None

    expected = (
        f"better_robot: input {input_name!r} has an unsupported layout for the Warp FK lane "
        "while CUDA graph capture is active; silent Torch fallback is disabled during capture. "
        "Make the input contiguous before capture, or disable graph capture."
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with pytest.raises(RuntimeError) as caught, torch.cuda.graph(graph, stream=stream):
        # Keep the intentionally aborted graph non-empty so PyTorch does not
        # emit its unrelated empty-capture warning on context teardown.
        torch.square(q)
        try_warp_forward_kinematics(model.structure, values, q)
    assert str(caught.value) == expected


class _TargetResidual:
    name = "target"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


class _NonlinearTargetResidual:
    name = "nonlinear_target"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] + 0.2 * ctx["x"].square() - ctx["target"]


def _lm_case(
    solver: LevenbergMarquardt,
    problem: Problem,
    *,
    batch_size: int,
    offset: float,
) -> tuple[dict[str, torch.Tensor], Any]:
    value = torch.linspace(-0.8, 0.5, batch_size, device="cuda").unsqueeze(-1) + offset
    values = {"x": value}
    return values, solver.init_state(values, problem)


def _assert_lm_close(actual: tuple[Any, Any], expected: tuple[Any, Any]) -> None:
    actual_values, actual_state = actual
    expected_values, expected_state = expected
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-5, atol=2e-6)
    for actual_tensor, expected_tensor in zip(actual_state, expected_state, strict=True):
        if actual_tensor.is_floating_point():
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-5, atol=2e-6)
        else:
            assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_default_python_residual_weight_is_capture_safe() -> None:
    """The ordinary scalar-weight path must not transfer from the host."""
    target = torch.tensor([0.75], device="cuda")
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("target", _TargetResidual()),),
        parameters={"target": target},
    )
    solver = LevenbergMarquardt(linearization="dense", jacobian_strategy="jacrev")
    inputs = _lm_case(solver, problem, batch_size=2, offset=0.0)
    eager = solver.update(*inputs, problem)
    executor = GraphExecutor(lambda values, state: solver.update(values, state, problem), warmup_runs=1)
    captured = executor(*inputs)

    _assert_lm_close(captured, eager)
    assert executor.is_captured
    assert executor.record_count == 1


@pytest.mark.cuda
@_CUDA_AVAILABLE
def test_fixed_lm_updates_capture_resize_replay_stability_and_memory_plateau() -> None:  # noqa: PLR0915
    """Capture nonlinear LM/jacrev work and prove a 100-replay plateau."""
    target = torch.tensor([0.75], device="cuda")
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("nonlinear_target", _NonlinearTargetResidual()),),
        parameters={"target": target},
    )
    solver = LevenbergMarquardt(
        max_iter=3,
        gtol=1e-7,
        xtol=1e-9,
        ftol=1e-9,
        linearization="dense",
        jacobian_strategy="jacrev",
    )

    def three_updates(values, state):
        # A changing output derived from the copied graph input makes stale
        # replay immediately visible even if the optimizer nearly converges.
        input_probe = values["x"].squeeze(-1).sin()
        for _ in range(3):
            values, state = solver.update(values, state, problem)
        return values, state, input_probe

    def assert_capture_close(actual, expected):
        _assert_lm_close(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=2e-6)

    executor = GraphExecutor(three_updates, warmup_runs=3)
    initial = _lm_case(solver, problem, batch_size=8, offset=0.0)
    expected = three_updates(*initial)
    actual = executor(*initial)
    assert_capture_close(actual, expected)
    assert executor.record_count == 1

    resized = _lm_case(solver, problem, batch_size=5, offset=0.1)
    resized_expected = three_updates(*resized)
    resized_actual = executor(*resized)
    assert_capture_close(resized_actual, resized_expected)
    assert executor.record_count == 2

    cases: list[tuple[tuple[Any, Any], tuple[Any, Any, torch.Tensor]]] = []
    for index in range(8):
        inputs = _lm_case(solver, problem, batch_size=5, offset=0.01 * index)
        cases.append((inputs, three_updates(*inputs)))
    for inputs, expected_case in cases:
        assert_capture_close(executor(*inputs), expected_case)

    expected_probes = [float(expected_case[2][0].item()) for _, expected_case in cases]
    assert max(expected_probes) - min(expected_probes) > 1e-3

    # All source cases and eager references are allocated before measuring.
    # Two independent windows must finish at the same plateau while every
    # replay proves that the copied input changed the captured output.
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    output_leaves = [
        leaf for leaf in tree_flatten(cases[0][1])[0] if isinstance(leaf, torch.Tensor)
    ]
    output_bytes = sum(leaf.numel() * leaf.element_size() for leaf in output_leaves)
    ending_allowance = max(4096, output_bytes)
    peak_allowance = ending_allowance + 2048 * len(output_leaves)

    def replay_window(start_index: int) -> tuple[int, int, int, list[float]]:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.synchronize()
        starting = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        observed: list[float] = []
        for index in range(start_index, start_index + 50):
            inputs, expected_case = cases[index % len(cases)]
            replayed = executor(*inputs)
            actual_probe = float(replayed[2][0].item())
            expected_probe = float(expected_case[2][0].item())
            assert actual_probe == pytest.approx(expected_probe, rel=2e-5, abs=2e-6)
            observed.append(actual_probe)
            del replayed
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.synchronize()
        return (
            starting,
            torch.cuda.memory_allocated(),
            torch.cuda.max_memory_allocated(),
            observed,
        )

    first_start, first_end, first_peak, first_observed = replay_window(0)
    second_start, second_end, second_peak, second_observed = replay_window(50)

    assert executor.record_count == 2
    assert executor.replay_count >= 108
    assert max(first_observed + second_observed) - min(first_observed + second_observed) > 1e-3
    assert first_end <= first_start + ending_allowance
    assert second_end <= second_start + ending_allowance
    assert abs(second_end - first_end) <= ending_allowance
    assert first_peak <= first_start + peak_allowance
    assert second_peak <= second_start + peak_allowance
    assert second_peak - second_start <= first_peak - first_start + ending_allowance
