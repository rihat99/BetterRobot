"""Parity, differentiation, batching, and compile contracts for Warp FK."""

from __future__ import annotations

import builtins
import dataclasses
import math
import warnings

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

pytest.importorskip("warp")

import better_robot.kinematics._warp_bridge as warp_bridge
import better_robot.kinematics.forward as forward_module
from better_robot.data_model.execution_batch import flatten_execution_batch
from better_robot.io.build_model import build_model
from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics._warp_bridge import (
    _warp_fk_forward,
    try_warp_forward_kinematics,
)
from better_robot.kinematics.forward import (
    forward_kinematics,
    forward_kinematics_raw,
    frame_placements_raw,
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _make_branched_model(dtype: torch.dtype):
    builder = ModelBuilder("warp_branched")
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


def _make_deep_chain(
    depth: int = 18,
    *,
    dtype: torch.dtype = torch.float32,
):
    builder = ModelBuilder("warp_deep_chain")
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


def _outputs(result) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return result.world, result.local, result.frames


def _torch_outputs(model, values, q) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = forward_kinematics_raw(model.structure, values, q)
    world = result.joint_pose_world
    local = result.joint_pose_local
    frames = frame_placements_raw(model.structure, values, world).frame_pose_world
    return world, local, frames


def _assert_outputs_close(actual, expected, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        rtol, atol = 2e-5, 2e-6
    else:
        rtol, atol = 2e-10, 2e-11
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=rtol,
            atol=atol,
        )


def _weighted_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total = outputs[0].new_zeros(())
    for scale, tensor in zip((0.7, -0.4, 0.2), outputs, strict=True):
        weights = torch.linspace(
            0.1,
            0.9,
            tensor.numel(),
            dtype=tensor.dtype,
            device=tensor.device,
        ).reshape(tensor.shape)
        total = total + scale * (weights * tensor).sum()
    return total


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
def test_branched_forward_matches_torch_and_public_opt_in(dtype: torch.dtype) -> None:
    model = _make_branched_model(dtype)
    q = torch.tensor([0.2, 0.1, -0.3], dtype=dtype)

    direct = try_warp_forward_kinematics(model.structure, model.values, q)
    assert direct is not None, "supported test input must not silently use torch fallback"
    expected = _torch_outputs(model, model.values, q)
    _assert_outputs_close(_outputs(direct), expected, dtype)

    public = forward_kinematics(model, q, compute_frames=True, use_warp=True)
    assert public.frame_pose_world is not None
    _assert_outputs_close(
        (public.joint_pose_world, public.joint_pose_local, public.frame_pose_world),
        expected,
        dtype,
    )


def test_smpl_free_flyer_and_spherical_branching_matches_torch() -> None:
    model = make_smpl_like_model(dtype=torch.float32)
    tangent = torch.linspace(-0.03, 0.03, model.nv)
    q = model.integrate(model.q_neutral, tangent).contiguous()

    direct = try_warp_forward_kinematics(model.structure, model.values, q)
    assert direct is not None
    assert model.structure.joint_kind_codes[1] == 12  # free_flyer ABI code
    assert max(len(children) for children in model.children) >= 3
    _assert_outputs_close(
        _outputs(direct),
        _torch_outputs(model, model.values, q),
        torch.float32,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
def test_degenerate_spherical_and_free_flyer_quaternions_match_torch(dtype: torch.dtype) -> None:
    model = make_smpl_like_model(dtype=dtype)
    q = model.q_neutral.clone()
    free_flyer = model.structure.joint_kind_codes.index(12)
    spherical = model.structure.joint_kind_codes.index(11)
    free_flyer_q = model.idx_qs[free_flyer]
    spherical_q = model.idx_qs[spherical]
    q[free_flyer_q + 3 : free_flyer_q + 7] = 0.0
    q[spherical_q : spherical_q + 4] = q.new_tensor((1.0e-10, -2.0e-10, 3.0e-10, -4.0e-10))

    direct = try_warp_forward_kinematics(model.structure, model.values, q)
    assert direct is not None
    _assert_outputs_close(
        _outputs(direct),
        _torch_outputs(model, model.values, q),
        dtype,
    )


def test_deep_chain_exceeds_sixteen_levels_and_matches_torch() -> None:
    model = _make_deep_chain()
    assert max(len(support) for support in model.supports) > 16
    q = torch.linspace(-0.25, 0.25, model.nq)

    direct = try_warp_forward_kinematics(model.structure, model.values, q)
    assert direct is not None
    _assert_outputs_close(
        _outputs(direct),
        _torch_outputs(model, model.values, q),
        torch.float32,
    )


def test_unsupported_layout_fallback_warns_once_without_copying(monkeypatch) -> None:
    monkeypatch.setattr(forward_module, "_WARNED_WARP_FALLBACKS", set())
    model = _make_branched_model(torch.float32)
    q_storage = torch.zeros(model.nq * 2, dtype=torch.float32)
    q = q_storage[::2]
    assert q.stride(-1) != 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert try_warp_forward_kinematics(model.structure, model.values, q) is None
        assert try_warp_forward_kinematics(model.structure, model.values, q) is None
    fallbacks = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(fallbacks) == 1
    assert "inputs ('q',) do not have unit stride" in str(fallbacks[0].message)

    q_batch = torch.zeros((3, 2, model.nq), dtype=torch.float32)
    placements = model.values.joint_placements.expand(3, 2, -1, -1).clone()
    frame_storage = model.values.frame_placements.expand(2, 3, -1, -1).clone()
    frames = frame_storage.transpose(0, 1)
    assert frames.stride(-1) == 1
    assert not frames.is_contiguous()
    values = dataclasses.replace(
        model.values,
        joint_placements=placements,
        frame_placements=frames,
    )
    with pytest.warns(RuntimeWarning, match="flattening would materialize"):
        assert try_warp_forward_kinematics(model.structure, values, q_batch) is None


def test_unknown_joint_kind_warns_and_falls_back() -> None:
    model = _make_branched_model(torch.float32)
    kind_codes = (*model.structure.joint_kind_codes[:-1], 99)
    kind_tensor = model.structure.joint_kind_tensor.clone()
    kind_tensor[-1] = 99
    structure = dataclasses.replace(
        model.structure,
        joint_kind_codes=kind_codes,
        joint_kind_tensor=kind_tensor,
    )

    with pytest.warns(RuntimeWarning, match="joint kind unsupported"):
        assert try_warp_forward_kinematics(structure, model.values, torch.zeros(model.nq)) is None


def test_chained_mimic_coordinates_run_on_warp_and_match_torch() -> None:
    builder = ModelBuilder("warp_mimic_chain")
    root = builder.add_body("root")
    source = builder.add_body("source")
    target = builder.add_body("target")
    chained = builder.add_body("chained")
    builder.add_revolute_z("source_joint", parent=root, child=source)
    builder.add_revolute_z(
        "target_joint",
        parent=source,
        child=target,
        mimic_source="source_joint",
        mimic_multiplier=-0.5,
        mimic_offset=0.1,
    )
    builder.add_revolute_z(
        "chained_joint",
        parent=target,
        child=chained,
        mimic_source="target_joint",
        mimic_multiplier=2.0,
        mimic_offset=-0.3,
    )
    model = build_model(builder.finalize())
    q = torch.tensor([0.4])

    assert model.has_mimic
    direct = try_warp_forward_kinematics(model.structure, model.values, q)
    assert direct is not None
    actual = forward_kinematics(model, q, use_warp=True)
    expected = forward_kinematics(model, q, use_warp=False)
    torch.testing.assert_close(direct.world, expected.joint_pose_world)
    torch.testing.assert_close(actual.joint_pose_world, expected.joint_pose_world)


def test_public_opt_in_does_not_mask_import_error_from_bridge(monkeypatch) -> None:
    model = _make_branched_model(torch.float32)
    q = torch.zeros(model.nq)

    def fail_inside_bridge(*_args):
        raise ImportError("fault inside the installed Warp lane")

    monkeypatch.setattr(warp_bridge, "try_warp_forward_kinematics", fail_inside_bridge)
    with pytest.raises(ImportError, match="fault inside the installed Warp lane"):
        forward_kinematics(model, q, use_warp=True)


def test_missing_warp_cpu_fallback_does_not_probe_cuda_capture(monkeypatch) -> None:
    model = _make_branched_model(torch.float32)
    q = torch.zeros(model.nq)
    import_function = builtins.__import__

    def import_without_warp(name, global_vars=None, local_vars=None, fromlist=(), level=0):
        package = None if global_vars is None else global_vars.get("__package__")
        if name == "_warp_bridge" and package == "better_robot.kinematics":
            raise ModuleNotFoundError("No module named 'warp'", name="warp")
        return import_function(name, global_vars, local_vars, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_warp)
    monkeypatch.setattr(forward_module, "_WARNED_WARP_FALLBACKS", set())

    def fail_if_probed():
        raise AssertionError("CPU fallback must not query CUDA capture state")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", fail_if_probed)
    with pytest.warns(RuntimeWarning, match="optional Warp runtime is unavailable"):
        actual = forward_kinematics(model, q, use_warp=True)
    expected = forward_kinematics(model, q, use_warp=False)
    torch.testing.assert_close(actual.joint_pose_world, expected.joint_pose_world)


def _broadcast_inputs(model):
    q = torch.tensor(
        [[[0.2, 0.1, -0.3]], [[-0.15, -0.05, 0.25]]],
        dtype=torch.float64,
    )
    placements = model.values.joint_placements[None, None].repeat(1, 3, 1, 1)
    placements[0, 1, :, 0] += 0.01
    placements[0, 2, :, 1] -= 0.02
    frames = model.values.frame_placements[None, None].repeat(1, 3, 1, 1)
    frames[0, 1, :, 2] += 0.02
    frames[0, 2, :, 0] -= 0.01
    return q, placements, frames


def _warp_outputs_with_values(model, q, placements, frames):
    values = dataclasses.replace(
        model.values,
        joint_placements=placements,
        frame_placements=frames,
    )
    result = try_warp_forward_kinematics(model.structure, values, q)
    assert result is not None
    return _outputs(result)


def _torch_outputs_with_values(model, q, placements, frames):
    values = dataclasses.replace(
        model.values,
        joint_placements=placements,
        frame_placements=frames,
    )
    return _torch_outputs(model, values, q)


def test_broadcast_maps_and_shared_value_gradient_reduction_match_torch() -> None:
    model = _make_branched_model(torch.float64)
    q_data, placement_data, frame_data = _broadcast_inputs(model)
    execution = flatten_execution_batch(
        q_data,
        (placement_data, frame_data),
        value_event_ndims=(2, 2),
    )
    assert execution.batch_shape == (2, 3)
    assert execution.q.batch_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert execution.values[0].batch_indices.tolist() == [0, 1, 2, 0, 1, 2]

    q = q_data.detach().clone().requires_grad_()
    placements = placement_data.detach().clone().requires_grad_()
    frames = frame_data.detach().clone().requires_grad_()
    warp_outputs = _warp_outputs_with_values(model, q, placements, frames)
    warp_gradients = torch.autograd.grad(
        _weighted_loss(warp_outputs),
        (q, placements, frames),
    )

    q_ref = q_data.detach().clone().requires_grad_()
    placements_ref = placement_data.detach().clone().requires_grad_()
    frames_ref = frame_data.detach().clone().requires_grad_()
    torch_outputs = _torch_outputs_with_values(
        model,
        q_ref,
        placements_ref,
        frames_ref,
    )
    torch_gradients = torch.autograd.grad(
        _weighted_loss(torch_outputs),
        (q_ref, placements_ref, frames_ref),
    )

    _assert_outputs_close(warp_outputs, torch_outputs, torch.float64)
    for actual, expected in zip(warp_gradients, torch_gradients, strict=True):
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


def test_inertia_only_value_batch_expands_warp_fk_execution_shape() -> None:
    model = _make_branched_model(torch.float64)
    inertias = model.values.body_inertias.expand(3, -1, -1).clone()
    values = dataclasses.replace(model.values, body_inertias=inertias)
    result = try_warp_forward_kinematics(
        model.structure,
        values,
        torch.zeros(model.nq, dtype=torch.float64),
    )
    assert result is not None
    expected = _torch_outputs(
        model,
        values,
        torch.zeros(model.nq, dtype=torch.float64),
    )
    assert result.world.shape == (3, model.njoints, 7)
    _assert_outputs_close(_outputs(result), expected, torch.float64)


def test_q_gradcheck_and_gradgradcheck() -> None:
    torch.manual_seed(0)
    model = _make_branched_model(torch.float64)
    # Exercise the revolute exponential exactly at theta=0, the singular
    # point whose torch derivative was repaired in M0.
    q = torch.zeros(model.nq, dtype=torch.float64, requires_grad=True)

    def function(q_input):
        result = try_warp_forward_kinematics(
            model.structure,
            model.values,
            q_input,
        )
        assert result is not None
        return _weighted_loss(_outputs(result))

    options = {"eps": 1e-6, "atol": 2e-5, "rtol": 2e-4}
    assert torch.autograd.gradcheck(function, (q,), **options)
    assert torch.autograd.gradgradcheck(function, (q,), **options)


def test_joint_and_frame_placement_gradcheck() -> None:
    model = _make_branched_model(torch.float64)
    q = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float64)
    placements = model.values.joint_placements.detach().clone().requires_grad_()
    frames = model.values.frame_placements.detach().clone().requires_grad_()

    def function(placement_input, frame_input):
        return _weighted_loss(
            _warp_outputs_with_values(
                model,
                q,
                placement_input,
                frame_input,
            )
        )

    assert torch.autograd.gradcheck(
        function,
        (placements, frames),
        eps=1e-6,
        atol=5e-5,
        rtol=5e-4,
        fast_mode=True,
    )


def test_deep_chain_q_and_value_gradcheck_at_singularity() -> None:
    """Pin the dynamic-loop adjoint beyond Warp's default unroll depth."""
    model = _make_deep_chain(dtype=torch.float64)
    q = torch.zeros(model.nq, dtype=torch.float64, requires_grad=True)
    placements = model.values.joint_placements.detach().clone().requires_grad_()
    frames = model.values.frame_placements.detach().clone().requires_grad_()

    def function(q_input, placement_input, frame_input):
        return _weighted_loss(
            _warp_outputs_with_values(
                model,
                q_input,
                placement_input,
                frame_input,
            )
        )

    assert torch.autograd.gradcheck(
        function,
        (q, placements, frames),
        eps=1e-6,
        atol=8e-5,
        rtol=8e-4,
        fast_mode=True,
    )


def _direct_custom_op_inputs(model):
    q = torch.tensor(
        [[0.2, 0.1, -0.3], [-0.15, -0.05, 0.25]],
        dtype=torch.float32,
    )
    structure = model.structure
    values = model.values
    return (
        q,
        values.joint_placements.reshape(1, model.njoints, 7),
        values.frame_placements.reshape(1, model.nframes, 7),
        torch.arange(q.shape[0], dtype=torch.int32),
        torch.zeros(q.shape[0], dtype=torch.int32),
        structure.parents_tensor,
        structure.topo_order_tensor,
        structure.joint_kind_tensor,
        structure.idx_qs_tensor,
        structure.idx_qs_full_tensor,
        structure.mimic_source_tensor,
        structure.q_expansion,
        structure.q_offset,
        structure.joint_axes,
        structure.joint_pitches,
        structure.frame_parent_joints,
    )


def test_direct_custom_op_fake_and_compile_fullgraph() -> None:
    compile_function = getattr(torch, "compile", None)
    if compile_function is None:
        pytest.skip("torch.compile is unavailable")

    model = _make_branched_model(torch.float32)
    inputs = _direct_custom_op_inputs(model)
    fake_mode = FakeTensorMode()
    with fake_mode:
        fake_outputs = _warp_fk_forward(*(fake_mode.from_tensor(tensor) for tensor in inputs))
    expected_shapes = (
        (2, model.njoints, 7),
        (2, model.njoints, 7),
        (2, model.nframes, 7),
    )
    assert tuple(output.shape for output in fake_outputs) == expected_shapes
    assert all(isinstance(output, FakeTensor) for output in fake_outputs)

    def direct_call(
        q,
        joint_placements,
        frame_placements,
        q_map,
        value_map,
        parents,
        topo_order,
        kinds,
        idx_qs,
        idx_qs_full,
        mimic_sources,
        q_expansion,
        q_offsets,
        axes,
        pitches,
        frame_parents,
    ):
        return _warp_fk_forward(
            q,
            joint_placements,
            frame_placements,
            q_map,
            value_map,
            parents,
            topo_order,
            kinds,
            idx_qs,
            idx_qs_full,
            mimic_sources,
            q_expansion,
            q_offsets,
            axes,
            pitches,
            frame_parents,
        )

    expected = direct_call(*inputs)
    compiled = compile_function(direct_call, fullgraph=True, backend="eager")
    actual = compiled(*inputs)
    _assert_outputs_close(actual, expected, torch.float32)
