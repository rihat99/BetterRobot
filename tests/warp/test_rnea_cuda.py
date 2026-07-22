"""CUDA parity and differentiation contracts for fused Warp RNEA."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
import importlib
import warnings

import pytest
import torch


pytest.importorskip("warp")

from better_robot.dynamics._warp_bridge import try_warp_rnea
from better_robot.dynamics.rnea import rnea, rnea_raw
from better_robot.io import load
from tests.support.branching_tree import make_branching_tree_model


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _unit_quat(placement: torch.Tensor) -> torch.Tensor:
    """Return an SE3 placement with its quaternion projected onto the unit sphere.

    Torch RNEA now runs the normalising matrix FK lane, while the Warp RNEA
    kernel composes raw transforms. They agree exactly only on unit-quaternion
    placements; URDF ``rpy``→quat conversion leaves Panda's placements slightly
    non-unit. Projecting onto SE3 removes that benign gauge, and doing it inside
    a gradcheck also drops the meaningless radial placement-gradient direction.
    """
    return torch.cat(
        [placement[..., :3], placement[..., 3:7] / placement[..., 3:7].norm(dim=-1, keepdim=True)],
        dim=-1,
    )


def _unit_placement_values(values):
    """Return ``values`` with unit-quaternion joint placements."""
    return dataclasses.replace(values, joint_placements=_unit_quat(values.joint_placements))


@lru_cache(maxsize=None)
def _model(kind: str, dtype: torch.dtype):
    if kind == "smpl":
        model = make_branching_tree_model(dtype=dtype)
    else:
        panda_description = pytest.importorskip(  # noqa: PLC0415
            "robot_descriptions.panda_description"
        )

        model = load(panda_description.URDF_PATH, dtype=dtype)
    return model.to(device="cuda:0")


def _inputs(model, batch_size: int, *, with_fext: bool):
    dtype = model.q_neutral.dtype
    tangent = torch.linspace(-0.03, 0.03, model.nv, dtype=dtype, device="cuda:0")
    q_seed = model.integrate(model.q_neutral, tangent)
    q = q_seed.expand(batch_size, -1).clone()
    velocity = torch.linspace(-0.1, 0.1, model.nv, dtype=dtype, device="cuda:0")
    velocity = velocity.expand(batch_size, -1).clone()
    acceleration = torch.linspace(0.2, -0.2, model.nv, dtype=dtype, device="cuda:0")
    acceleration = acceleration.expand(batch_size, -1).clone()
    fext = None
    if with_fext:
        row = torch.linspace(
            -0.05,
            0.05,
            model.njoints * 6,
            dtype=dtype,
            device="cuda:0",
        ).reshape(model.njoints, 6)
        fext = row.expand(batch_size, -1, -1).clone()
    return q, velocity, acceleration, fext


def _result_tensors(result) -> tuple[torch.Tensor, ...]:
    return (
        result.tau,
        result.joint_pose_world,
        result.joint_pose_local,
        result.joint_velocity_local,
        result.joint_acceleration_local,
        result.joint_forces,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
@pytest.mark.parametrize("model_kind", ("panda", "smpl"))
@pytest.mark.parametrize("batch_size", (1, 4096), ids=("b1", "b4096"))
@pytest.mark.parametrize("with_fext", (False, True), ids=("no-fext", "fext"))
def test_rnea_forward_matches_torch(
    dtype: torch.dtype,
    model_kind: str,
    batch_size: int,
    with_fext: bool,
) -> None:
    """Cover both flagship models, dtypes, target batches, and fext modes."""
    model = _model(model_kind, dtype)
    q, velocity, acceleration, fext = _inputs(model, batch_size, with_fext=with_fext)
    # Project placements onto SE3 so the normalising Torch FK lane and the Warp
    # kernel agree at fp64 (Panda's URDF placement quaternions are non-unit).
    values = _unit_placement_values(model.values)
    actual = try_warp_rnea(
        model.structure,
        values,
        q,
        velocity,
        acceleration,
        fext=fext,
    )
    assert actual is not None
    expected = rnea_raw(
        model.structure,
        values,
        q,
        velocity,
        acceleration,
        fext=fext,
    )
    tolerances = (3e-4, 2e-4) if dtype == torch.float32 else (2e-9, 2e-10)
    for actual_tensor, expected_tensor in zip(
        _result_tensors(actual),
        _result_tensors(expected),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=tolerances[0],
            atol=tolerances[1],
        )


def test_rnea_recompute_vjp_matches_torch_for_all_inputs() -> None:
    """Differentiate q/v/a, fext, placements, inertias, and gravity."""
    model = _model("panda", torch.float32)
    q_data, velocity_data, acceleration_data, fext_data = _inputs(model, 1, with_fext=True)
    assert fext_data is not None

    def evaluate(use_warp: bool):
        inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (
                q_data,
                velocity_data,
                acceleration_data,
                fext_data,
                model.values.joint_placements,
                model.values.body_inertias,
                model.values.gravity,
            )
        )
        q, velocity, acceleration, fext, placements, inertias, gravity = inputs
        values = dataclasses.replace(
            model.values,
            joint_placements=placements,
            body_inertias=inertias,
            gravity=gravity,
        )
        if use_warp:
            result = try_warp_rnea(
                model.structure,
                values,
                q,
                velocity,
                acceleration,
                fext=fext,
            )
            assert result is not None
        else:
            result = rnea_raw(
                model.structure,
                values,
                q,
                velocity,
                acceleration,
                fext=fext,
            )
        weights = torch.linspace(
            0.1,
            0.9,
            result.tau.numel(),
            dtype=result.tau.dtype,
            device=result.tau.device,
        ).reshape_as(result.tau)
        return result.tau, torch.autograd.grad((weights * result.tau).sum(), inputs)

    actual, actual_gradients = evaluate(True)
    expected, expected_gradients = evaluate(False)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-4)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)


def test_rnea_recompute_vjp_matches_torch_for_output_subset() -> None:
    """A loss over a strict, non-contiguous subset of outputs still matches Torch.

    The Warp recompute differentiates only outputs that receive an upstream
    gradient; None-seeded outputs are dropped. This guards that the drop keeps
    exact gradient semantics and pairs the surviving seeds with the right output.
    """
    model = _model("panda", torch.float32)
    q_data, velocity_data, acceleration_data, _ = _inputs(model, 4, with_fext=False)

    def evaluate(use_warp: bool):
        inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (q_data, velocity_data, acceleration_data)
        )
        q, velocity, acceleration = inputs
        if use_warp:
            result = try_warp_rnea(model.structure, model.values, q, velocity, acceleration)
            assert result is not None
        else:
            result = rnea_raw(model.structure, model.values, q, velocity, acceleration)
        # Seed a non-contiguous subset: tau (0), joint_velocity_local (3),
        # joint_forces (5); the other three outputs receive no gradient.
        loss = result.tau.sum() + (2.0 * result.joint_velocity_local).sum() + (0.5 * result.joint_forces).sum()
        return torch.autograd.grad(loss, inputs)

    actual_gradients = evaluate(True)
    expected_gradients = evaluate(False)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)


def test_rnea_float32_gradcheck_includes_joint_placements() -> None:
    """Check the recompute VJP against Warp finite differences off manifold."""
    model = _model("panda", torch.float32)
    q, velocity, acceleration, _ = _inputs(model, 1, with_fext=False)
    differentiable = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (q, velocity, acceleration, model.values.joint_placements)
    )

    def function(q_input, velocity_input, acceleration_input, placements_input):
        # Project placements onto SE3 so the gradcheck exercises only the
        # tangential placement gradient; the radial (quaternion-norm) direction
        # is a gauge on which the matrix VJP and the Warp forward disagree.
        values = dataclasses.replace(model.values, joint_placements=_unit_quat(placements_input))
        result = try_warp_rnea(
            model.structure,
            values,
            q_input,
            velocity_input,
            acceleration_input,
        )
        assert result is not None
        weights = torch.linspace(
            0.2,
            0.8,
            model.nv,
            dtype=q_input.dtype,
            device=q_input.device,
        )
        return (result.tau * weights).sum()

    assert torch.autograd.gradcheck(
        function,
        differentiable,
        eps=1e-3,
        atol=3e-2,
        rtol=3e-2,
        fast_mode=True,
    )


def test_rnea_second_order_q_gradient_matches_torch() -> None:
    model = _model("panda", torch.float32)
    q_data, velocity, acceleration, _ = _inputs(model, 1, with_fext=False)

    def derivatives(use_warp: bool):
        q = q_data.detach().clone().requires_grad_()
        if use_warp:
            result = try_warp_rnea(
                model.structure,
                model.values,
                q,
                velocity,
                acceleration,
            )
            assert result is not None
        else:
            result = rnea_raw(
                model.structure,
                model.values,
                q,
                velocity,
                acceleration,
            )
        first = torch.autograd.grad(result.tau.sum(), q, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), q)[0]
        return first, second

    actual = derivatives(True)
    expected = derivatives(False)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)


def test_frame_value_batch_participates_in_execution_shape() -> None:
    model = _model("smpl", torch.float32)
    q, velocity, acceleration, _ = _inputs(model, 1, with_fext=False)
    q = q.squeeze(0)
    velocity = velocity.squeeze(0)
    acceleration = acceleration.squeeze(0)
    frame_placements = model.values.frame_placements.unsqueeze(0).repeat(3, 1, 1)
    values = dataclasses.replace(model.values, frame_placements=frame_placements)
    actual = try_warp_rnea(
        model.structure,
        values,
        q,
        velocity,
        acceleration,
    )
    assert actual is not None
    expected = rnea_raw(model.structure, values, q, velocity, acceleration)
    assert actual.tau.shape == (3, model.nv)
    torch.testing.assert_close(actual.tau, expected.tau, rtol=3e-4, atol=2e-4)


def test_public_opt_in_and_cuda_graph_replay() -> None:
    model = _model("panda", torch.float32)
    q, velocity, acceleration, _ = _inputs(model, 1, with_fext=False)
    expected = rnea_raw(model.structure, model.values, q, velocity, acceleration).tau
    torch.testing.assert_close(
        rnea(model, q, velocity, acceleration, use_warp=True),
        expected,
        rtol=3e-4,
        atol=2e-4,
    )

    q_static = q.clone()
    try_warp_rnea(model.structure, model.values, q_static, velocity, acceleration)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = try_warp_rnea(
            model.structure,
            model.values,
            q_static,
            velocity,
            acceleration,
        )
        assert captured is not None

    delta = torch.linspace(0.01, -0.01, model.nv, device="cuda:0")
    q_static.copy_(model.integrate(q, delta))
    graph.replay()
    torch.cuda.synchronize()
    expected_replay = rnea_raw(
        model.structure,
        model.values,
        q_static,
        velocity,
        acceleration,
    )
    torch.testing.assert_close(captured.tau, expected_replay.tau, rtol=3e-4, atol=2e-4)


def test_layout_decline_is_a_hard_error_during_capture() -> None:
    model = _model("smpl", torch.float32)
    q_contiguous, velocity, acceleration, _ = _inputs(model, 1, with_fext=False)
    storage = torch.empty(
        q_contiguous.shape[0],
        q_contiguous.shape[1] * 2,
        dtype=q_contiguous.dtype,
        device=q_contiguous.device,
    )
    q = storage[..., ::2]
    q.copy_(q_contiguous)
    assert q.stride(-1) == 2
    with pytest.warns(RuntimeWarning, match="unit trailing stride"):
        assert try_warp_rnea(model.structure, model.values, q, velocity, acceleration) is None

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with (
        pytest.raises(RuntimeError, match="cannot fall back.*capture"),
        torch.cuda.graph(
            graph,
            stream=stream,
        ),
    ):
        torch.square(q)
        try_warp_rnea(model.structure, model.values, q, velocity, acceleration)


def test_cpu_request_warns_once_and_falls_back(monkeypatch) -> None:
    model = make_branching_tree_model(dtype=torch.float32)
    q, velocity, acceleration, _ = _inputs(
        model.to(device="cuda:0"),
        1,
        with_fext=False,
    )
    q = q.cpu()
    velocity = velocity.cpu()
    acceleration = acceleration.cpu()
    rnea_module = importlib.import_module("better_robot.dynamics.rnea")
    monkeypatch.setattr(rnea_module, "_WARNED_WARP_RNEA_FALLBACKS", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = rnea(model, q, velocity, acceleration, use_warp=True)
        repeated = rnea(model, q, velocity, acceleration, use_warp=True)
    warp_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning) and "Warp RNEA lane" in str(warning.message)
    ]
    assert len(warp_warnings) == 1
    expected = rnea(model, q, velocity, acceleration)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)
