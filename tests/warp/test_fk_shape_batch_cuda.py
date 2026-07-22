"""CUDA parity for the Warp FK lane on shape-batched models.

Baking a body's shape with a batch leaves ``ModelValues.joint_placements``
batched (``(B, njoints, 7)``) while ``frame_placements`` stays unbatched
(``(nframes, 7)``) — the training-style shape+pose path. The Warp lane must run
this case with a separate frame execution-index map instead of declining, and
match the Torch lane on both forward and backward.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch


pytest.importorskip("warp")

import better_robot.kinematics.forward as forward_module
from better_robot.io import load
from tests.support.branching_tree import make_branching_tree_model
from better_robot.kinematics._warp_bridge import try_warp_forward_kinematics
from better_robot.kinematics.forward import (
    forward_kinematics,
    forward_kinematics_raw,
    frame_placements_raw,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

DEVICE = torch.device("cuda:0")
_BATCH = 4


def _model(kind: str):
    if kind == "free_flyer":
        return make_branching_tree_model(dtype=torch.float32).to(device=DEVICE)
    if kind == "panda":
        pytest.importorskip("robot_descriptions")
        from robot_descriptions import panda_description  # noqa: PLC0415

        return load(panda_description.URDF_PATH, device=DEVICE, dtype=torch.float32)
    raise AssertionError(f"unknown shape-batch test model {kind!r}")


def _batched_inputs(model):
    """Return distinct-per-element ``q`` and batched joint placements.

    ``q`` has shape ``(B, nq)`` and joint placements ``(B, njoints, 7)``; the
    frame placements stay unbatched so the joint and frame execution maps
    genuinely differ.
    """
    base = torch.linspace(-0.03, 0.03, model.nv, dtype=torch.float32, device=DEVICE)
    rows = [
        model.integrate(model.q_neutral, base + 0.01 * (index + 1))
        for index in range(_BATCH)
    ]
    q = torch.stack(rows, dim=0).contiguous()

    joint_placements = model.values.joint_placements.expand(_BATCH, model.njoints, 7).clone()
    joint_placements[..., 2, 0] += torch.linspace(0.01, 0.05, _BATCH, dtype=torch.float32, device=DEVICE)
    return q, joint_placements


def _shape_batched_values(model, joint_placements):
    return dataclasses.replace(model.values, joint_placements=joint_placements)


def _torch_outputs(model, values, q):
    result = forward_kinematics_raw(model.structure, values, q)
    frames = frame_placements_raw(model.structure, values, result.joint_pose_world).frame_pose_world
    return result.joint_pose_world, result.joint_pose_local, frames


def _assert_close(actual, expected) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-5, atol=2e-6)


def _weighted_loss(outputs) -> torch.Tensor:
    result = outputs[0].new_zeros(())
    for scale, tensor in zip((0.7, -0.4, 0.2), outputs, strict=True):
        weight = torch.linspace(0.1, 0.9, tensor.numel(), dtype=tensor.dtype, device=tensor.device).reshape(tensor.shape)
        result = result + scale * (weight * tensor).sum()
    return result


@pytest.mark.parametrize("kind", ("free_flyer", "panda"))
def test_cuda_shape_batched_forward_matches_torch(kind: str, monkeypatch) -> None:
    monkeypatch.setattr(forward_module, "_WARNED_WARP_FALLBACKS", set())
    model = _model(kind)
    q, joint_placements = _batched_inputs(model)
    values = _shape_batched_values(model, joint_placements)
    assert tuple(values.joint_placements.shape[:-2]) == (_BATCH,)
    assert tuple(values.frame_placements.shape[:-2]) == ()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = try_warp_forward_kinematics(model.structure, values, q)
    assert result is not None, "shape-batched joint placements must run on the Warp lane"
    assert not [item for item in caught if "Warp FK lane" in str(item.message)]

    _assert_close((result.world, result.local, result.frames), _torch_outputs(model, values, q))


@pytest.mark.parametrize("kind", ("free_flyer", "panda"))
def test_cuda_shape_batched_backward_matches_torch(kind: str) -> None:
    model = _model(kind)
    q_data, joint_data = _batched_inputs(model)

    q = q_data.detach().clone().requires_grad_()
    joint_placements = joint_data.detach().clone().requires_grad_()
    values = _shape_batched_values(model, joint_placements)
    result = try_warp_forward_kinematics(model.structure, values, q)
    assert result is not None
    actual = (result.world, result.local, result.frames)
    actual_gradients = torch.autograd.grad(_weighted_loss(actual), (q, joint_placements))

    q_ref = q_data.detach().clone().requires_grad_()
    joint_ref = joint_data.detach().clone().requires_grad_()
    values_ref = _shape_batched_values(model, joint_ref)
    expected = _torch_outputs(model, values_ref, q_ref)
    expected_gradients = torch.autograd.grad(_weighted_loss(expected), (q_ref, joint_ref))

    assert actual_gradients[0].shape == q_data.shape
    assert actual_gradients[1].shape == joint_data.shape
    _assert_close(actual_gradients, expected_gradients)


def test_cuda_shape_batched_public_opt_in_matches_torch(monkeypatch) -> None:
    monkeypatch.setattr(forward_module, "_WARNED_WARP_FALLBACKS", set())
    model = _model("free_flyer")
    q, joint_placements = _batched_inputs(model)
    rebound = model.with_values(joint_placements=joint_placements)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warp_data = forward_kinematics(rebound, q, compute_frames=True, use_warp=True)
    assert not [item for item in caught if "Warp FK lane" in str(item.message)]

    torch_data = forward_kinematics(rebound, q, compute_frames=True, use_warp=False)
    _assert_close(
        (warp_data.joint_pose_world, warp_data.joint_pose_local, warp_data.frame_pose_world),
        (torch_data.joint_pose_world, torch_data.joint_pose_local, torch_data.frame_pose_world),
    )
