"""The two-lane structure/value seam has one consistent topology."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.utils import _pytree

from better_robot.data_model.model_values import packed_inertias_to_6x6
from better_robot.io import load
from better_robot.io.builders.smpl_like import make_smpl_like_body


@pytest.fixture(scope="module", params=("panda", "smpl"))
def model(request):
    if request.param == "smpl":
        return load(make_smpl_like_body)
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    return load(panda_description.URDF_PATH)


def test_dual_topology_representations_are_consistent(model):
    structure = model.structure
    structure.validate_consistency()
    assert tuple(structure.parents_tensor.cpu().tolist()) == model.parents
    assert tuple(structure.joint_kind_tensor.cpu().tolist()) == structure.joint_kind_codes
    assert tuple(structure.idx_qs_tensor.cpu().tolist()) == model.idx_qs
    assert tuple(structure.idx_vs_tensor.cpu().tolist()) == model.idx_vs
    assert structure.joint_motion_subspaces.shape == (
        model.njoints,
        6,
        max(model.nvs, default=0),
    )

    expected_axes = []
    for joint in model.joint_models:
        if joint.kind.endswith("_rx") or joint.kind.endswith("_px"):
            expected_axes.append((1.0, 0.0, 0.0))
        elif joint.kind.endswith("_ry") or joint.kind.endswith("_py"):
            expected_axes.append((0.0, 1.0, 0.0))
        elif joint.kind.endswith("_rz") or joint.kind.endswith("_pz"):
            expected_axes.append((0.0, 0.0, 1.0))
        elif joint.axis is not None:
            expected_axes.append(tuple(float(v) for v in joint.axis))
        else:
            expected_axes.append((0.0, 0.0, 0.0))
    torch.testing.assert_close(
        structure.joint_axes.cpu(),
        torch.tensor(expected_axes, dtype=structure.joint_axes.dtype),
    )


def test_model_values_is_tensor_pytree_and_frames_move(model):
    leaves, spec = _pytree.tree_flatten(model.values)
    assert leaves
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert rebuilt.frame_placements.shape == (model.nframes, 7)

    moved = model.to(dtype=torch.float64)
    assert moved.values.frame_placements.dtype == torch.float64
    assert all(frame.joint_placement.dtype == torch.float64 for frame in moved.frames)
    assert moved.structure.parents_tensor.dtype == torch.int32


def test_dataclasses_replace_keeps_spatial_inertia_physics_and_gradients_live(model):
    replacement = model.values.body_inertias.detach().clone()
    replacement[..., 0] = replacement[..., 0] + 0.25
    replacement.requires_grad_()
    replaced = dataclasses.replace(model.values, body_inertias=replacement)

    spatial = replaced.spatial_inertias()
    torch.testing.assert_close(spatial, packed_inertias_to_6x6(replacement))
    assert not torch.equal(spatial, model.values.spatial_inertias())

    gradient = torch.autograd.grad(spatial.square().sum(), replacement)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
