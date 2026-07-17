"""Grouped ``Model.integrate``/``difference`` parity and contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from better_robot.data_model.joint_models import (
    JointComposite,
    JointPX,
    JointRX,
    JointTranslation,
)
from better_robot.exceptions import DeviceMismatchError, ShapeError
from better_robot.io import ModelBuilder, build_model, load
from better_robot.io.builders.smpl_like import make_smpl_like_model


def _loop_integrate(model, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The pre-M3 per-joint implementation, retained as a test oracle."""

    dtype = torch.promote_types(q.dtype, v.dtype)
    q = q.to(dtype=dtype)
    v = v.to(dtype=dtype)
    parts: list[torch.Tensor] = []
    for joint_id, joint in enumerate(model.joint_models):
        nq_joint = model.nqs[joint_id]
        nv_joint = model.nvs[joint_id]
        if nq_joint == 0:
            continue
        iq = model.idx_qs[joint_id]
        iv = model.idx_vs[joint_id]
        parts.append(
            joint.integrate(
                q[..., iq : iq + nq_joint],
                v[..., iv : iv + nv_joint],
            )
        )
    if not parts:
        batch_shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
        dtype = torch.promote_types(q.dtype, v.dtype)
        return q.to(dtype=dtype).expand(*batch_shape, model.nq).clone()
    return torch.cat(parts, dim=-1)


def _loop_difference(model, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """The pre-M3 per-joint implementation, retained as a test oracle."""

    dtype = torch.promote_types(q0.dtype, q1.dtype)
    q0 = q0.to(dtype=dtype)
    q1 = q1.to(dtype=dtype)
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
    if not parts:
        batch_shape = torch.broadcast_shapes(q0.shape[:-1], q1.shape[:-1])
        dtype = torch.promote_types(q0.dtype, q1.dtype)
        return torch.zeros(*batch_shape, model.nv, device=q0.device, dtype=dtype)
    return torch.cat(parts, dim=-1)


def _assert_manifold_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Grouped contiguous reductions can differ from strided per-joint kernels
    # by a few ULPs; the M3 benchmark note records the observed maxima.
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=5e-6, atol=2e-7)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


def _load_named_model(name: str, dtype: torch.dtype):
    if name == "smpl_like":
        return make_smpl_like_model(dtype=dtype)

    pytest.importorskip("robot_descriptions")
    if name == "panda":
        from robot_descriptions import panda_description  # noqa: PLC0415 - optional fixture

        return load(panda_description.URDF_PATH, dtype=dtype)
    if name == "g1":
        from robot_descriptions import g1_description  # noqa: PLC0415 - optional fixture

        return load(g1_description.URDF_PATH, free_flyer=True, dtype=dtype)
    raise ValueError(name)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("model_name", ("panda", "g1", "smpl_like"))
def test_grouped_manifolds_match_per_joint_loop(model_name: str, dtype: torch.dtype) -> None:
    model = _load_named_model(model_name, dtype)
    generator = torch.Generator().manual_seed(20250717)
    tangent = torch.randn(2, 3, model.nv, generator=generator, dtype=dtype) * 0.05
    neutral = model.q_neutral.expand(2, 3, -1).clone()
    q = _loop_integrate(model, neutral, tangent * 0.25)

    expected_q = _loop_integrate(model, q, tangent)
    actual_q = model.integrate(q, tangent)
    _assert_manifold_close(actual_q, expected_q)

    expected_v = _loop_difference(model, q, expected_q)
    actual_v = model.difference(q, expected_q)
    _assert_manifold_close(actual_v, expected_v)


def test_right_aligned_multi_axis_broadcast_matches_loop() -> None:
    model = make_smpl_like_model(dtype=torch.float64)
    q = model.q_neutral.reshape(1, 1, -1).expand(2, 1, -1).clone()
    v = torch.randn(3, model.nv, dtype=torch.float64) * 0.02

    actual_q = model.integrate(q, v)
    expected_q = _loop_integrate(model, q, v)
    assert actual_q.shape == (2, 3, model.nq)
    _assert_manifold_close(actual_q, expected_q)

    q1 = expected_q[:1]
    actual_v = model.difference(q, q1)
    expected_v = _loop_difference(model, q, q1)
    assert actual_v.shape == (2, 3, model.nv)
    _assert_manifold_close(actual_v, expected_v)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_theta_zero_roundtrip_and_gradients_are_finite(dtype: torch.dtype) -> None:
    model = make_smpl_like_model(dtype=dtype)
    q = model.q_neutral.expand(2, -1).clone().requires_grad_(True)
    v = (torch.randn(2, model.nv, dtype=dtype) * 1e-3).requires_grad_(True)

    q_next = model.integrate(q, v)
    recovered = model.difference(q, q_next)
    _assert_manifold_close(recovered, v)

    (q_next.square().sum() + recovered.square().sum()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@dataclass(frozen=True)
class _ScaledCustomJoint(JointRX):
    """Out-of-tree subclass whose retraction proves exact-class fallback."""

    kind: str = "scaled_custom"

    def integrate(self, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor:
        return q_slice + 2.0 * v_slice

    def difference(self, q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor:
        return 0.5 * (q1_slice - q0_slice)


def _mixed_manifold_model():
    builder = ModelBuilder("mixed_manifolds")
    parent = builder.add_body("root")
    builder.add_free_flyer_root(child=parent)

    def child(name: str) -> str:
        nonlocal parent
        next_body = builder.add_body(f"{name}_body")
        previous = parent
        parent = next_body
        return previous

    previous = child("scalar")
    builder.add_revolute_x("scalar", parent=previous, child=parent)
    previous = child("translation")
    builder.add_joint("translation", kind=JointTranslation(), parent=previous, child=parent)
    previous = child("helical")
    builder.add_helical(
        "helical",
        parent=previous,
        child=parent,
        axis=torch.tensor([0.0, 0.0, 1.0]),
        pitch=0.2,
    )
    previous = child("spherical")
    builder.add_spherical("spherical", parent=previous, child=parent)
    previous = child("continuous")
    builder.add_revolute(
        "continuous",
        parent=previous,
        child=parent,
        axis=torch.tensor([0.0, 0.0, 1.0]),
        unbounded=True,
    )
    previous = child("planar")
    builder.add_planar("planar", parent=previous, child=parent)
    previous = child("composite")
    builder.add_joint(
        "composite",
        kind=JointComposite((JointRX(), JointPX())),
        parent=previous,
        child=parent,
    )
    previous = child("custom")
    builder.add_joint(
        "custom",
        kind=_ScaledCustomJoint(),
        parent=previous,
        child=parent,
    )
    return build_model(builder.finalize(), dtype=torch.float64)


def test_mixed_groups_flatten_scalar_and_translation_and_fallback_exactly() -> None:
    model = _mixed_manifold_model()
    structure = model.structure
    euclidean_ids = set(structure.manifold_euclidean_joint_ids)
    fallback_ids = set(structure.manifold_fallback_joint_ids)

    assert model.joint_id("scalar") in euclidean_ids
    assert model.joint_id("translation") in euclidean_ids
    assert structure.manifold_euclidean_q_indices.numel() == 5  # scalar + xyz + helical
    assert model.joint_id("composite") in fallback_ids
    assert model.joint_id("custom") in fallback_ids
    assert structure.manifold_spherical_q_indices.shape == (1, 4)
    assert structure.manifold_free_flyer_q_indices.shape == (1, 7)
    assert structure.manifold_unbounded_q_indices.shape == (1, 2)
    assert structure.manifold_planar_q_indices.shape == (1, 4)

    # Mixed q/v dtypes follow torch promotion, including both fallback joints.
    q = model.q_neutral.to(dtype=torch.float32).expand(4, -1).clone()
    v = torch.randn(4, model.nv, dtype=torch.float64) * 0.02
    expected_q = _loop_integrate(model, q, v)
    actual_q = model.integrate(q, v)
    assert actual_q.dtype == torch.float64
    _assert_manifold_close(actual_q, expected_q)
    _assert_manifold_close(
        model.difference(q, expected_q),
        _loop_difference(model, q, expected_q),
    )


def test_reduced_mimic_target_is_not_a_public_manifold_group() -> None:
    builder = ModelBuilder("mimic_manifold_group")
    root = builder.add_body("root")
    source = builder.add_body("source")
    target = builder.add_body("target")
    builder.add_revolute_z("source_joint", parent=root, child=source)
    builder.add_revolute_z(
        "target_joint",
        parent=root,
        child=target,
        mimic_source="source_joint",
        mimic_multiplier=-0.5,
        mimic_offset=0.1,
    )
    model = build_model(builder.finalize(), dtype=torch.float64)

    source_id = model.joint_id("source_joint")
    target_id = model.joint_id("target_joint")
    assert source_id in model.structure.manifold_euclidean_joint_ids
    assert target_id not in model.structure.manifold_euclidean_joint_ids
    assert target_id not in model.structure.manifold_fallback_joint_ids
    assert model.nqs[target_id] == model.nvs[target_id] == 0

    q = torch.tensor([0.2], dtype=torch.float64)
    v = torch.tensor([-0.05], dtype=torch.float64)
    _assert_manifold_close(model.difference(q, model.integrate(q, v)), v)


def test_zero_dof_models_still_broadcast_and_promote_dtype() -> None:
    builder = ModelBuilder("zero_dof")
    builder.add_body("base")
    model = build_model(builder.finalize())
    q0 = torch.empty(2, 1, 0, dtype=torch.float32)
    q1 = torch.empty(3, 0, dtype=torch.float64)

    integrated = model.integrate(q0, q1)
    difference = model.difference(q0, q1)
    assert integrated.shape == (2, 3, 0)
    assert difference.shape == (2, 3, 0)
    assert integrated.dtype == difference.dtype == torch.float64


def test_trailing_dimension_and_device_errors_are_explicit() -> None:
    model = make_smpl_like_model()
    with pytest.raises(ShapeError, match=r"q has shape .* trailing dimension 99"):
        model.integrate(torch.zeros(model.nq - 1), torch.zeros(model.nv))
    with pytest.raises(ShapeError, match=r"v has shape .* trailing dimension 75"):
        model.integrate(torch.zeros(model.nq), torch.zeros(model.nv + 1))
    with pytest.raises(ShapeError, match=r"q1 has shape .* trailing dimension 99"):
        model.difference(torch.zeros(model.nq), torch.zeros(model.nq + 1))

    q_meta = torch.empty(model.nq, device="meta")
    with pytest.raises(DeviceMismatchError, match=r"q.device=meta != model.device=cpu"):
        model.integrate(q_meta, torch.zeros(model.nv))

    moved = model.to(dtype=torch.float64)
    assert moved.structure.manifold_euclidean_q_indices.dtype == torch.long
    assert moved.structure.manifold_spherical_q_indices.dtype == torch.long
