"""Name-to-slice permutations for mixed-DOF, arbitrarily batched vectors."""

from __future__ import annotations

import pytest
import torch

from better_robot.io import ModelBuilder, build_model


@pytest.fixture
def mixed_dof_model():
    builder = ModelBuilder("mixed_dof_permutation")
    base = builder.add_body("base")
    ball_link = builder.add_body("ball_link")
    hinge_link = builder.add_body("hinge_link")
    plane_link = builder.add_body("plane_link")
    builder.add_free_flyer_root("root", child=base)
    builder.add_spherical("ball", parent=base, child=ball_link)
    builder.add_revolute_z("hinge", parent=base, child=hinge_link)
    builder.add_planar("plane", parent=ball_link, child=plane_link)
    return build_model(builder.finalize(), preserve_joint_order=True)


def _reference_name_remap(
    model,
    other_order: tuple[str, ...],
    values: torch.Tensor,
    *,
    tangent: bool,
) -> torch.Tensor:
    """Slow synthetic oracle matching the former per-joint remap structure."""
    dimensions = model.nvs if tangent else model.nqs
    external_slices: dict[str, torch.Tensor] = {}
    offset = 0
    for name in other_order:
        width = dimensions[model.joint_id(name)]
        external_slices[name] = values[..., offset : offset + width]
        offset += width
    return torch.cat(
        [external_slices[name] for joint_id, name in enumerate(model.joint_names) if dimensions[joint_id] > 0],
        dim=-1,
    )


def test_q_permutation_maps_multi_dof_joint_slices(mixed_dof_model):
    external_order = ("plane", "root", "hinge", "ball")
    perm_q, perm_v = mixed_dof_model.q_permutation(external_order)

    assert perm_q.dtype == torch.long
    assert perm_v.dtype == torch.long
    assert perm_q.device == mixed_dof_model.joint_placements.device
    assert perm_v.device == mixed_dof_model.joint_placements.device
    assert perm_q.tolist() == [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 11, 0, 1, 2, 3]
    assert perm_v.tolist() == [3, 4, 5, 6, 7, 8, 10, 11, 12, 9, 0, 1, 2]


def test_q_permutation_roundtrip_and_batched_trailing_gather(mixed_dof_model):
    external_order = ("plane", "root", "hinge", "ball")
    perm_q, perm_v = mixed_dof_model.q_permutation(external_order)
    q_external = torch.arange(
        2 * 3 * mixed_dof_model.nq,
        dtype=torch.float32,
    ).reshape(2, 3, mixed_dof_model.nq)
    v_external = torch.arange(
        2 * 3 * mixed_dof_model.nv,
        dtype=torch.float32,
    ).reshape(2, 3, mixed_dof_model.nv)

    q_model = q_external[..., perm_q]
    v_model = v_external[..., perm_v]
    torch.testing.assert_close(
        q_model,
        _reference_name_remap(
            mixed_dof_model,
            external_order,
            q_external,
            tangent=False,
        ),
    )
    torch.testing.assert_close(
        v_model,
        _reference_name_remap(
            mixed_dof_model,
            external_order,
            v_external,
            tangent=True,
        ),
    )
    torch.testing.assert_close(q_model[..., torch.argsort(perm_q)], q_external)
    torch.testing.assert_close(v_model[..., torch.argsort(perm_v)], v_external)


def test_q_permutation_identity_accepts_zero_dof_names(mixed_dof_model):
    perm_q, perm_v = mixed_dof_model.q_permutation(mixed_dof_model.joint_names)
    torch.testing.assert_close(perm_q, torch.arange(mixed_dof_model.nq))
    torch.testing.assert_close(perm_v, torch.arange(mixed_dof_model.nv))


@pytest.mark.parametrize(
    ("order", "match"),
    [
        (("root", "ball", "hinge", "plane", "ball"), "duplicate"),
        (("root", "ball", "hinge", "plane", "ghost"), "unknown"),
        (("root", "ball", "hinge"), "missing"),
    ],
)
def test_q_permutation_rejects_invalid_joint_orders(
    mixed_dof_model,
    order,
    match,
):
    with pytest.raises(ValueError, match=match):
        mixed_dof_model.q_permutation(order)
