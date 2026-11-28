"""Tests for ``src/better_robot/model/model.py``."""

import pytest
import torch

from better_robot.model.model import Model
from better_robot.exceptions import ArgumentError, DtypeMismatchError, ModelError
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import support
from better_robot.model import (
    JointHelical,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
)

from tests.support.robots import two_link_arm


def test_model_random_configuration():
    model = two_link_arm(torch.float32)
    q = model.random_configuration(generator=torch.Generator().manual_seed(0))
    assert q.shape == (2,)
    assert (q >= model.lower_position_limit).all()
    assert (q <= model.upper_position_limit).all()


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.complex64, "float64"], ids=str)
def test_model_to_refuses_a_dtype_other_than_float32_or_float64(dtype):
    with pytest.raises(DtypeMismatchError, match="is not float32 or float64; pass torch.float32 or torch.float64"):
        two_link_arm(torch.float32).to(dtype=dtype)


def test_model_support():
    model = two_link_arm(torch.float32)
    sp = support(model, joint=3)
    assert sp == (0, 1, 2, 3)


def _make_floating_model(dtype: torch.dtype) -> Model:
    """Free-flyer root plus one revolute joint without position limits."""
    builder = ModelBuilder("floating")
    root = builder.add_body("root")
    link = builder.add_body("link")
    builder.add_free_flyer_root("root_joint", child=root)
    builder.add_revolute("spin", axis=(0.0, 0.0, 1.0), parent=root, child=link)
    return build_model(builder.finalize()).to(dtype=dtype)


def test_random_configuration_follows_model_dtype():
    builder = ModelBuilder("root_only")
    builder.add_free_flyer_root("root_joint", child=builder.add_body("root"))
    model = build_model(builder.finalize()).to(dtype=torch.float64)
    assert model.random_configuration(generator=torch.Generator().manual_seed(0)).dtype == torch.float64


def test_random_configuration_is_finite_for_unlimited_revolute():
    model = _make_floating_model(torch.float64)
    assert not torch.isfinite(model.lower_position_limit[-1])
    generator = torch.Generator().manual_seed(0)
    samples = torch.stack([model.random_configuration(generator=generator) for _ in range(32)])
    assert torch.isfinite(samples).all()
    assert (samples[:, -1].abs() <= torch.pi).all()


def _every_kind_model() -> Model:
    """A chain with one joint of every kind, limits ``[-0.5, 0.5]``, under a free-flyer root."""
    builder = ModelBuilder("every_kind")
    parent = builder.add_body("root")
    builder.add_free_flyer_root("root_joint", child=parent)
    axis = (0.0, 0.0, 1.0)
    kinds = (
        JointRevolute(axis=axis),
        JointRevoluteUnbounded(axis=axis),
        JointPrismatic(axis=axis),
        JointHelical(axis=axis, pitch=0.1),
        JointSpherical(),
        JointPlanar(),
        JointTranslation(),
    )
    for index, kind in enumerate(kinds):
        child = builder.add_body(f"link{index}")
        builder.add_joint(f"joint{index}", kind=kind, parent=parent, child=child)
        parent = child
    model = build_model(builder.finalize())
    lower, upper = model.lower_position_limit.clamp(min=-0.5), model.upper_position_limit.clamp(max=0.5)
    return model.with_values(lower_position_limit=lower, upper_position_limit=upper)


def test_random_configuration_of_a_meta_model_reads_no_values():
    model = _every_kind_model().to(device="meta")
    q = model.random_configuration(generator=torch.Generator())
    assert q.device.type == "meta" and q.shape == (model.nq,)


@pytest.mark.cuda
def test_random_configuration_draws_on_the_model_device():
    model = _every_kind_model().to(device="cuda")
    q = model.random_configuration(generator=torch.Generator(device="cuda").manual_seed(0))
    assert q.device == model.device and q.shape == (model.nq,)
    assert bool(torch.isfinite(q).all())


def test_random_configuration_requires_a_generator():
    """A draw never falls back to the global random stream (principle 5.1)."""
    with pytest.raises(ArgumentError, match="generator"):
        two_link_arm(torch.float32).random_configuration(generator=None)


def test_random_configuration_rejects_unlimited_prismatic():
    builder = ModelBuilder("slider")
    base = builder.add_body("base")
    link = builder.add_body("link")
    builder.add_prismatic("slide", axis=(0.0, 0.0, 1.0), parent=base, child=link)
    model = build_model(builder.finalize())
    with pytest.raises(ModelError, match="slide"):
        model.random_configuration(generator=torch.Generator().manual_seed(0))
