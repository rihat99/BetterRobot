"""Singularity and manifold-correctness tests for tangent-space autograd."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from better_robot.io import load
from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.kinematics import forward_kinematics
from better_robot.optim import (
    Euclidean,
    RobotConfig,
    SE3Manifold,
    SO3Manifold,
    Values,
    VarSpec,
)
from better_robot.optim.blocks.autograd import perturb_values, tangent_grad
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual


_Observable = Callable[[torch.Tensor], torch.Tensor]


@pytest.fixture(scope="module")
def panda_model():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    return load(panda_description.URDF_PATH, dtype=torch.float32)


def _identity_so3() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 1.0])


def _identity_se3() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _singular_case(case: str) -> tuple[VarSpec, torch.Tensor, _Observable, bool]:
    if case == "euclidean":
        manifold = Euclidean()
        value = torch.zeros(3)
        return VarSpec("value", (3,), manifold), value, lambda x: x, True
    if case == "so3_identity":
        manifold = SO3Manifold()
        value = _identity_so3()
        return (
            VarSpec("value", (4,), manifold),
            value,
            lambda x: manifold.difference(value, x),
            True,
        )
    if case == "se3_identity":
        manifold = SE3Manifold()
        value = _identity_se3()
        return (
            VarSpec("value", (7,), manifold),
            value,
            lambda x: manifold.difference(value, x),
            True,
        )
    if case == "identical_quaternions":
        manifold = SO3Manifold()
        value = torch.tensor([0.2, -0.15, 0.1])
        value = manifold.retract(_identity_so3(), value)
        return (
            VarSpec("value", (4,), manifold),
            value,
            lambda x: manifold.difference(value, x),
            True,
        )
    if case == "near_pi":
        manifold = SO3Manifold()
        axis = torch.tensor([1.0, 0.2, -0.1])
        axis = axis / axis.norm()
        value = manifold.retract(_identity_so3(), axis * (math.pi - 0.05))
        identity = _identity_so3()
        return (
            VarSpec("value", (4,), manifold),
            value,
            lambda x: manifold.difference(identity, x),
            False,
        )
    raise AssertionError(f"unknown case {case}")


@pytest.mark.parametrize(
    "case",
    ("euclidean", "so3_identity", "se3_identity", "identical_quaternions", "near_pi"),
)
def test_tangent_perturbation_gradcheck_at_singular_points(case: str) -> None:
    spec, value, observable, expect_identity_gradient = _singular_case(case)
    zero = torch.zeros(spec.free_dim, dtype=torch.float32, requires_grad=True)

    def closure(delta: torch.Tensor) -> torch.Tensor:
        perturbed = perturb_values(
            (spec,),
            {spec.name: value},
            {spec.name: delta},
        )
        return observable(perturbed[spec.name])

    assert torch.autograd.gradcheck(
        closure,
        (zero,),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True,
    )

    gradient = tangent_grad(
        lambda values: observable(values[spec.name]),
        (spec,),
        {spec.name: value},
    )[spec.name]
    assert gradient.shape == (spec.free_dim,)
    assert torch.isfinite(gradient).all()
    if expect_identity_gradient:
        torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=2e-4, rtol=2e-4)


def test_tangent_grad_returns_only_free_masked_coordinates() -> None:
    spec = VarSpec(
        name="x",
        shape=(4,),
        manifold=Euclidean(),
        mask=torch.tensor([True, False, True, False]),
    )
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])

    gradient = tangent_grad(
        lambda values: values["x"].square(),
        (spec,),
        {"x": value},
    )["x"]

    assert gradient.shape == (2,)
    torch.testing.assert_close(gradient, torch.tensor([2.0, 6.0]))


def test_create_graph_supports_a_second_derivative_smoke() -> None:
    spec = VarSpec(name="x", shape=(3,), manifold=Euclidean())
    value = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)

    first = tangent_grad(
        lambda values: values["x"].square(),
        (spec,),
        {"x": value},
        create_graph=True,
    )["x"]
    second = torch.autograd.grad(first.sum(), value)[0]

    torch.testing.assert_close(first, 2.0 * value)
    torch.testing.assert_close(second, torch.full_like(value, 2.0))


def test_create_graph_keeps_constant_and_unused_block_zeros_connected() -> None:
    specs = (
        VarSpec(name="x", shape=(1,)),
        VarSpec(name="unused", shape=(1,)),
    )
    x = torch.tensor([0.5], requires_grad=True)
    unused = torch.tensor([1.2], requires_grad=True)

    gradients = tangent_grad(
        lambda values: values["x"],
        specs,
        {"x": x, "unused": unused},
        create_graph=True,
    )
    second_x, second_unused = torch.autograd.grad(
        gradients["x"].sum() + gradients["unused"].sum(),
        (x, unused),
    )

    torch.testing.assert_close(gradients["x"], torch.ones(1))
    torch.testing.assert_close(gradients["unused"], torch.zeros(1))
    torch.testing.assert_close(second_x, torch.zeros(1))
    torch.testing.assert_close(second_unused, torch.zeros(1))


def test_spherical_tree_rest_pose_tangent_gradient_is_finite() -> None:
    model = make_smpl_like_model(dtype=torch.float32)
    spec = VarSpec(
        name="q",
        shape=(model.nq,),
        manifold=RobotConfig(model),
    )
    neutral = model.q_neutral
    batched = neutral.expand(2, 3, model.nq).clone()

    gradient = tangent_grad(
        lambda values: model.difference(neutral, values["q"]),
        (spec,),
        {"q": batched},
    )["q"]

    assert gradient.shape == (2, 3, model.nv)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=2e-4, rtol=2e-4)


def test_panda_pose_tangent_gradient_matches_existing_analytic_path(panda_model) -> None:
    model = panda_model
    q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
    data = forward_kinematics(model, q, compute_frames=True)
    frame_name = "body_panda_hand" if "body_panda_hand" in model.frame_name_to_id else model.frame_names[-1]
    frame_id = model.frame_id(frame_name)
    target = data.frame_pose_world[frame_id].detach().clone()
    residual = PoseResidual(frame_id=frame_id, target=target)
    state = ResidualState(model=model, data=data, variables=q)
    cotangent = torch.tensor([0.7, -0.4, 0.2, -0.3, 0.5, 0.6])
    analytic_jacobian = residual.jacobian(state)
    assert analytic_jacobian is not None
    expected = analytic_jacobian.mT @ cotangent

    spec = VarSpec(name="q", shape=(model.nq,), manifold=RobotConfig(model))

    def objective(values: Values) -> torch.Tensor:
        q_value = values["q"]
        value_data = forward_kinematics(model, q_value, compute_frames=True)
        value_state = ResidualState(model=model, data=value_data, variables=q_value)
        return residual(value_state) @ cotangent

    actual = tangent_grad(objective, (spec,), {"q": q})["q"]

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
