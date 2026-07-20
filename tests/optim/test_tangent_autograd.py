"""Singularity and manifold-correctness tests for tangent-space autograd."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math

import pytest
import torch

from better_robot.io import load
from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.kinematics import forward_kinematics
from better_robot.optim import Problem, RobotVariable, SE3Variable, SO3Variable, Variable
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


def _retracted_gradient(
    fn: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    variables: Sequence[Variable],
    *,
    create_graph: bool = False,
) -> dict[str, torch.Tensor]:
    deltas = {
        variable.name: variable.tensor.new_zeros(*variable.batch_shape, variable.free_dim, requires_grad=True)
        for variable in variables
    }
    values = {variable.name: variable.retract(deltas[variable.name]) for variable in variables}
    output = fn(values)
    computed = torch.autograd.grad(
        output.sum(),
        tuple(deltas.values()),
        create_graph=create_graph,
        allow_unused=True,
    )
    anchors = [variable.tensor.sum() * 0.0 for variable in variables if create_graph and variable.tensor.requires_grad]
    anchor = sum(anchors[1:], anchors[0]) if anchors else None
    result: dict[str, torch.Tensor] = {}
    for variable, gradient in zip(variables, computed, strict=True):
        value = torch.zeros_like(deltas[variable.name]) if gradient is None else gradient
        result[variable.name] = value + anchor if anchor is not None else value
    return result


def _singular_case(case: str) -> tuple[Variable, _Observable, bool]:
    if case == "euclidean":
        variable = Variable(torch.zeros(3), name="value")
        return variable, lambda x: x, True
    if case == "so3_identity":
        value = _identity_so3()
        variable = SO3Variable(value, name="value")
        return variable, lambda x: variable._difference_from(value, x), True
    if case == "se3_identity":
        value = _identity_se3()
        variable = SE3Variable(value, name="value")
        return variable, lambda x: variable._difference_from(value, x), True
    if case == "identical_quaternions":
        tangent = torch.tensor([0.2, -0.15, 0.1])
        value = SO3Variable(_identity_so3()).retract(tangent)
        variable = SO3Variable(value, name="value")
        return variable, lambda x: variable._difference_from(value, x), True
    if case == "near_pi":
        axis = torch.tensor([1.0, 0.2, -0.1])
        axis = axis / axis.norm()
        identity = _identity_so3()
        value = SO3Variable(identity).retract(axis * (math.pi - 0.05))
        variable = SO3Variable(value, name="value")
        return variable, lambda x: variable._difference_from(identity, x), False
    raise AssertionError(f"unknown case {case}")


@pytest.mark.parametrize(
    "case",
    ("euclidean", "so3_identity", "se3_identity", "identical_quaternions", "near_pi"),
)
def test_tangent_perturbation_gradcheck_at_singular_points(case: str) -> None:
    variable, observable, expect_identity_gradient = _singular_case(case)
    zero = variable.tensor.new_zeros(variable.free_dim, requires_grad=True)

    def closure(delta: torch.Tensor) -> torch.Tensor:
        return observable(variable.retract(delta))

    assert torch.autograd.gradcheck(
        closure,
        (zero,),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        fast_mode=True,
    )

    gradient = _retracted_gradient(
        lambda values: observable(values[variable.name]),
        (variable,),
    )[variable.name]
    assert gradient.shape == (variable.free_dim,)
    assert torch.isfinite(gradient).all()
    if expect_identity_gradient:
        torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=2e-4, rtol=2e-4)


def test_create_graph_supports_a_second_derivative_smoke() -> None:
    value = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    x = Variable(value, name="x")

    first = _retracted_gradient(
        lambda values: values["x"].square(),
        (x,),
        create_graph=True,
    )["x"]
    second = torch.autograd.grad(first.sum(), value)[0]

    torch.testing.assert_close(first, 2.0 * value)
    torch.testing.assert_close(second, torch.full_like(value, 2.0))


def test_create_graph_keeps_constant_and_unused_block_zeros_connected() -> None:
    x_tensor = torch.tensor([0.5], requires_grad=True)
    unused_tensor = torch.tensor([1.2], requires_grad=True)
    x = Variable(x_tensor, name="x")
    unused = Variable(unused_tensor, name="unused")

    gradients = _retracted_gradient(
        lambda values: values["x"],
        (x, unused),
        create_graph=True,
    )
    second_x, second_unused = torch.autograd.grad(
        gradients["x"].sum() + gradients["unused"].sum(),
        (x_tensor, unused_tensor),
    )

    torch.testing.assert_close(gradients["x"], torch.ones(1))
    torch.testing.assert_close(gradients["unused"], torch.zeros(1))
    torch.testing.assert_close(second_x, torch.zeros(1))
    torch.testing.assert_close(second_unused, torch.zeros(1))


def test_spherical_tree_rest_pose_tangent_gradient_is_finite() -> None:
    model = make_smpl_like_model(dtype=torch.float32)
    neutral = model.q_neutral
    q = RobotVariable(model, neutral.expand(2, 3, model.nq).clone(), name="q")

    gradient = _retracted_gradient(
        lambda values: model.difference(neutral, values["q"]),
        (q,),
    )["q"]

    assert gradient.shape == (2, 3, model.nv)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=2e-4, rtol=2e-4)


def test_panda_pose_tangent_gradient_matches_existing_analytic_path(panda_model) -> None:
    model = panda_model
    q_tensor = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
    data = forward_kinematics(model, q_tensor, compute_frames=True)
    frame_name = "body_panda_hand" if "body_panda_hand" in model.frame_name_to_id else model.frame_names[-1]
    frame_id = model.frame_id(frame_name)
    target = data.frame_pose_world[frame_id].detach().clone()
    q = RobotVariable(model, q_tensor, name="q")
    problem = Problem([PoseResidual(q, frame_id=frame_id, target=target, name="pose")])
    cotangent = torch.tensor([0.7, -0.4, 0.2, -0.3, 0.5, 0.6])

    analytic_jacobian = problem.dense_jacobian(strategy="analytic")
    autodiff_jacobian = problem.dense_jacobian(strategy="jacrev")
    expected = analytic_jacobian.mT @ cotangent
    actual = autodiff_jacobian.mT @ cotangent

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
