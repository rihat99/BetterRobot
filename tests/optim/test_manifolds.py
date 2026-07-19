"""Geometry and validation regressions for object-owned variables."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.io import ModelBuilder, build_model, load
from better_robot.optim import Bounds, RobotVariable, SE3Variable, SO3Variable, Variable

_BATCH_SHAPES = ((), (4,), (2, 3))


def _pattern(batch_shape: tuple[int, ...], width: int, *, scale: float) -> torch.Tensor:
    return torch.linspace(-scale, scale, math.prod(batch_shape) * width).reshape(*batch_shape, width)


@pytest.fixture(scope="module")
def panda_model():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    return load(panda_description.URDF_PATH, dtype=torch.float32)


@pytest.fixture(scope="module")
def floating_spherical_model():
    builder = ModelBuilder("floating_spherical")
    base = builder.add_body("base", mass=1.0)
    middle = builder.add_body("middle", mass=1.0)
    tip = builder.add_body("tip", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    builder.add_spherical("ball", parent=base, child=middle)
    builder.add_revolute_z("hinge", parent=middle, child=tip, lower=-0.25, upper=0.25)
    return build_model(builder.finalize(), dtype=torch.float32)


@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize("kind", ("euclidean", "so3", "se3"))
def test_retract_difference_roundtrip(kind: str, batch_shape: tuple[int, ...]) -> None:
    if kind == "euclidean":
        value = _pattern(batch_shape, 5, scale=0.3)
        variable = Variable(value, batch_ndim=len(batch_shape))
        delta = _pattern(batch_shape, 5, scale=0.02)
    elif kind == "so3":
        value = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(*batch_shape, 4).clone()
        variable = SO3Variable(value)
        delta = _pattern(batch_shape, 3, scale=0.02)
    else:
        value = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(*batch_shape, 7).clone()
        variable = SE3Variable(value)
        delta = _pattern(batch_shape, 6, scale=0.02)
    actual = variable._difference_from(value, variable.retract(delta))
    torch.testing.assert_close(actual, delta, rtol=2e-4, atol=2e-5)


def _assert_robot_variable(model, batch_shape: tuple[int, ...]) -> None:
    q = model.q_neutral.expand(*batch_shape, model.nq).clone()
    variable = RobotVariable(model, q)
    delta = _pattern(batch_shape, model.nv, scale=0.01)
    integrated = variable.retract(delta)
    torch.testing.assert_close(integrated, model.integrate(q, delta), rtol=0.0, atol=0.0)
    torch.testing.assert_close(variable._difference_from(q, integrated), delta, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
def test_panda_robot_variable_parity(panda_model, batch_shape) -> None:
    _assert_robot_variable(panda_model, batch_shape)


@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
def test_floating_spherical_robot_variable_parity(floating_spherical_model, batch_shape) -> None:
    assert floating_spherical_model.nq != floating_spherical_model.nv
    _assert_robot_variable(floating_spherical_model, batch_shape)


def _floating_bounds(model) -> Bounds:
    lower, upper = torch.full((model.nq,), -torch.inf), torch.full((model.nq,), torch.inf)
    lower[0], upper[0], lower[-1], upper[-1] = -0.5, 0.5, -0.25, 0.25
    return Bounds(lower, upper)


def test_retraction_clamps_only_box_coordinates_and_preserves_units(floating_spherical_model) -> None:
    model = floating_spherical_model
    variable = RobotVariable(model, model.q_neutral.clone(), bounds=_floating_bounds(model))
    delta = torch.zeros(model.nv)
    delta[0], delta[3:6], delta[6:9], delta[-1] = (
        2.0,
        torch.tensor([0.1, -0.2, 0.15]),
        torch.tensor([0.2, 0.1, -0.1]),
        1.0,
    )
    unconstrained = model.integrate(variable.tensor, delta)
    projected = variable.retract(delta)
    assert projected[0].item() == pytest.approx(0.5)
    assert projected[-1].item() == pytest.approx(0.25)
    torch.testing.assert_close(projected[3:11], unconstrained[3:11])
    torch.testing.assert_close(projected[3:7].norm(), torch.tensor(1.0))
    torch.testing.assert_close(projected[7:11].norm(), torch.tensor(1.0))


@pytest.mark.parametrize(("cls", "width", "kind"), ((SO3Variable, 4, "SO3"), (SE3Variable, 7, "SE3")))
def test_group_bounds_raise_actionably(cls, width: int, kind: str) -> None:
    bounds = Bounds(torch.full((width,), -torch.inf), torch.full((width,), torch.inf))
    identity = torch.zeros(width)
    identity[-1] = 1.0
    with pytest.raises(ValueError, match=rf"{kind} variables have no meaningful global box bound"):
        cls(identity, name="orientation", bounds=bounds)


def test_robot_variable_rejects_bounds_on_quaternion_coordinates(floating_spherical_model) -> None:
    bounds = _floating_bounds(floating_spherical_model)
    lower = bounds.lower.clone()
    lower[3] = -1.0
    with pytest.raises(ValueError, match="non-box manifold coordinates"):
        RobotVariable(floating_spherical_model, bounds=Bounds(lower, bounds.upper))


def test_mask_eliminates_fixed_tangent_coordinates() -> None:
    variable = Variable(torch.zeros(4), mask=torch.tensor([True, False, True, False]))
    reduced = torch.tensor([0.25, -0.5])
    expanded = variable.expand_tangent(reduced)
    assert variable.tangent_dim() == 4 and variable.free_indices.tolist() == [0, 2]
    torch.testing.assert_close(expanded, torch.tensor([0.25, 0.0, -0.5, 0.0]))
    torch.testing.assert_close(variable.gather_tangent(expanded), reduced)
    torch.testing.assert_close(variable.retract(reduced), expanded)


def test_scalar_event_shape_supports_independent_batch_axes() -> None:
    scalar = Variable(torch.tensor(1.0))
    torch.testing.assert_close(scalar.retract(torch.tensor([0.2])), torch.tensor(1.2))
    values = torch.tensor([0.2, -0.3, 0.5])
    batched = Variable(values, batch_ndim=1)
    assert batched.batch_shape == (3,) and batched.tangent_dim() == batched.free_dim == 1
    torch.testing.assert_close(batched.retract(torch.tensor([[0.1], [0.2], [-0.4]])), torch.tensor([0.3, -0.1, 0.1]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.int64))
def test_values_reject_unsupported_working_dtypes(dtype: torch.dtype) -> None:
    with pytest.raises(DtypeMismatchError, match="torch.float32 or torch.float64"):
        Variable(torch.ones(2, dtype=dtype))


def test_retraction_and_difference_reject_mixed_dtype() -> None:
    variable = Variable(torch.ones(2))
    with pytest.raises(DtypeMismatchError, match="delta and value must share dtype"):
        variable.retract(torch.zeros(2, dtype=torch.float64))
    with pytest.raises(DtypeMismatchError, match="difference inputs must share dtype"):
        variable._difference_from(torch.ones(2), torch.zeros(2, dtype=torch.float64))
    assert torch.isnan(variable._difference_from(torch.ones(2), torch.tensor([0.0, float("nan")]))[-1])


def test_bounds_reject_nan_endpoints() -> None:
    with pytest.raises(ValueError, match="must not contain NaN"):
        Bounds(torch.tensor([float("nan")]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="must not contain NaN"):
        Bounds(torch.tensor([-1.0]), torch.tensor([float("nan")]))


def test_value_validation_is_structural(floating_spherical_model) -> None:
    SO3Variable(torch.zeros(4))
    SE3Variable(torch.zeros(7))
    q = floating_spherical_model.q_neutral.clone()
    q[3:7] = 0.0
    RobotVariable(floating_spherical_model, q)
