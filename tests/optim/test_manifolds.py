"""Contracts for the draft state-manifold and variable-block primitives."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError, QuaternionNormError
from better_robot.io import ModelBuilder, build_model, load
from better_robot.optim import (
    Bounds,
    Euclidean,
    RobotConfig,
    SE3Manifold,
    SO3Manifold,
    VarSpec,
)


_BATCH_SHAPES = ((), (4,), (2, 3))


def _pattern(batch_shape: tuple[int, ...], width: int, *, scale: float) -> torch.Tensor:
    count = math.prod(batch_shape) * width
    return torch.linspace(-scale, scale, count, dtype=torch.float32).reshape(*batch_shape, width)


def _identity_so3(batch_shape: tuple[int, ...]) -> torch.Tensor:
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    return identity.expand(*batch_shape, 4).clone()


def _identity_se3(batch_shape: tuple[int, ...]) -> torch.Tensor:
    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return identity.expand(*batch_shape, 7).clone()


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
    builder.add_revolute_z(
        "hinge",
        parent=middle,
        child=tip,
        lower=-0.25,
        upper=0.25,
    )
    return build_model(builder.finalize(), dtype=torch.float32)


@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize("kind", ("euclidean", "so3", "se3"))
def test_retract_difference_roundtrip(kind: str, batch_shape: tuple[int, ...]) -> None:
    if kind == "euclidean":
        manifold = Euclidean()
        x = _pattern(batch_shape, 5, scale=0.3)
        dv = _pattern(batch_shape, 5, scale=0.02)
    elif kind == "so3":
        manifold = SO3Manifold()
        x = _identity_so3(batch_shape)
        dv = _pattern(batch_shape, 3, scale=0.02)
    else:
        manifold = SE3Manifold()
        x = _identity_se3(batch_shape)
        dv = _pattern(batch_shape, 6, scale=0.02)

    actual = manifold.difference(x, manifold.retract(x, dv))

    assert actual.shape == dv.shape
    torch.testing.assert_close(actual, dv, rtol=2e-4, atol=2e-5)


def _assert_robot_config_wrapper(model, batch_shape: tuple[int, ...]) -> None:
    manifold = RobotConfig(model)
    q = model.q_neutral.expand(*batch_shape, model.nq).clone()
    dv = _pattern(batch_shape, model.nv, scale=0.01)

    wrapped_q = manifold.retract(q, dv)
    direct_q = model.integrate(q, dv)
    torch.testing.assert_close(wrapped_q, direct_q, rtol=0.0, atol=0.0)

    wrapped_difference = manifold.difference(q, wrapped_q)
    direct_difference = model.difference(q, direct_q)
    torch.testing.assert_close(
        wrapped_difference,
        direct_difference,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(wrapped_difference, dv, rtol=2e-4, atol=2e-5)
    assert manifold.tangent_dim((model.nq,)) == model.nv


@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
def test_panda_robot_config_wrapper_parity(panda_model, batch_shape) -> None:
    assert panda_model.nq == panda_model.nv == 8
    _assert_robot_config_wrapper(panda_model, batch_shape)


@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
def test_floating_spherical_robot_config_wrapper_parity(
    floating_spherical_model,
    batch_shape,
) -> None:
    assert floating_spherical_model.nq == 12
    assert floating_spherical_model.nv == 10
    assert floating_spherical_model.nq != floating_spherical_model.nv
    _assert_robot_config_wrapper(floating_spherical_model, batch_shape)


def _floating_bounds(model) -> Bounds:
    lower = torch.full((model.nq,), -torch.inf)
    upper = torch.full((model.nq,), torch.inf)
    lower[0], upper[0] = -0.5, 0.5
    lower[-1], upper[-1] = -0.25, 0.25
    return Bounds(lower=lower, upper=upper)


def test_feasible_retraction_clamps_only_box_coordinates_and_preserves_units(
    floating_spherical_model,
) -> None:
    model = floating_spherical_model
    bounds = _floating_bounds(model)
    spec = VarSpec(
        name="q",
        shape=(model.nq,),
        manifold=RobotConfig(model),
        bounds=bounds,
    )
    q = model.q_neutral.clone()
    dv = torch.zeros(model.nv)
    dv[0] = 2.0
    dv[3:6] = torch.tensor([0.1, -0.2, 0.15])
    dv[6:9] = torch.tensor([0.2, 0.1, -0.1])
    dv[-1] = 1.0

    unconstrained = model.integrate(q, dv)
    projected = spec.retract(q, dv)

    assert projected[0].item() == pytest.approx(0.5)
    assert projected[-1].item() == pytest.approx(0.25)
    torch.testing.assert_close(projected[3:7], unconstrained[3:7])
    torch.testing.assert_close(projected[7:11], unconstrained[7:11])
    torch.testing.assert_close(projected[3:7].norm(), torch.tensor(1.0))
    torch.testing.assert_close(projected[7:11].norm(), torch.tensor(1.0))


def test_robot_config_prevalidated_projection_skips_bounds_revalidation(
    floating_spherical_model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = floating_spherical_model
    manifold = RobotConfig(model)
    bounds = _floating_bounds(model)
    q = model.q_neutral.clone()
    dv = torch.zeros(model.nv)
    dv[0] = 2.0
    dv[-1] = 1.0
    candidate = model.integrate(q, dv)
    expected = manifold.project(candidate, bounds)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("private RobotConfig projection revalidated static bounds")

    monkeypatch.setattr(RobotConfig, "validate_bounds", forbidden)
    actual = manifold._project_prevalidated(candidate, bounds)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("manifold", "shape", "kind"),
    ((SO3Manifold(), (4,), "SO3"), (SE3Manifold(), (7,), "SE3")),
)
def test_group_bounds_raise_the_exact_global_box_error(manifold, shape, kind) -> None:
    bounds = Bounds(torch.full(shape, -torch.inf), torch.full(shape, torch.inf))

    with pytest.raises(ValueError) as exc_info:
        VarSpec(name="orientation", shape=shape, manifold=manifold, bounds=bounds)

    expected = (
        f"{kind} variable blocks have no meaningful global box bound — neither in "
        "state space nor in tangent space. Express rotation limits as residuals "
        "(rotation prior / swing-twist, roadmap M3), or use RobotConfig with joint "
        f"limits. Got bounds={bounds!r} on VarSpec 'orientation'."
    )
    assert str(exc_info.value) == expected


def test_robot_config_rejects_bounds_on_quaternion_coordinates_exactly(
    floating_spherical_model,
) -> None:
    bounds = _floating_bounds(floating_spherical_model)
    lower = bounds.lower.clone()
    lower[3] = -1.0
    invalid = Bounds(lower, bounds.upper)

    with pytest.raises(ValueError) as exc_info:
        VarSpec(
            name="q",
            shape=(floating_spherical_model.nq,),
            manifold=RobotConfig(floating_spherical_model),
            bounds=invalid,
        )

    assert str(exc_info.value) == (
        "RobotConfig VarSpec 'q' bounds must be (-inf, +inf) on non-box "
        "manifold coordinates [3, 4, 5, 6, 7, 8, 9, 10]; quaternion/unit-circle "
        "coordinates are never clamped"
    )


def test_infeasible_panda_neutral_is_rejected_without_clamping(panda_model) -> None:
    spec = VarSpec(
        name="q",
        shape=(panda_model.nq,),
        manifold=RobotConfig(panda_model),
        bounds=Bounds(panda_model.lower_pos_limit, panda_model.upper_pos_limit),
    )

    with pytest.raises(ValueError) as exc_info:
        spec.validate_value(panda_model.q_neutral)

    assert str(exc_info.value) == (
        "Initial value for VarSpec 'q' is outside its state bounds; initial values "
        "are validated, not silently clamped. Supply a feasible start (notably, "
        "Panda q_neutral violates joint-4 limits and must be projected explicitly "
        "by the caller)."
    )
    torch.testing.assert_close(
        panda_model.q_neutral,
        torch.zeros_like(panda_model.q_neutral),
    )


def test_public_retract_and_difference_reject_infeasible_base_state() -> None:
    spec = VarSpec(
        name="x",
        shape=(1,),
        bounds=Bounds(torch.tensor([-1.0]), torch.tensor([1.0])),
    )
    infeasible = torch.tensor([2.0])

    with pytest.raises(ValueError, match="outside its state bounds"):
        spec.retract(infeasible, torch.zeros(1))
    with pytest.raises(ValueError, match="outside its state bounds"):
        spec.difference(infeasible, torch.zeros(1))


def test_mask_eliminates_fixed_tangent_coordinates() -> None:
    spec = VarSpec(
        name="x",
        shape=(4,),
        manifold=Euclidean(),
        mask=torch.tensor([True, False, True, False]),
    )

    assert spec.tangent_dim == 4
    assert spec.free_dim == 2
    assert spec.free_indices.tolist() == [0, 2]
    reduced = torch.tensor([0.25, -0.5])
    expanded = spec.expand_tangent(reduced)
    torch.testing.assert_close(expanded, torch.tensor([0.25, 0.0, -0.5, 0.0]))
    torch.testing.assert_close(spec.gather_tangent(expanded), reduced)
    torch.testing.assert_close(spec.retract(torch.zeros(4), reduced), expanded)


def test_scalar_event_shape_supports_independent_batch_axes() -> None:
    spec = VarSpec(name="scale", shape=())
    torch.testing.assert_close(
        spec.retract(torch.tensor(1.0), torch.tensor([0.2])),
        torch.tensor(1.2),
    )
    values = torch.tensor([0.2, -0.3, 0.5])
    steps = torch.tensor([[0.1], [0.2], [-0.4]])

    assert spec.batch_shape(values) == (3,)
    assert spec.tangent_dim == spec.free_dim == 1
    torch.testing.assert_close(spec.retract(values, steps), torch.tensor([0.3, -0.1, 0.1]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.int64))
def test_values_reject_unsupported_working_dtypes(dtype: torch.dtype) -> None:
    with pytest.raises(DtypeMismatchError, match="use torch.float32 or torch.float64"):
        VarSpec(name="x", shape=(2,)).validate_value(torch.ones(2, dtype=dtype))


def test_retraction_rejects_a_mixed_dtype_step() -> None:
    spec = VarSpec(name="x", shape=(2,))

    with pytest.raises(DtypeMismatchError, match="Step and value.*share dtype"):
        spec.retract(
            torch.ones(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float64),
        )


def test_difference_validates_both_inputs_and_rejects_mixed_dtype() -> None:
    spec = VarSpec(name="x", shape=(2,))

    with pytest.raises(DtypeMismatchError, match="Difference inputs.*share dtype"):
        spec.difference(
            torch.ones(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="only finite entries"):
        spec.difference(torch.ones(2), torch.tensor([0.0, float("nan")]))


def test_bounds_reject_nan_endpoints() -> None:
    with pytest.raises(ValueError, match="must not contain NaN"):
        Bounds(torch.tensor([float("nan")]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="must not contain NaN"):
        Bounds(torch.tensor([-1.0]), torch.tensor([float("nan")]))


def test_initial_group_values_must_belong_to_their_manifold() -> None:
    with pytest.raises(QuaternionNormError, match="not on its configuration manifold"):
        VarSpec("rotation", (4,), manifold=SO3Manifold()).validate_value(torch.zeros(4))
    with pytest.raises(QuaternionNormError, match="not on its configuration manifold"):
        VarSpec("pose", (7,), manifold=SE3Manifold()).validate_value(torch.zeros(7))


def test_initial_robot_configuration_rejects_invalid_unit_coordinates(
    floating_spherical_model,
) -> None:
    q = floating_spherical_model.q_neutral.clone()
    q[3:7] = 0.0

    with pytest.raises(QuaternionNormError, match="unit quaternion/unit-circle"):
        VarSpec(
            "q",
            (floating_spherical_model.nq,),
            manifold=RobotConfig(floating_spherical_model),
        ).validate_value(q)
