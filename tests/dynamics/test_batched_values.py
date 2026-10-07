"""Value-batched rigid-body dynamics and inertia-autograd contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.dynamics.state import (
    centroidal_map,
    centroidal_momentum,
    forward_dynamics,
    inverse_dynamics,
    mass_matrix,
)
from better_robot.exceptions import ShapeError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics_state


def _pose(x: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model(dtype: torch.dtype = torch.float64):
    builder = ModelBuilder("batched_values_dynamics")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.15)
    builder.add_body(
        "link",
        mass=2.0,
        com=torch.tensor([0.2, 0.0, 0.0]),
        inertia=torch.diag(torch.tensor([0.2, 0.3, 0.4])),
    )
    builder.add_revolute_y(
        "joint",
        parent="base",
        child="link",
        origin=_pose(0.5),
        lower=-2.0,
        upper=2.0,
    )
    return build_model(builder.finalize(), dtype=dtype)


def _inertia_batch(model, batch_shape: tuple[int, ...]) -> torch.Tensor:
    inertias = model.values.body_inertias.expand(
        *batch_shape,
        model.nbodies,
        10,
    ).clone()
    count = 1
    for size in batch_shape:
        count *= size
    scales = torch.linspace(0.8, 1.3, count, dtype=inertias.dtype).reshape(*batch_shape)
    inertias[..., 2, 0] *= scales
    inertias[..., 2, 4:10] *= scales[..., None]
    return inertias


def test_centroidal_map_and_momentum_shapes() -> None:
    model = _model()
    state = forward_kinematics_state(model, model.q_neutral)
    velocity = torch.zeros(model.nv, dtype=model.q_neutral.dtype)

    assert centroidal_map(model, state).shape == (6, model.nv)
    assert centroidal_momentum(model, state, v=velocity).shape == (6,)


@pytest.mark.parametrize("batch_shape", ((3,), (2, 2)))
def test_batched_inertias_match_scalar_dynamics_loop(
    value_batch_loop_oracle,
    batch_shape: tuple[int, ...],
) -> None:
    model = _model()
    inertias = _inertia_batch(model, batch_shape)
    rebound = model.with_values(body_inertias=inertias)
    q = torch.tensor([0.3], dtype=inertias.dtype)
    v = torch.tensor([0.2], dtype=inertias.dtype)
    a = torch.tensor([-0.4], dtype=inertias.dtype)

    def evaluate(current_model):
        state = forward_kinematics_state(current_model, q)
        tau = inverse_dynamics(current_model, state, v, a)
        mass = mass_matrix(current_model, state)
        ddq = forward_dynamics(current_model, state, v, tau)
        return (
            tau,
            mass,
            ddq,
            centroidal_map(current_model, state),
            centroidal_momentum(current_model, state, v=v),
        )

    value_batch_loop_oracle(
        lambda: evaluate(rebound),
        lambda index: evaluate(model.with_values(body_inertias=inertias[index])),
        execution_batch_shape=batch_shape,
        rtol=2e-10,
        atol=2e-11,
    )


def test_batched_inertia_autograd_reaches_leaf() -> None:
    model = _model()
    delta = torch.zeros(
        2,
        model.nbodies,
        10,
        dtype=model.values.body_inertias.dtype,
        requires_grad=True,
    )
    inertias = model.values.body_inertias + delta
    rebound = model.with_values(body_inertias=inertias)
    q = torch.tensor([0.35], dtype=inertias.dtype)
    v = torch.tensor([0.1], dtype=inertias.dtype)
    a = torch.tensor([0.2], dtype=inertias.dtype)
    tau = inverse_dynamics(rebound, forward_kinematics_state(rebound, q), v, a)
    tau.square().sum().backward()

    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_dynamics_value_batch_mismatch_is_an_honest_shape_error() -> None:
    model = _model()
    inertias = _inertia_batch(model, (2,))
    rebound = model.with_values(body_inertias=inertias)
    q = model.q_neutral.expand(2, 3, -1)
    v = torch.zeros(2, 3, model.nv, dtype=q.dtype)
    with pytest.raises(
        ShapeError,
        match=r"cannot broadcast q batch \(2, 3\) with body_inertias batch \(2,\)",
    ):
        inverse_dynamics(rebound, forward_kinematics_state(rebound, q), v, v)
