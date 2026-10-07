"""Public passes trust attached model values and validate call inputs once."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest
import torch

from better_robot.data_model.model_values import ModelValues
from better_robot.dynamics import (
    centroidal_map,
    centroidal_momentum,
    forward_dynamics,
    inverse_dynamics,
    mass_matrix,
)
from better_robot.dynamics.state import center_of_mass
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics_state


def _model():
    builder = ModelBuilder("validation_count")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=1.0, inertia=torch.eye(3) * 0.2)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_frame(
        "tip",
        parent_body="link",
        placement=torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    return build_model(builder.finalize(), dtype=torch.float64)


def test_public_passes_trust_attached_values_and_validate_inputs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    q = torch.tensor([0.2], dtype=torch.float64)
    v = torch.tensor([0.1], dtype=torch.float64)
    a = torch.tensor([-0.05], dtype=torch.float64)
    tau = torch.tensor([0.3], dtype=torch.float64)
    state = forward_kinematics_state(model, q)

    value_validation_count = 0
    input_validation_count = 0
    original_validate = ModelValues.validate
    state_module = importlib.import_module("better_robot.kinematics.state")
    original_validate_q = state_module._validate_q

    def counted_validate(values: ModelValues, structure) -> None:
        nonlocal value_validation_count
        value_validation_count += 1
        original_validate(values, structure)

    def counted_validate_q(structure, values, q) -> None:
        nonlocal input_validation_count
        input_validation_count += 1
        original_validate_q(structure, values, q)

    monkeypatch.setattr(ModelValues, "validate", counted_validate)
    monkeypatch.setattr(state_module, "_validate_q", counted_validate_q)

    # q is validated once, at the forward-kinematics boundary; dynamics on a state trusts it.
    public_calls: dict[str, tuple[Callable[[], object], int]] = {
        "forward_kinematics_state": (lambda: forward_kinematics_state(model, q), 1),
        "inverse_dynamics": (lambda: inverse_dynamics(model, state, v=v, a=a), 0),
        "forward_dynamics": (lambda: forward_dynamics(model, state, v=v, tau=tau), 0),
        "mass_matrix": (lambda: mass_matrix(model, state), 0),
        "center_of_mass": (lambda: center_of_mass(model, state), 0),
        "centroidal_map": (lambda: centroidal_map(model, state), 0),
        "centroidal_momentum": (lambda: centroidal_momentum(model, state, v=v), 0),
        "inverse_dynamics(forward_kinematics_state)": (
            lambda: inverse_dynamics(model, forward_kinematics_state(model, q), v=v, a=a),
            1,
        ),
    }

    for name, (public_call, expected_input_checks) in public_calls.items():
        inputs_before = input_validation_count
        public_call()
        assert value_validation_count == 0, name
        assert input_validation_count == inputs_before + expected_input_checks, name
