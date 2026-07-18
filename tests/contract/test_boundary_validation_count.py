"""Public passes trust attached model values and validate call inputs once."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest
import torch

from better_robot.data_model.model_values import ModelValues
from better_robot.dynamics import (
    aba,
    bias_forces,
    ccrba,
    center_of_mass,
    compute_centroidal_map,
    compute_centroidal_momentum,
    compute_generalized_gravity,
    crba,
    rnea,
)
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics, update_frame_placements


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
    placed_data = forward_kinematics(model, q)
    data_input = model.create_data()
    data_input.q = q

    value_validation_count = 0
    input_validation_count = 0
    original_validate = ModelValues.validate
    forward_module = importlib.import_module("better_robot.kinematics.forward")
    execution_module = importlib.import_module("better_robot.dynamics._execution")
    original_validate_q = forward_module._validate_q

    def counted_validate(values: ModelValues, structure) -> None:
        nonlocal value_validation_count
        value_validation_count += 1
        original_validate(values, structure)

    def counted_validate_q(structure, values, q) -> None:
        nonlocal input_validation_count
        input_validation_count += 1
        original_validate_q(structure, values, q)

    monkeypatch.setattr(ModelValues, "validate", counted_validate)
    monkeypatch.setattr(forward_module, "_validate_q", counted_validate_q)
    monkeypatch.setattr(execution_module, "_validate_q", counted_validate_q)

    public_calls: dict[str, tuple[Callable[[], object], int]] = {
        "forward_kinematics(tensor)": (lambda: forward_kinematics(model, q), 1),
        "forward_kinematics(data)": (lambda: forward_kinematics(model, data_input), 1),
        "forward_kinematics(frames)": (lambda: forward_kinematics(model, q, compute_frames=True), 1),
        "update_frame_placements": (lambda: update_frame_placements(model, placed_data), 0),
        "rnea": (lambda: rnea(model, q, v, a), 1),
        "bias_forces": (lambda: bias_forces(model, q, v), 1),
        "compute_generalized_gravity": (lambda: compute_generalized_gravity(model, q), 1),
        "aba": (lambda: aba(model, q, v, tau), 1),
        "crba": (lambda: crba(model, q), 1),
        "center_of_mass": (lambda: center_of_mass(model, q, v), 1),
        "compute_centroidal_map": (lambda: compute_centroidal_map(model, q), 1),
        "compute_centroidal_momentum": (lambda: compute_centroidal_momentum(model, q, v), 1),
        "ccrba": (lambda: ccrba(model, q, v), 1),
    }

    for name, (public_call, expected_input_checks) in public_calls.items():
        inputs_before = input_validation_count
        public_call()
        assert value_validation_count == 0, name
        assert input_validation_count == inputs_before + expected_input_checks, name
