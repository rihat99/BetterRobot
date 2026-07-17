"""Public FK and dynamics calls validate model values exactly once."""

from __future__ import annotations

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


def test_each_public_fk_and_dynamics_call_validates_values_once(
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

    validation_count = 0
    original_validate = ModelValues.validate

    def counted_validate(values: ModelValues, structure) -> None:
        nonlocal validation_count
        validation_count += 1
        original_validate(values, structure)

    monkeypatch.setattr(ModelValues, "validate", counted_validate)

    public_calls: dict[str, Callable[[], object]] = {
        "forward_kinematics(tensor)": lambda: forward_kinematics(model, q),
        "forward_kinematics(data)": lambda: forward_kinematics(model, data_input),
        "forward_kinematics(frames)": lambda: forward_kinematics(model, q, compute_frames=True),
        "update_frame_placements": lambda: update_frame_placements(model, placed_data),
        "rnea": lambda: rnea(model, model.create_data(), q, v, a),
        "bias_forces": lambda: bias_forces(model, model.create_data(), q, v),
        "compute_generalized_gravity": lambda: compute_generalized_gravity(model, model.create_data(), q),
        "aba": lambda: aba(model, model.create_data(), q, v, tau),
        "crba": lambda: crba(model, model.create_data(), q),
        "center_of_mass": lambda: center_of_mass(model, model.create_data(), q, v),
        "compute_centroidal_map": lambda: compute_centroidal_map(model, model.create_data(), q),
        "compute_centroidal_momentum": lambda: compute_centroidal_momentum(model, model.create_data(), q, v),
        "ccrba": lambda: ccrba(model, model.create_data(), q, v),
    }

    for name, public_call in public_calls.items():
        count_before = validation_count
        public_call()
        assert validation_count == count_before + 1, name
