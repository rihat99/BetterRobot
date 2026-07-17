"""Truthful validation of named-block IK solver configuration."""

from __future__ import annotations

import pytest
import torch

from better_robot.data_model.model import Model
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import JacobianStrategy, forward_kinematics
from better_robot.tasks.ik import OptimizerConfig, solve_ik


@pytest.fixture(scope="module")
def ik_case() -> tuple[Model, dict[str, torch.Tensor]]:
    builder = ModelBuilder("ik_config_truth")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        origin=torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    model = build_model(builder.finalize())
    frame_name = "body_tip"
    target = forward_kinematics(
        model,
        model.q_neutral,
        compute_frames=True,
    ).frame_pose_world[model.frame_id(frame_name)]
    return model, {frame_name: target}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (OptimizerConfig(optimizer="newton"), "Unknown optimizer"),
        (OptimizerConfig(linear_solver="qr"), "Unknown linear_solver"),
        (OptimizerConfig(damping="trust_region"), "Unknown damping"),
        (OptimizerConfig(kernel="geman_mcclure"), "Unknown kernel"),
        (OptimizerConfig(jacobian_strategy="reverse_mode"), "Unknown jacobian_strategy"),
    ],
)
def test_invalid_solver_knobs_fail_at_facade_boundary(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
    config: OptimizerConfig,
    message: str,
) -> None:
    model, targets = ik_case
    with pytest.raises(ValueError, match=message):
        solve_ik(model, targets, optimizer_cfg=config)


def test_jacobian_strategy_requires_the_public_enum(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
) -> None:
    model, targets = ik_case
    with pytest.raises(ValueError, match="JacobianStrategy member"):
        solve_ik(
            model,
            targets,
            optimizer_cfg=OptimizerConfig(jacobian_strategy="auto"),
        )


@pytest.mark.parametrize(
    ("config", "field"),
    [
        (
            OptimizerConfig(optimizer="adam", linear_solver="lstsq"),
            "linear_solver",
        ),
        (
            OptimizerConfig(
                optimizer="adam",
                jacobian_strategy=JacobianStrategy.ANALYTIC,
            ),
            "jacobian_strategy",
        ),
        (OptimizerConfig(optimizer="adam", damping="constant"), "damping"),
    ],
)
def test_adam_rejects_unused_normal_equation_knobs(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
    config: OptimizerConfig,
    field: str,
) -> None:
    model, targets = ik_case
    with pytest.raises(ValueError, match=rf"optimizer='adam'.*{field}"):
        solve_ik(model, targets, optimizer_cfg=config)


def test_adam_accepts_its_default_facade_configuration(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
) -> None:
    model, targets = ik_case
    result = solve_ik(
        model,
        targets,
        optimizer_cfg=OptimizerConfig(optimizer="adam", max_iter=0),
    )
    assert result.q.shape == model.q_neutral.shape


def test_gauss_newton_rejects_unused_damping_selector(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
) -> None:
    model, targets = ik_case
    with pytest.raises(ValueError, match=r"optimizer='gn'.*damping"):
        solve_ik(
            model,
            targets,
            optimizer_cfg=OptimizerConfig(optimizer="gn", damping="constant"),
        )


@pytest.mark.parametrize("optimizer", ["lm", "gn", "adam"])
def test_single_stage_optimizers_reject_refinement_only_items(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
    optimizer: str,
) -> None:
    model, targets = ik_case
    config = OptimizerConfig(
        optimizer=optimizer,
        refine_disabled_items=("rest",),
    )
    with pytest.raises(ValueError, match="refine_disabled_items.*lm_then_adam"):
        solve_ik(model, targets, optimizer_cfg=config)


def test_phased_optimizer_accepts_refinement_disabled_items(
    ik_case: tuple[Model, dict[str, torch.Tensor]],
) -> None:
    model, targets = ik_case
    result = solve_ik(
        model,
        targets,
        optimizer_cfg=OptimizerConfig(
            optimizer="lm_then_adam",
            max_iter=0,
            refine_disabled_items=("rest",),
        ),
    )
    assert result.q.shape == model.q_neutral.shape
