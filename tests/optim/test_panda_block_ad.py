"""Real Panda pose-Jacobian parity through the named-block provider seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from better_robot.io import ModelBuilder, build_model, load
from better_robot.kinematics import forward_kinematics
from better_robot.lie import se3
from better_robot.optim import Problem, ResidualItem, RobotConfig, RobotStateProvider, VarSpec
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual


class _PandaBlockPose:
    """Adapt the existing stateful pose residual to the block context contract."""

    name = "panda_pose"
    reads = ("q", "data")
    dim = 6

    def __init__(self, model, residual: PoseResidual) -> None:
        self.model = model
        self.residual = residual

    def _state(self, ctx: Mapping[str, Any]) -> ResidualState:
        return ResidualState(
            model=self.model,
            data=ctx["data"],
            variables=ctx["q"],
        )

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return self.residual(self._state(ctx))

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        full = self.residual.jacobian(self._state(ctx))
        assert full is not None
        free = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, free)}


class _PlanarOrientationResidual:
    name = "planar_orientation"
    reads = ("data",)
    dim = 2

    def __init__(self, joint_id: int) -> None:
        self.joint_id = joint_id

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        poses = ctx["data"].joint_pose_world
        assert poses is not None
        return poses[..., self.joint_id, 5:7]


@pytest.fixture(scope="module")
def panda_pose_case():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    model = load(panda_description.URDF_PATH, dtype=torch.float32)

    q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit).clone()
    offset_q = torch.tensor([0.10, -0.12, 0.08, -0.05, 0.07, -0.09, 0.06, 0.005, 0.004])
    assert offset_q.shape == q.shape
    q = (q + offset_q).clamp(model.lower_pos_limit, model.upper_pos_limit)

    frame_id = model.frame_id("body_panda_hand")
    data = forward_kinematics(model, q, compute_frames=True)
    target_offset = torch.tensor([0.02, -0.01, 0.015, 0.03, -0.02, 0.01])
    target = se3.compose(
        data.frame_pose_world[frame_id],
        se3.exp(target_offset),
    ).detach()
    legacy = PoseResidual(
        frame_id=frame_id,
        target=target,
        pos_weight=0.8,
        ori_weight=1.2,
    )
    return model, q, _PandaBlockPose(model, legacy)


def _problem(model, residual: _PandaBlockPose, *, mask: torch.Tensor | None = None) -> Problem:
    return Problem(
        vars=(
            VarSpec(
                "q",
                (model.nq,),
                manifold=RobotConfig(model),
                mask=mask,
            ),
        ),
        residuals=(ResidualItem("panda_pose", residual),),
        providers=(RobotStateProvider(model, var="q", output="data"),),
    )


def _planar_problem() -> tuple[Problem, torch.Tensor]:
    builder = ModelBuilder("planar_ad")
    base = builder.add_body("base", mass=1.0)
    link = builder.add_body("link", mass=1.0)
    builder.add_planar("planar", parent=base, child=link)
    model = build_model(builder.finalize(), dtype=torch.float32)
    residual = _PlanarOrientationResidual(model.joint_id("planar"))
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), manifold=RobotConfig(model)),),
        residuals=(ResidualItem("planar_orientation", residual),),
        providers=(RobotStateProvider(model),),
    )
    return problem, model.q_neutral


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
def test_panda_pose_forced_ad_matches_analytic(panda_pose_case, strategy: str) -> None:
    model, q, residual = panda_pose_case
    problem = _problem(model, residual)

    analytic = problem.jacobian_blocks({"q": q}, strategy="analytic")[("panda_pose", "q")]
    forced = problem.jacobian_blocks({"q": q}, strategy=strategy)[("panda_pose", "q")]

    assert analytic.shape == forced.shape == (6, model.nv)
    assert forced.dtype == q.dtype == torch.float32
    assert torch.isfinite(forced).all()
    torch.testing.assert_close(forced, analytic, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
def test_panda_pose_masked_ad_uses_reduced_columns(panda_pose_case, strategy: str) -> None:
    model, q, residual = panda_pose_case
    assert model.nv == 9
    mask = torch.tensor([True, False, True, True, False, True, True, False, True])
    full_problem = _problem(model, residual)
    masked_problem = _problem(model, residual, mask=mask)

    full = full_problem.jacobian_blocks({"q": q}, strategy="analytic")[("panda_pose", "q")]
    analytic = masked_problem.jacobian_blocks({"q": q}, strategy="analytic")[("panda_pose", "q")]
    forced = masked_problem.jacobian_blocks({"q": q}, strategy=strategy)[("panda_pose", "q")]

    expected = full.index_select(-1, torch.nonzero(mask, as_tuple=False).flatten())
    assert analytic.shape == forced.shape == (6, int(mask.sum()))
    assert forced.dtype == q.dtype == torch.float32
    torch.testing.assert_close(analytic, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(forced, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
def test_planar_robot_config_fk_is_ad_clean_at_neutral(strategy: str) -> None:
    problem, q = _planar_problem()

    block = problem.jacobian_blocks({"q": q}, strategy=strategy)[("planar_orientation", "q")]
    gradient = problem.gradient({"q": q})["q"]

    expected = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    assert block.dtype == q.dtype == torch.float32
    assert torch.isfinite(block).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(block, expected, atol=2e-6, rtol=2e-6)
