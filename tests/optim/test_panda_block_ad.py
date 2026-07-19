"""Real Panda pose-Jacobian parity through object-referenced RobotState."""

from __future__ import annotations

import pytest
import torch

from better_robot.io import ModelBuilder, build_model, load
from better_robot.kinematics import forward_kinematics
from better_robot.lie import se3
from better_robot.optim import Problem, Residual, RobotVariable
from better_robot.residuals.nodes import RobotState
from better_robot.residuals.pose import PoseResidual


class _PlanarOrientationResidual(Residual):
    def __init__(self, state: RobotState, joint_id: int) -> None:
        self.state = state
        self.nodes = (state,)
        self.joint_id = joint_id
        super().__init__(state.q, dim=2, name="planar_orientation")

    def error(self) -> torch.Tensor:
        return self.state.value().joint_pose_world[..., self.joint_id, 5:7]


@pytest.fixture(scope="module")
def panda_pose_case():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    model = load(panda_description.URDF_PATH, dtype=torch.float32)

    q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit).clone()
    offset_q = torch.zeros_like(q)
    offsets_by_joint = {
        "panda_joint1": 0.10,
        "panda_joint2": -0.12,
        "panda_joint3": 0.08,
        "panda_joint4": -0.05,
        "panda_joint5": 0.07,
        "panda_joint6": -0.09,
        "panda_joint7": 0.06,
        "panda_finger_joint1": 0.005,
    }
    for joint_name, offset in offsets_by_joint.items():
        joint_id = model.joint_id(joint_name)
        assert model.nqs[joint_id] == 1
        offset_q[model.idx_qs[joint_id]] = offset
    q = (q + offset_q).clamp(model.lower_pos_limit, model.upper_pos_limit)

    frame_id = model.frame_id("body_panda_hand")
    data = forward_kinematics(model, q, compute_frames=True)
    target_offset = torch.tensor([0.02, -0.01, 0.015, 0.03, -0.02, 0.01])
    target = se3.compose(data.frame_pose_world[frame_id], se3.exp(target_offset)).detach()
    return model, q, frame_id, target


def _problem(
    model,
    q_tensor: torch.Tensor,
    frame_id: int,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[RobotVariable, Problem]:
    q = RobotVariable(model, q_tensor.clone(), name="q", mask=mask)
    residual = PoseResidual(
        q,
        frame_id=frame_id,
        target=target,
        pos_weight=0.8,
        ori_weight=1.2,
        name="panda_pose",
    )
    return q, Problem([residual])


def _planar_problem() -> tuple[Problem, RobotVariable]:
    builder = ModelBuilder("planar_ad")
    base = builder.add_body("base", mass=1.0)
    link = builder.add_body("link", mass=1.0)
    builder.add_planar("planar", parent=base, child=link)
    model = build_model(builder.finalize(), dtype=torch.float32)
    q = RobotVariable(model, model.q_neutral, name="q")
    residual = _PlanarOrientationResidual(RobotState(q), model.joint_id("planar"))
    return Problem([residual]), q


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
def test_panda_pose_forced_ad_matches_analytic(panda_pose_case, strategy: str) -> None:
    model, q_tensor, frame_id, target = panda_pose_case
    _q, problem = _problem(model, q_tensor, frame_id, target)

    analytic = problem.jacobian_blocks(strategy="analytic")[("panda_pose", "q")]
    forced = problem.jacobian_blocks(strategy=strategy)[("panda_pose", "q")]

    assert analytic.shape == forced.shape == (6, model.nv)
    assert forced.dtype == q_tensor.dtype == torch.float32
    assert torch.isfinite(forced).all()
    torch.testing.assert_close(forced, analytic, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
def test_panda_pose_masked_ad_uses_reduced_columns(panda_pose_case, strategy: str) -> None:
    model, q_tensor, frame_id, target = panda_pose_case
    assert model.nv == 8
    mask = torch.zeros(model.nv, dtype=torch.bool)
    for joint_name in (
        "panda_joint1",
        "panda_joint3",
        "panda_joint4",
        "panda_joint6",
        "panda_joint7",
    ):
        joint_id = model.joint_id(joint_name)
        assert model.nvs[joint_id] == 1
        mask[model.idx_vs[joint_id]] = True
    _full_q, full_problem = _problem(model, q_tensor, frame_id, target)
    _masked_q, masked_problem = _problem(model, q_tensor, frame_id, target, mask=mask)

    full = full_problem.jacobian_blocks(strategy="analytic")[("panda_pose", "q")]
    analytic = masked_problem.jacobian_blocks(strategy="analytic")[("panda_pose", "q")]
    forced = masked_problem.jacobian_blocks(strategy=strategy)[("panda_pose", "q")]

    expected = full.index_select(-1, torch.nonzero(mask, as_tuple=False).flatten())
    assert analytic.shape == forced.shape == (6, int(mask.sum()))
    assert forced.dtype == q_tensor.dtype == torch.float32
    torch.testing.assert_close(analytic, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(forced, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
def test_planar_robot_config_fk_is_ad_clean_at_neutral(strategy: str) -> None:
    problem, q = _planar_problem()

    block = problem.jacobian_blocks(strategy=strategy)[("planar_orientation", "q")]
    gradient = problem.gradient()["q"]

    expected = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    assert block.dtype == q.tensor.dtype == torch.float32
    assert torch.isfinite(block).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(block, expected, atol=2e-6, rtol=2e-6)
