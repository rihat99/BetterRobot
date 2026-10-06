"""Robot terms: analytic Jacobians equal autodiff; IK reaches reachable targets inside the limits."""

import pytest
import torch

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state

FRAME = "body_panda_hand"


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _random_q(model, batch, seed):
    return terms.random_start(model, model.q_neutral.expand(batch, -1), torch.Generator().manual_seed(seed))


def test_analytic_jacobians_match_autodiff(panda):
    q = _random_q(panda, 6, 0)
    goal = forward_kinematics_state(panda, _random_q(panda, 6, 1)).frame_pose[:, panda.frame_id(FRAME)]
    problem = terms.ik_problem(panda, {FRAME: goal}, q + 4.0, limit_weight=2.0, rest_weight=0.5)
    problem.data["point"] = goal[:, :3]
    problem.residual("position", terms.frame_position(panda, FRAME, target="point"), kernel=optim2.Huber(delta=0.1))
    errors = optim2.check_jacobian(problem, {"q": q + 4.0})
    assert set(errors) == {f"pose/{FRAME}", "limits", "rest", "position"}
    # 1e-5: URDF values are float32-rounded, which limits analytic-vs-autodiff agreement of H.
    assert max(errors.values()) < 1e-5, errors


def test_ik_with_restarts_reaches_every_target_inside_limits(panda):
    goal = forward_kinematics_state(panda, _random_q(panda, 32, 2)).frame_pose[:, panda.frame_id(FRAME)]
    q0 = _random_q(panda, 32, 3)
    result = terms.solve_ik(panda, {FRAME: goal}, q0, restarts=16, generator=torch.Generator().manual_seed(4))
    q = result.x["q"]
    reached = forward_kinematics_state(panda, q).frame_pose[:, panda.frame_id(FRAME)]
    assert float((reached[:, :3] - goal[:, :3]).norm(dim=-1).max()) < 1e-6
    assert bool(((q >= panda.lower_pos_limit) & (q <= panda.upper_pos_limit)).all())
