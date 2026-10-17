"""Robot terms: analytic Jacobians equal autodiff; IK reaches reachable targets inside the limits."""

from dataclasses import replace

import pytest
import torch

import better_robot as br
from better_robot import optim, terms
from better_robot.kinematics import forward_kinematics_state
from better_robot.lie import so3

FRAME = "body_panda_hand"


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _random_q(model, batch, seed):
    return terms.random_start(model, model.q_neutral.expand(batch, -1), generator=torch.Generator().manual_seed(seed))


def test_analytic_jacobians_match_autodiff(panda):
    q = _random_q(panda, 6, 0)
    goal = forward_kinematics_state(panda, _random_q(panda, 6, 1)).frame_pose[:, panda.frame_id(FRAME)]
    problem = terms.ik_problem(panda, {FRAME: goal}, q0=q + 4.0, limit_weight=2.0, rest_weight=0.5)
    problem.data["point"] = goal[:, :3]
    problem.residual("position", terms.frame_position(panda, frame=FRAME, target="point"), kernel=optim.Huber(delta=0.1))
    errors = optim.check_jacobian(problem, {"q": q + 4.0})
    assert set(errors) == {f"pose/{FRAME}", "limits", "rest", "position"}
    # 1e-5: URDF values are float32-rounded, which limits analytic-vs-autodiff agreement of H.
    assert max(errors.values()) < 1e-5, errors


def test_ik_with_restarts_reaches_every_target_inside_limits(panda):
    goal = forward_kinematics_state(panda, _random_q(panda, 32, 2)).frame_pose[:, panda.frame_id(FRAME)]
    q0 = _random_q(panda, 32, 3)
    result = terms.solve_ik(panda, {FRAME: goal}, q0=q0, restarts=16, generator=torch.Generator().manual_seed(4))
    q = result.x["q"]
    reached = forward_kinematics_state(panda, q).frame_pose[:, panda.frame_id(FRAME)]
    assert float((reached[:, :3] - goal[:, :3]).norm(dim=-1).max()) < 1e-6
    assert bool(((q >= panda.lower_pos_limit) & (q <= panda.upper_pos_limit)).all())


def test_result_getitem_and_ik_problem_state_handle(panda):
    goal = forward_kinematics_state(panda, _random_q(panda, 4, 2)).frame_pose[:, panda.frame_id(FRAME)]
    result = terms.solve_ik(panda, {FRAME: goal}, q0=_random_q(panda, 4, 3))
    assert result["q"] is result.x["q"]
    problem = terms.ik_problem(panda, {FRAME: goal}, q0=_random_q(panda, 4, 3))
    assert problem.terms[f"pose/{FRAME}"].reads[0] == "state"


def test_robot_manifold_accepts_an_empty_batch(panda):
    # tangent_dim used to index x[0], which raises IndexError for B == 0.
    problem = optim.Problem({"q": torch.zeros(0, panda.nq, dtype=torch.float64)}, manifolds={"q": terms.RobotManifold(panda)})
    problem.residual("rest", terms.rest(panda, torch.zeros(panda.nq, dtype=torch.float64)))
    result = optim.solve(problem, optim.LevenbergMarquardt(), iters=2)
    assert result.x["q"].shape == (0, panda.nq) and result.loss.shape == (0,)


@pytest.fixture(scope="module")
def g1():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415

    return br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)


def test_finite_difference_any_order_in_world_translation_and_velocity_limit(g1):
    generator = torch.Generator().manual_seed(0)
    q = g1.integrate(g1.q_neutral.expand(1, 8, -1), 0.3 * torch.randn(1, 8, g1.nv, generator=generator, dtype=torch.float64))
    dt = 1 / 30
    # World-frame root step, body-frame root rotation step, joint steps; jerk = second difference of the steps.
    rotation = so3.log(so3.compose(so3.inverse(q[:, :-1, 3:7]), q[:, 1:, 3:7]))
    step = torch.cat([q[:, 1:, :3] - q[:, :-1, :3], rotation, q[:, 1:, 7:] - q[:, :-1, 7:]], dim=-1)
    jerk = step[:, 2:] - 2 * step[:, 1:-1] + step[:, :-2]
    term = terms.finite_difference(g1, order=3, dt=dt, world_translation=True)
    torch.testing.assert_close(term.fn(q), jerk / dt**3, rtol=1e-9, atol=1e-6)
    limit = terms.velocity_limit(g1, dt=dt, fraction=0.1)
    speed = terms.finite_difference(g1, order=1, dt=dt).fn(q)
    excess = (speed.abs() - 0.1 * g1.velocity_limit).clamp_min(0.0)
    torch.testing.assert_close(limit.fn(q), speed.sign() * excess)
    assert bool((excess[..., 6:] > 0).any()) and bool((limit.fn(q)[..., :6] == 0).all())  # root: no limit
    unset = replace(g1, values=replace(g1.values, velocity_limit=torch.zeros_like(g1.velocity_limit)))  # MJCF stores 0
    assert bool((terms.velocity_limit(unset, dt=dt).fn(q) == 0).all())  # a stored 0 is no limit, not a zero one
    problem = optim.Problem({"q": q}, manifolds={"q": terms.RobotManifold(g1)}, time=("q",))
    problem.residual("jerk", term)
    problem.residual("velocity_limit", limit)
    assert max(optim.check_jacobian(problem).values()) < 1e-9
