"""Tests for solvers."""

import functools

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.costs import CostTerm, pose_residual, limit_residual, rest_residual
from better_robot.solvers import LevenbergMarquardt, PyposeLevenbergMarquardt, Problem
from better_robot.algorithms.kinematics import compute_jacobian, limit_jacobian, rest_jacobian


def test_lm_solves_simple_quadratic():
    """LM should minimize (x - 2)^2 + (y - 3)^2 to find [2, 3]."""
    target = torch.tensor([2.0, 3.0])

    def residual(x):
        return x - target

    problem = Problem(
        variables=torch.zeros(2),
        costs=[CostTerm(residual_fn=residual, weight=1.0, kind="soft")]
    )

    solver = LevenbergMarquardt()
    solution = solver.solve(problem, max_iter=50)

    assert solution.shape == (2,)
    assert torch.allclose(solution, target, atol=1e-4), f"Expected {target}, got {solution}"


def test_lm_returns_correct_shape():
    def residual(x):
        return x ** 2

    problem = Problem(
        variables=torch.ones(5),
        costs=[CostTerm(residual_fn=residual, weight=1.0)]
    )
    solver = LevenbergMarquardt()
    sol = solver.solve(problem, max_iter=20)
    assert sol.shape == (5,)


def test_problem_total_residual():
    def r1(x): return x[:2]
    def r2(x): return x[2:]

    problem = Problem(
        variables=torch.zeros(4),
        costs=[
            CostTerm(residual_fn=r1, weight=2.0, kind="soft"),
            CostTerm(residual_fn=r2, weight=0.5, kind="soft"),
            CostTerm(residual_fn=r1, weight=1.0, kind="constraint_leq_zero"),
        ]
    )
    x = torch.ones(4)
    r = problem.total_residual(x)
    # r1 gives [1,1]*2.0 = [2,2], r2 gives [1,1]*0.5 = [0.5,0.5]
    assert r.shape == (4,)
    assert torch.allclose(r[:2], torch.tensor([2.0, 2.0]))
    assert torch.allclose(r[2:], torch.tensor([0.5, 0.5]))


def test_problem_constraint_residual():
    def r1(x): return x[:2]
    problem = Problem(
        variables=torch.zeros(4),
        costs=[
            CostTerm(residual_fn=r1, weight=1.0, kind="soft"),
            CostTerm(residual_fn=r1, weight=3.0, kind="constraint_leq_zero"),
        ]
    )
    x = torch.ones(4)
    cr = problem.constraint_residual(x)
    assert cr.shape == (2,)
    assert torch.allclose(cr, torch.tensor([3.0, 3.0]))


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return br.load_urdf(urdf)


def test_our_lm_converges_quadratic():
    """Our LM converges on a simple quadratic."""
    b = torch.tensor([1.0, 2.0, 3.0])

    def residual(x):
        return x - b

    problem = Problem(
        variables=torch.zeros(3),
        costs=[CostTerm(residual_fn=residual, weight=1.0)],
        lower_bounds=None,
        upper_bounds=None,
    )
    result = LevenbergMarquardt().solve(problem, max_iter=10)
    assert torch.allclose(result, b, atol=1e-4), f"Expected {b}, got {result}"


def test_problem_jacobian_fn_is_called():
    """jacobian_fn is stored on Problem and called by our LM."""
    calls = []

    def residual(x):
        return x

    def jac_fn(x):
        calls.append(1)
        return torch.eye(len(x))

    problem = Problem(
        variables=torch.zeros(3),
        costs=[CostTerm(residual_fn=residual, weight=1.0)],
        lower_bounds=None,
        upper_bounds=None,
        jacobian_fn=jac_fn,
    )
    LevenbergMarquardt().solve(problem, max_iter=1)
    assert len(calls) >= 1, "jacobian_fn was never called"


def test_our_lm_autodiff_matches_pypose_lm(panda):
    """Our LM (autodiff) and PyPose LM reach comparable IK solutions (within 1 cm)."""
    q0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(q0)[hand_idx].detach().clone()
    target[0] += 0.05

    def make_problem():
        costs = [
            CostTerm(functools.partial(pose_residual, robot=panda, target_link_index=hand_idx,
                                       target_pose=target, pos_weight=1.0, ori_weight=0.1), weight=1.0),
            CostTerm(functools.partial(limit_residual, robot=panda), weight=0.1),
            CostTerm(functools.partial(rest_residual, q_rest=panda._q_default), weight=0.01),
        ]
        return Problem(variables=q0.clone(), costs=costs,
                       lower_bounds=panda.joints.lower_limits.clone(),
                       upper_bounds=panda.joints.upper_limits.clone())

    result_ours = LevenbergMarquardt().solve(make_problem(), max_iter=20)
    result_pypose = PyposeLevenbergMarquardt().solve(make_problem(), max_iter=20)

    fk_ours = panda.forward_kinematics(result_ours)
    fk_pypose = panda.forward_kinematics(result_pypose)
    err_ours = (fk_ours[hand_idx, :3] - target[:3]).norm().item()
    err_pypose = (fk_pypose[hand_idx, :3] - target[:3]).norm().item()
    assert abs(err_ours - err_pypose) < 0.01, \
        f"Our LM err={err_ours:.4f}, PyPose err={err_pypose:.4f}"


def test_our_lm_uses_jacobian_fn_when_provided(panda):
    """Our LM with jacobian_fn gives correct IK solution."""
    q0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(q0)[hand_idx].detach()
    rest = panda._q_default.clone()

    costs = [
        CostTerm(functools.partial(pose_residual, robot=panda, target_link_index=hand_idx,
                                   target_pose=target, pos_weight=1.0, ori_weight=0.1), weight=1.0),
        CostTerm(functools.partial(limit_residual, robot=panda), weight=0.1),
        CostTerm(functools.partial(rest_residual, q_rest=rest), weight=0.01),
    ]

    def jac_fn(q):
        rows = [
            compute_jacobian(panda, q, hand_idx, target, 1.0, 0.1) * 1.0,
            limit_jacobian(q, panda) * 0.1,
            rest_jacobian(q, rest) * 0.01,
        ]
        return torch.cat(rows, dim=0)

    problem = Problem(
        variables=q0.clone(), costs=costs,
        lower_bounds=panda.joints.lower_limits.clone(),
        upper_bounds=panda.joints.upper_limits.clone(),
        jacobian_fn=jac_fn,
    )
    result = LevenbergMarquardt().solve(problem, max_iter=5)
    fk = panda.forward_kinematics(result)
    assert (fk[hand_idx, :3] - target[:3]).norm().item() < 0.05


def test_gn_solves_simple_quadratic():
    """GN should solve min 0.5*||x - target||^2."""
    from better_robot.solvers import SOLVERS
    from better_robot.solvers.problem import Problem
    from better_robot.costs.cost_term import CostTerm
    import torch
    target = torch.tensor([1.0, 2.0, 3.0])
    ct = CostTerm(residual_fn=lambda x: x - target, weight=1.0)
    problem = Problem(variables=torch.zeros(3), costs=[ct])
    solver = SOLVERS.get("gn")()
    result = solver.solve(problem, max_iter=20)
    assert result.shape == (3,)
    assert torch.allclose(result, target, atol=1e-3)


def test_adam_solves_simple_quadratic():
    """Adam should minimize ||x - target||^2."""
    from better_robot.solvers import SOLVERS
    from better_robot.solvers.problem import Problem
    from better_robot.costs.cost_term import CostTerm
    import torch
    target = torch.tensor([1.0, 2.0, 3.0])
    ct = CostTerm(residual_fn=lambda x: x - target, weight=1.0)
    problem = Problem(variables=torch.zeros(3), costs=[ct])
    solver = SOLVERS.get("adam")(lr=0.1)
    result = solver.solve(problem, max_iter=200)
    assert result.shape == (3,)
    assert torch.allclose(result, target, atol=0.1)


def test_lbfgs_solves_simple_quadratic():
    """LBFGS should solve min 0.5*||x - target||^2."""
    from better_robot.solvers import SOLVERS
    from better_robot.solvers.problem import Problem
    from better_robot.costs.cost_term import CostTerm
    import torch
    target = torch.tensor([1.0, 2.0, 3.0])
    ct = CostTerm(residual_fn=lambda x: x - target, weight=1.0)
    problem = Problem(variables=torch.zeros(3), costs=[ct])
    solver = SOLVERS.get("lbfgs")()
    result = solver.solve(problem, max_iter=40)
    assert result.shape == (3,)
    assert torch.allclose(result, target, atol=1e-3)
