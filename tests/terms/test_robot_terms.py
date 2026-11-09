"""Robot terms: analytic Jacobians equal autodiff; IK reaches reachable targets inside the limits."""

from dataclasses import dataclass
from typing import ClassVar, final

import pytest
import torch

import better_robot as br
from better_robot import optim, terms
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics
from better_robot.lie import so3
from better_robot.model import JointManifold, JointModel, JointRevolute
from tests.support.robots import g1_urdf, panda_urdf

FRAME = "panda_hand"


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_urdf(), dtype=torch.float64)


def _random_q(model, batch, seed):
    return terms.random_start(model, model.q_neutral.expand(batch, -1), generator=torch.Generator().manual_seed(seed))


def test_analytic_jacobians_match_autodiff(panda):
    q = _random_q(panda, 6, 0)
    goal = forward_kinematics(panda, _random_q(panda, 6, 1)).frame_poses[:, panda.frame_id(FRAME)]
    problem = terms.ik_problem(panda, {FRAME: goal}, q0=q + 4.0, limit_weight=2.0, rest_weight=0.5)
    problem.cached(lambda pose: pose[:, :3], name="point", reads=(f"target/{FRAME}",))
    problem.residual(
        "position", terms.frame_position(panda, frame=FRAME, target="point"), kernel=optim.Huber(delta=0.1)
    )
    errors = optim.check_jacobian(problem, {"q": q + 4.0})
    assert set(errors) == {f"pose/{FRAME}", "limits", "rest", "position"}
    # 1e-5: URDF values are float32-rounded, which limits analytic-vs-autodiff agreement of H.
    assert max(errors.values()) < 1e-5, errors


def test_ik_with_restarts_reaches_every_target_inside_limits(panda):
    goal = forward_kinematics(panda, _random_q(panda, 32, 2)).frame_poses[:, panda.frame_id(FRAME)]
    q0 = _random_q(panda, 32, 3)
    result = terms.solve_ik(panda, {FRAME: goal}, q0=q0, restarts=16, generator=torch.Generator().manual_seed(4))
    q = result.x["q"]
    reached = forward_kinematics(panda, q).frame_poses[:, panda.frame_id(FRAME)]
    assert float((reached[:, :3] - goal[:, :3]).norm(dim=-1).max()) < 1e-6
    assert bool(((q >= panda.lower_position_limit) & (q <= panda.upper_position_limit)).all())


def test_ik_restarts_need_a_generator(panda):
    q0 = panda.q_neutral[None]
    goal = forward_kinematics(panda, q0).frame_poses[:, panda.frame_id(FRAME)]
    with pytest.raises(br.exceptions.ArgumentError, match="generator"):
        terms.solve_ik(panda, {FRAME: goal}, q0=q0, restarts=1)


def test_random_start_and_solve_ik_check_their_arguments(panda):
    q0 = panda.q_neutral.expand(2, -1)
    generator = torch.Generator().manual_seed(0)
    for wrong in (panda.q_neutral, q0[:, :3]):
        with pytest.raises(br.exceptions.ShapeError, match=f"q0 must be \\(B, {panda.nq}\\)"):
            terms.random_start(panda, wrong, generator=generator)
    with pytest.raises(br.exceptions.ArgumentError, match="generator"):
        terms.random_start(panda, q0, generator=None)
    goal = forward_kinematics(panda, q0).frame_poses[:, panda.frame_id(FRAME)]
    with pytest.raises(br.exceptions.ArgumentError, match="restarts must be at least 0"):
        terms.solve_ik(panda, {FRAME: goal}, q0=q0, restarts=-3)


def test_ik_problem_terms_read_the_state_by_name(panda):
    goal = forward_kinematics(panda, _random_q(panda, 4, 2)).frame_poses[:, panda.frame_id(FRAME)]
    problem = terms.ik_problem(panda, {FRAME: goal}, q0=_random_q(panda, 4, 3))
    assert problem.terms[f"pose/{FRAME}"].reads[0] == "state"


def test_robot_manifold_accepts_an_empty_batch(panda):
    # tangent_dim used to index x[0], which raises IndexError for B == 0.
    problem = optim.Problem(
        {"q": torch.zeros(0, panda.nq, dtype=torch.float64)}, manifolds={"q": terms.RobotManifold(panda)}
    )
    problem.residual("rest", terms.rest(panda, torch.zeros(panda.nq, dtype=torch.float64)))
    result = optim.solve(problem, optim.LevenbergMarquardt(), iterations=2)
    assert result.x["q"].shape == (0, panda.nq) and result.loss.shape == (0,)


@pytest.fixture(scope="module")
def g1():
    return br.load(g1_urdf(), root_joint=br.model.JointFreeFlyer(), dtype=torch.float64)


def test_finite_difference_any_order_in_world_translation_and_velocity_limit(g1):
    generator = torch.Generator().manual_seed(0)
    q = br.kinematics.integrate(
        g1, g1.q_neutral.expand(1, 8, -1), 0.3 * torch.randn(1, 8, g1.nv, generator=generator, dtype=torch.float64)
    )
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
    unset = g1.with_values(velocity_limit=torch.zeros_like(g1.velocity_limit))  # MJCF stores 0
    assert bool((terms.velocity_limit(unset, dt=dt).fn(q) == 0).all())  # a stored 0 is no limit, not a zero one
    problem = optim.Problem({"q": q}, manifolds={"q": terms.RobotManifold(g1)}, time=("q",))
    problem.residual("jerk", term)
    problem.residual("velocity_limit", limit)
    assert max(optim.check_jacobian(problem).values()) < 1e-9


# A joint kind that stores a rotation vector: nq == nv == 3, but on its own manifold (record 0005 rejected this storage
# for the library's ball joint; a user kind may still choose it, record 0025).
ROTATION_VECTOR = JointManifold(
    name="rotation_vector",
    nq=3,
    nv=3,
    integrate=lambda q, v: so3.log(so3.compose(so3.exp(q), so3.exp(v))),
    difference=lambda q0, q1: so3.log(so3.compose(so3.inverse(so3.exp(q0)), so3.exp(q1))),
)


@final
@dataclass(frozen=True)
class JointRotationVector(JointModel):
    """Ball joint in exponential coordinates: ``q`` is a rotation vector, ``v`` the body angular velocity."""

    nq: ClassVar[int] = 3
    nv: ClassVar[int] = 3
    manifold: ClassVar[JointManifold] = ROTATION_VECTOR
    axis: ClassVar[None] = None

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.zeros_like(q), so3.exp(q)), -1)

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
        return torch.cat((torch.zeros_like(eye), eye)).expand(*q.shape[:-1], 6, 3)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(3)

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(3, generator=generator, dtype=lower.dtype)


def test_terms_follow_the_declared_manifold_not_nq_equal_nv():
    """A rotation-vector joint has ``nq == nv`` but is not a vector space: no bound, sample or identity Jacobian."""
    builder = ModelBuilder("rotation_vector_arm")
    base, link, tip = builder.add_body("base"), builder.add_body("link"), builder.add_body("tip")
    builder.add_joint("ball", kind=JointRotationVector(), parent=base, child=link, lower=-1.0, upper=1.0)
    origin = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    elbow = JointRevolute(axis=(0.0, 1.0, 0.0))
    builder.add_joint("elbow", kind=elbow, parent=link, child=tip, origin=origin, lower=-2.0, upper=2.0)
    model = build_model(builder.finalize()).to(dtype=torch.float64)
    q_columns, v_columns = br.kinematics.euclidean_columns(model)
    assert q_columns.tolist() == [3] and v_columns.tolist() == [3]

    manifold = terms.RobotManifold(model)
    x = torch.tensor([[0.0, 0.0, 0.9, 0.3]], dtype=torch.float64)
    low, high = manifold.tangent_bounds(x, lower=model.lower_position_limit, upper=model.upper_position_limit)
    assert bool(torch.isinf(low[:, :3]).all() and torch.isinf(high[:, :3]).all())  # a rotation step is no box step
    torch.testing.assert_close(high[:, 3:], model.upper_position_limit[3:] - x[:, 3:])

    q = torch.tensor([[0.8, -0.7, 1.3, 2.5]], dtype=torch.float64)
    for term in (terms.rest(model, model.q_neutral), terms.joint_limits(model)):
        assert term.jacobian is None  # autodiff through retract: right on any manifold
    path = torch.stack([q[0] * (0.2 * t) for t in range(5)])[None]
    problem = optim.Problem({"q": path}, manifolds={"q": manifold}, time=("q",))
    problem.residual("rest", terms.rest(model, model.q_neutral), window=1)
    problem.residual("limits", terms.joint_limits(model), window=1)
    problem.residual("acceleration", terms.finite_difference(model, order=2, dt=0.1))
    errors = optim.check_jacobian(problem)
    assert set(errors) == {"rest", "limits", "acceleration"} and max(errors.values()) < 1e-9, errors

    start = terms.random_start(model, model.q_neutral.expand(4, -1), generator=torch.Generator().manual_seed(0))
    assert bool((start[:, :3] == 0).all()) and bool((start[:, 3] != 0).all())  # only the elbow is sampled
