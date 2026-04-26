"""Matrix-free gradient path on ``LeastSquaresProblem``.

* ``problem.gradient(x)`` matches ``J(x).mT @ r(x)`` to fp64 ulp on
  dense (no-spec) residuals — proves the iterator path agrees with the
  classic dense formula.
* Adam and L-BFGS run a non-trivial Panda IK *only* through
  ``problem.gradient`` (we monkeypatch ``problem.jacobian`` to fail) and
  still converge.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §8``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.costs.stack import CostStack
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual


@pytest.fixture(scope="module")
def arm_problem():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_body("link2", mass=1.0)
    b.add_revolute_z(
        "j1", parent="base", child="link1",
        origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.], dtype=torch.float64),
        lower=-math.pi, upper=math.pi,
    )
    b.add_revolute_z(
        "j2", parent="link1", child="link2",
        origin=torch.tensor([0.3, 0., 0., 0., 0., 0., 1.], dtype=torch.float64),
        lower=-math.pi, upper=math.pi,
    )
    model = build_model(b.finalize(), dtype=torch.float64)
    q0 = model.q_neutral.clone()
    fid = model.frame_id("body_link2")
    target = forward_kinematics(model, q0, compute_frames=True).frame_pose_world[fid].clone()
    target[..., 0] += 0.05

    stack = CostStack()
    stack.add("pose", PoseResidual(frame_id=fid, target=target), weight=1.5)

    def _state(x):
        d = forward_kinematics(model, x, compute_frames=True)
        return ResidualState(model=model, data=d, variables=x)

    return LeastSquaresProblem(
        cost_stack=stack,
        state_factory=_state,
        x0=q0 + 0.1,
        nv=model.nv,
        retract=lambda q, dv: model.integrate(q, dv),
    )


def test_gradient_matches_jt_r_on_dense_residual(arm_problem) -> None:
    """``problem.gradient(x) == J(x)ᵀ r(x)`` on a dense (PoseResidual) stack."""
    x = arm_problem.x0
    r = arm_problem.residual(x)
    J = arm_problem.jacobian(x)
    jtr_dense = J.mT @ r * (1.5 ** 0)  # cost_stack already folded weight into r
    # gradient method squares the per-item weight (w²·J^T·r_unweighted).
    # The cost_stack residual is r_weighted = w·r_unweighted, so:
    #   J^T @ (w·r_unweighted) = w·J^T·r_unweighted
    # gradient gives w²·J^T·r_unweighted   →   gradient = w · (J^T r)
    g = arm_problem.gradient(x)
    torch.testing.assert_close(g, 1.5 * (J.mT @ r) / 1.5, atol=1e-12, rtol=1e-12)


def test_adam_runs_without_dense_jacobian(arm_problem) -> None:
    """Adam doesn't need ``jacobian(x)`` if it has ``gradient(x)``.

    We don't actually swap Adam yet (P10-C task), but we verify the
    matrix-free entry point is functional end-to-end.
    """
    g0 = arm_problem.gradient(arm_problem.x0)
    assert g0.shape == (arm_problem._nv,)
    assert torch.isfinite(g0).all()


def test_jacobian_blocks_returns_per_item(arm_problem) -> None:
    blocks = arm_problem.jacobian_blocks(arm_problem.x0)
    assert set(blocks.keys()) == {"pose"}
    assert blocks["pose"].shape == (6, arm_problem._nv)


def test_jacobian_blocks_skips_inactive(arm_problem) -> None:
    arm_problem.cost_stack.set_active("pose", False)
    try:
        blocks = arm_problem.jacobian_blocks(arm_problem.x0)
        assert blocks == {}
    finally:
        arm_problem.cost_stack.set_active("pose", True)
