"""``optim.track``: one step of differential inverse kinematics, a warm quadratic program per call."""

import pytest
import torch

from better_robot import exceptions, optim


def _chase() -> tuple[optim.Problem, torch.Tensor]:
    """``(problem, targets)``: 16 points in the plane that chase targets moving from ``(0, 0)`` to ``(2, 1.5)`` over
    40 calls, then held, under the row ``x0 <= 0.5`` and the box ``x1`` in ``[-1, 1]``; ``targets`` is ``(60, 16, 2)``."""
    end = torch.tensor([2.0, 1.5], dtype=torch.float64) + 0.1 * torch.randn(
        16, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    share = (torch.arange(1, 61, dtype=torch.float64) / 40).clamp(max=1.0)
    targets = share[:, None, None] * end
    bounds = {
        "x": (
            torch.tensor([-torch.inf, -1.0], dtype=torch.float64),
            torch.tensor([torch.inf, 1.0], dtype=torch.float64),
        )
    }
    problem = optim.Problem({"x": torch.zeros(16, 2, dtype=torch.float64)}, data={"target": targets[0]}, bounds=bounds)
    problem.residual("chase", lambda x, target: x - target)
    problem.inequality("wall", lambda x: x[:, :1], upper=0.5)
    return problem, targets


def test_a_linear_row_and_the_box_close_in_by_at_most_the_gain_per_call_and_hold() -> None:
    """For rows linear in ``x`` the barrier is exact: each call keeps ``1 - gain`` of the distance to the row and to
    the box, no step exceeds ``step``, and the points end at the feasible point nearest their targets."""
    problem, targets = _chase()
    x, warm = dict(problem.variables), None
    for target in targets:
        before = torch.stack((0.5 - x["x"][:, 0], 1.0 - x["x"][:, 1]))
        new, warm = optim.track(problem.with_(data={"target": target}), x, warm=warm, gain=0.4, step=0.08)
        after = torch.stack((0.5 - new["x"][:, 0], 1.0 - new["x"][:, 1]))
        assert float(after.min()) >= -1e-9
        assert float((0.6 * before - after).max()) <= 1e-6
        assert float((new["x"] - x["x"]).abs().max()) <= 0.08 + 1e-12
        x = new
    torch.testing.assert_close(x["x"], torch.tensor([0.5, 1.0], dtype=torch.float64).expand(16, 2), rtol=0, atol=1e-6)


def test_track_refuses_what_it_cannot_step() -> None:
    """A gain outside (0, 1], a negative damping, a step limit that is not positive or has the wrong shape, and
    banded normal equations; a problem with nothing but a step limit is stepped."""
    problem, _ = _chase()
    x = dict(problem.variables)
    for options in ({"gain": 0.0}, {"gain": 1.5}, {"damping": -1.0}, {"step": 0.0}, {"step": -0.1}):
        with pytest.raises(exceptions.ArgumentError):
            optim.track(problem, x, **{"step": 0.1} | options)
    with pytest.raises(exceptions.ShapeError):
        optim.track(problem, x, step=torch.ones(3, dtype=torch.float64))
    free = optim.Problem({"x": torch.zeros(4, 2)}, data={"target": torch.ones(4, 2)})
    free.residual("chase", lambda x, target: x - target)
    assert optim.track(free, dict(free.variables), step=0.1)[0]["x"].shape == (4, 2)
    timed = optim.Problem({"x": torch.zeros(2, 5, 2)}, time=("x",), bounds={"x": (-torch.ones(2), torch.ones(2))})
    timed.residual("smooth", lambda x: x[:, 1:] - x[:, :-1], window=2)
    with pytest.raises(exceptions.ArgumentError):
        optim.track(timed, dict(timed.variables), step=0.1)


def _walled() -> optim.Problem:
    """64 points in six dimensions that chase far targets under 24 random rows pressing at once."""
    generator = torch.Generator().manual_seed(0)
    walls = torch.randn(24, 6, generator=generator, dtype=torch.float64)
    target = 3 * torch.randn(64, 6, generator=generator, dtype=torch.float64)
    problem = optim.Problem({"x": torch.zeros(64, 6, dtype=torch.float64)}, data={"target": target})
    problem.residual("chase", lambda x, target: x - target)
    problem.inequality("walls", lambda x: x @ walls.T, upper=1.0)
    return problem


def test_more_iterations_per_call_make_more_programs_exact() -> None:
    """``iterations`` reaches `solve_qp`: one cold call is exact on 1 of the 64 points at the default 20 iterations
    and on 31 at 200, with a smaller primal residual."""
    problem = _walled()
    few = optim.track(problem, dict(problem.variables), step=10.0)[1]
    many = optim.track(problem, dict(problem.variables), step=10.0, iterations=200)[1]
    assert int(few.exact.sum()) <= 4
    assert int(many.exact.sum()) >= 16
    assert float(many.primal_residual.max()) < 0.5 * float(few.primal_residual.max())
