"""Hard constraints by the augmented-Lagrangian method, on top of any solver."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import torch

from .phases import Result, solve
from .problem import Problem
from .solvers import Solver


def _shifted(term, shift_name: str, inequality: bool):
    """The term ``c + shift`` (clamped at 0 from below for ``c <= 0``), reading the shift from data."""

    def fn(*args):
        value = term.fn(*args[:-1]) + args[-1]
        return value.clamp_min(0.0) if inequality else value

    jac = None
    if term.jac is not None and not inequality:
        jac = lambda *args: term.jac(*args[:-1])  # noqa: E731 - the shift is constant
    return dataclasses.replace(term, fn=fn, reads=(*term.reads, shift_name), jac=jac, kernel=None)


def solve_constrained(
    problem: Problem,
    solver: Solver,
    *,
    equalities: Sequence[str] = (),
    inequalities: Sequence[str] = (),
    outer: int = 10,
    iters: int = 30,
    penalty: float = 10.0,
    growth: float = 10.0,
    tolerance: float = 1e-6,
    x: Mapping[str, torch.Tensor] | None = None,
) -> Result:
    """Minimize the problem subject to ``term = 0`` (``equalities``) and ``term <= 0`` (``inequalities``).

    The constraints are residual terms already registered on ``problem``, named here; their
    own weight and kernel are ignored. Each of at most ``outer`` rounds runs ``solver`` for
    ``iters`` steps on the problem with every constraint replaced by the least-squares term
    ``½ ρ ‖c + λ/ρ‖²``, then updates the multipliers ``λ``. The penalty ``ρ`` (per batch
    element, starting at ``penalty``) grows by ``growth`` when the violation did not fall
    to a quarter. Stops when the largest violation of every batch element is below
    ``tolerance``. ``Result.history["violation"]`` is ``(rounds, B)``.
    """
    names = [*equalities, *inequalities]
    unknown = [name for name in names if name not in problem.terms or problem.terms[name].kind != "residual"]
    if unknown or not names:
        raise KeyError(f"constraints must name residual terms of the problem; not found: {unknown}")
    view = problem.with_()
    for name in names:
        view.terms[name] = _shifted(problem.terms[name], f"{name}.shift", name in inequalities)
    plain = problem.with_(weights=dict.fromkeys(names, 1.0))
    current = {key: value.detach() for key, value in (x if x is not None else problem.variables).items()}
    with torch.no_grad():
        values = plain.residuals(current)
    multipliers = {name: torch.zeros_like(values[name]) for name in names}
    rho = problem._zeros() + penalty
    previous = torch.full_like(rho, float("inf"))
    violations, total, result = [], 0, None
    for _ in range(outer):
        weights = {}
        for name in names:
            shaped = rho.reshape(-1, *([1] * (multipliers[name].ndim - 1)))
            view.data[f"{name}.shift"] = multipliers[name] / shaped
            weights[name] = shaped
        result = solve(view.with_(weights=weights), solver, iters=iters, x=current)
        current, total = result.x, total + result.iters
        with torch.no_grad():
            values = plain.residuals(current)
        violation = torch.zeros_like(rho)
        for name in names:
            shaped = rho.reshape(-1, *([1] * (multipliers[name].ndim - 1)))
            multipliers[name] = multipliers[name] + shaped * values[name]
            broken = values[name]
            if name in inequalities:
                multipliers[name] = multipliers[name].clamp_min(0.0)
                broken = broken.clamp_min(0.0)
            violation = torch.maximum(violation, broken.abs().flatten(1).amax(-1))
        violations.append(violation)
        if bool((violation < tolerance).all()):
            break
        rho = torch.where(violation > 0.25 * previous, rho * growth, rho)
        previous = violation
    history = {"violation": torch.stack(violations)}
    return Result(x=current, loss=result.loss, done=violations[-1] < tolerance, iters=total, history=history)


__all__ = ["solve_constrained"]
