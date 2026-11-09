"""One evaluation of a ``Problem``: values computed at most once, weighted terms and their losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, final

import torch

from ..exceptions import ShapeError

if TYPE_CHECKING:
    from .problem import Problem, Term

Values = Mapping[str, torch.Tensor]


@final
class Evaluation:
    """Values of one evaluation: variables, data and cached values computed at most once."""

    def __init__(self, problem: Problem, x: Values) -> None:
        self._problem = problem
        self._values: dict[str, Any] = {**problem.data, **x}

    def __getitem__(self, name: str) -> Any:
        if name not in self._values:
            fn, reads = self._problem._cached[name]
            self._values[name] = fn(*(self[read] for read in reads))
        return self._values[name]

    def args(self, reads: tuple[str, ...]) -> list[Any]:
        """Values of ``reads`` in order."""
        return [self[read] for read in reads]


def active_terms(problem: Problem) -> list[Term]:
    """Terms whose weight, after the problem's overrides, is not the number 0."""
    out = []
    for term in problem.terms.values():
        weight = weight_of(problem, term)
        if isinstance(weight, torch.Tensor) or weight != 0.0:
            out.append(term)
    return out


def weight_of(problem: Problem, term: Term) -> float | torch.Tensor:
    """Weight of ``term`` under the problem's overrides."""
    return problem.weights.get(term.name, term.weight)


def zero_loss(problem: Problem) -> torch.Tensor:
    """``(B,)`` zeros in the dtype and on the device of the first variable."""
    first = next(iter(problem.variables.values()))
    return first.new_zeros(first.shape[0])


def depends(problem: Problem, reads: tuple[str, ...]) -> set[str]:
    """Variables that ``reads`` depend on, following cached values."""
    out: set[str] = set()
    for read in reads:
        if read in problem.variables:
            out.add(read)
        elif read in problem._cached:
            out |= depends(problem, problem._cached[read][1])
    return out


def term_output(term: Term, evaluation: Evaluation, info: dict[str, dict[str, Any]] | None = None) -> torch.Tensor:
    """Value of ``term``; the ``info`` dict of a term returning ``(value, info)`` goes to ``info[term.name]``."""
    value = term.fn(*evaluation.args(term.reads))
    if isinstance(value, tuple):
        value, extra = value
        if info is not None:
            info[term.name] = {key: v.detach() if isinstance(v, torch.Tensor) else v for key, v in extra.items()}
    return value


def weighted(
    problem: Problem, term: Term, evaluation: Evaluation, info: dict[str, dict[str, Any]] | None = None
) -> torch.Tensor:
    """Scalar term times its weight, or residual times the square root of its weight."""
    value = term_output(term, evaluation, info)
    weight = weight_of(problem, term)
    if isinstance(weight, torch.Tensor) and weight.ndim not in (0, value.ndim - 1, value.ndim):
        raise ShapeError(
            f"weight of {term.name!r} has shape {tuple(weight.shape)} but the term returns "
            f"{tuple(value.shape)}; give the weight that shape, with or without the batch axis"
        )
    if term.kind == "scalar":
        return value * weight
    return value * (weight.sqrt() if isinstance(weight, torch.Tensor) else weight**0.5)


def residual_loss(term: Term, weighted_value: torch.Tensor) -> torch.Tensor:
    """``(B,)`` loss of a weighted residual: half its squared norm, or the robust kernel over its last axis."""
    if term.kernel is None:
        return 0.5 * weighted_value.square().flatten(1).sum(-1)
    return term.kernel.rho(weighted_value.square().sum(-1)).reshape(weighted_value.shape[0], -1).sum(-1)


def irls(term: Term, weighted_value: torch.Tensor) -> torch.Tensor | float:
    """Square root of the robust weight of each group, broadcastable to the residual."""
    if term.kernel is None:
        return 1.0
    return term.kernel.weight(weighted_value.detach().square().sum(-1)).sqrt().unsqueeze(-1)


def evaluate(
    problem: Problem, x: Values | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Total loss ``(B,)``, the loss ``{name: (B,)}`` of every active term and the terms' ``info``."""
    evaluation = Evaluation(problem, x if x is not None else problem.variables)
    terms: dict[str, torch.Tensor] = {}
    info: dict[str, dict[str, Any]] = {}
    for term in active_terms(problem):
        value = weighted(problem, term, evaluation, info)
        terms[term.name] = value if term.kind == "scalar" else residual_loss(term, value)
    return sum(terms.values(), start=zero_loss(problem)), terms, info


def residuals(problem: Problem, x: Values | None = None) -> dict[str, torch.Tensor]:
    """Unweighted output of every active residual term."""
    evaluation = Evaluation(problem, x if x is not None else problem.variables)
    return {t.name: term_output(t, evaluation) for t in active_terms(problem) if t.kind == "residual"}


__all__ = [
    "Evaluation",
    "active_terms",
    "depends",
    "evaluate",
    "irls",
    "residual_loss",
    "residuals",
    "term_output",
    "weight_of",
    "weighted",
    "zero_loss",
]
