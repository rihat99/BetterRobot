"""One evaluation of a ``Problem``: values computed at most once, weighted terms and their losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, final

import torch

from ..exceptions import ArgumentError, ShapeError, UnknownNameError

if TYPE_CHECKING:
    from .problem import Problem, Term

Values = Mapping[str, torch.Tensor]


def checked_values(problem: Problem, x: Values | None) -> Values:
    """``x``, or the problem's variables when ``x`` is ``None``, checked to hold one ``(B, *event)`` tensor per variable
    with the problem's batch size ``B``: an unknown name raises ``UnknownNameError`` (``x`` never overrides data), a
    missing variable ``ArgumentError``, another batch size or a 0-d tensor ``ShapeError``."""
    variables = problem.variables
    if x is None:
        return variables
    for name in x:
        if name not in variables:
            raise UnknownNameError("variable", name, valid=list(variables))
    missing = [name for name in variables if name not in x]
    if missing:
        raise ArgumentError(f"x has no value for the variables {missing}; give every variable")
    batch = next(iter(variables.values())).shape[0]
    for name, value in x.items():
        if value.ndim == 0 or value.shape[0] != batch:
            raise ShapeError(f"x[{name!r}] has shape {tuple(value.shape)}; a variable is (B, *event) with B = {batch}")
    return x


@final
class Evaluation:
    """Values of one evaluation: variables, data and cached values computed at most once."""

    def __init__(self, problem: Problem, x: Values) -> None:
        self._cached = problem.cached_values
        self._values: dict[str, Any] = {**problem.data, **x}

    def __getitem__(self, name: str) -> Any:
        if name not in self._values:
            cached = self._cached[name]
            self._values[name] = cached.fn(*(self[read] for read in cached.reads))
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
        elif read in problem.cached_values:
            out |= depends(problem, problem.cached_values[read].reads)
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
    evaluation = Evaluation(problem, checked_values(problem, x))
    terms: dict[str, torch.Tensor] = {}
    info: dict[str, dict[str, Any]] = {}
    for term in active_terms(problem):
        value = weighted(problem, term, evaluation, info)
        terms[term.name] = value if term.kind == "scalar" else residual_loss(term, value)
    return sum(terms.values(), start=zero_loss(problem)), terms, info


def residuals(problem: Problem, x: Values | None = None) -> dict[str, torch.Tensor]:
    """Unweighted output of every active residual term."""
    evaluation = Evaluation(problem, checked_values(problem, x))
    return {t.name: term_output(t, evaluation) for t in active_terms(problem) if t.kind == "residual"}


__all__ = [
    "Evaluation",
    "active_terms",
    "checked_values",
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
