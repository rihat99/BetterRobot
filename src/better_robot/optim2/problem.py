"""``Problem``: named tensors, cached intermediates and cost terms written as plain functions.

A variable is a tensor ``(B, *event)``; the ``B`` problems of a batch are independent.
A term is a function whose parameter names are the variables, data tensors or cached
values it reads. ``Problem.loss`` is an ordinary differentiable ``(B,)`` tensor, so any
``torch.optim`` loop can minimize it; ``Problem.linearize`` gives Gauss-Newton and LM
their normal equations from the same terms.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .kernels import RobustKernel
from .linear import Normal
from .manifold import Euclidean, Manifold

Values = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TermFn:
    """A term function bundled with what it reads and, optionally, its analytic Jacobian.

    ``fn(*reads)`` returns the residual. ``jac(*reads)`` returns ``{variable: (B, m, n)}``:
    one block per variable, in that variable's tangent coordinates, with the chain rule
    through any cached value already applied.
    """

    fn: Callable[..., torch.Tensor]
    reads: tuple[str, ...]
    jac: Callable[..., dict[str, torch.Tensor]] | None = None

    def __call__(self, *args: Any) -> torch.Tensor:
        return self.fn(*args)


@dataclass(frozen=True)
class Term:
    """One registered cost term."""

    name: str
    fn: Callable[..., torch.Tensor]
    kind: str  #: ``"residual"`` or ``"scalar"``.
    reads: tuple[str, ...]
    weight: float | torch.Tensor
    kernel: RobustKernel | None
    jac: Callable[..., dict[str, torch.Tensor]] | None


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


class Problem:
    """A batch of independent optimization problems over named tensors.

    Parameters
    ----------
    variables : initial values, each ``(B, *event)`` with the same ``B``.
    manifolds : per-variable :class:`Manifold`; missing entries are Euclidean.
    bounds : per-variable ``(lower, upper)`` broadcastable to the variable; enforced by
        clamping after every step.
    data : tensors that terms read but that are not optimized (targets, network outputs).
    """

    def __init__(
        self,
        variables: Mapping[str, torch.Tensor],
        *,
        manifolds: Mapping[str, Manifold] | None = None,
        bounds: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        if not variables:
            raise ValueError("a Problem needs at least one variable")
        batch_sizes = {name: value.shape[0] for name, value in variables.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(f"variables must share the leading batch size, got {batch_sizes}")
        self.variables: dict[str, torch.Tensor] = dict(variables)
        self.data: dict[str, Any] = dict(data or {})
        clash = set(self.variables) & set(self.data)
        if clash:
            raise ValueError(f"names used both as variable and data: {sorted(clash)}")
        self.manifolds: dict[str, Manifold] = {name: Euclidean() for name in self.variables}
        self.manifolds.update(self._known_variables(manifolds, "manifolds"))
        self.bounds = dict(self._known_variables(bounds, "bounds"))
        self.terms: dict[str, Term] = {}
        self.weights: dict[str, float] = {}
        self.free: dict[str, torch.Tensor] | None = None
        self._cached: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {}
        self._slices: dict[str, slice] = {}
        offset = 0
        for name, value in self.variables.items():
            size = self.manifolds[name].tangent_dim(value)
            self._slices[name] = slice(offset, offset + size)
            offset += size
        self.tangent_dim = offset

    # ── registration ────────────────────────────────────────────────────

    def cached(self, fn: Callable[..., Any], *, name: str | None = None, reads: tuple[str, ...] | None = None):
        """Register a value shared by terms, computed at most once per evaluation.

        Usable as a decorator; the name defaults to the function name.
        """
        name = name or fn.__name__
        self._require_new_name(name)
        self._cached[name] = (fn, self._resolve_reads(name, fn, reads))
        return fn

    def residual(
        self,
        name: str,
        fn: Callable[..., torch.Tensor],
        *,
        weight: float | torch.Tensor = 1.0,
        kernel: RobustKernel | None = None,
        jac: Callable[..., dict[str, torch.Tensor]] | None = None,
        reads: tuple[str, ...] | None = None,
    ) -> None:
        """Add a least-squares term ``fn(...) -> (B, ..., k)``.

        Its loss is ``½ · weight · Σ r²``; with a ``kernel`` the last axis is one robust
        group and the loss is ``Σ kernel.rho(weight · ‖r‖²)`` over the other axes.
        ``jac`` is optional: without it the Jacobian comes from autodiff.
        """
        self._add(name, fn, "residual", weight, kernel, jac if jac is not None else getattr(fn, "jac", None), reads)

    def scalar(
        self,
        name: str,
        fn: Callable[..., torch.Tensor],
        *,
        weight: float | torch.Tensor = 1.0,
        reads: tuple[str, ...] | None = None,
    ) -> None:
        """Add a term ``fn(...) -> (B,)`` that is already a loss.

        First-order solvers minimize it like any other term. Gauss-Newton and LM only add its
        gradient; its curvature is ignored.
        """
        self._add(name, fn, "scalar", weight, None, None, reads)

    def with_(self, *, weights: Mapping[str, float] | None = None, free: Mapping[str, torch.Tensor] | None = None):
        """Return a view of this problem with term weights overridden and/or coordinates frozen.

        ``weights[name] = 0`` switches a term off. ``free[variable]`` is a bool mask over that
        variable's tangent coordinates, ``(n,)`` or ``(B, n)``; unlisted variables stay free.
        """
        view = copy.copy(self)
        if weights is not None:
            unknown = set(weights) - set(self.terms)
            if unknown:
                raise KeyError(f"weights for unknown terms {sorted(unknown)}; terms are {sorted(self.terms)}")
            view.weights = {**self.weights, **weights}
        if free is not None:
            view.free = dict(self._known_variables(free, "free"))
        return view

    # ── evaluation ──────────────────────────────────────────────────────

    def losses(self, x: Values | None = None) -> dict[str, torch.Tensor]:
        """Loss of every active term. Returns ``{name: (B,)}``."""
        evaluation = Evaluation(self, x if x is not None else self.variables)
        out = {}
        for term in self._active():
            value = self._weighted(term, evaluation)
            out[term.name] = value if term.kind == "scalar" else self._residual_loss(term, value)
        return out

    def loss(self, x: Values | None = None) -> torch.Tensor:
        """Total loss per batch element, differentiable w.r.t. ``x``. Returns ``(B,)``."""
        return sum(self.losses(x).values(), start=self._zeros())

    def residuals(self, x: Values | None = None) -> dict[str, torch.Tensor]:
        """Unweighted output of every active residual term."""
        evaluation = Evaluation(self, x if x is not None else self.variables)
        return {t.name: t.fn(*evaluation.args(t.reads)) for t in self._active() if t.kind == "residual"}

    def linearize(self, x: Values | None = None) -> tuple[Normal, torch.Tensor]:
        """Gauss-Newton normal equations and the loss at ``x``. Returns ``(Normal, (B,))``.

        All terms without an analytic Jacobian are differentiated together in one traced
        evaluation, so cached values are computed once and their dependence on the variables
        is kept. Robust kernels enter as iteratively reweighted least squares.
        """
        x = {name: value.detach() for name, value in (x if x is not None else self.variables).items()}
        terms = self._active()
        traced = [t for t in terms if t.kind == "scalar" or t.jac is None]
        batch, size = self._batch(), self.tangent_dim
        loss = self._zeros()
        H = loss.new_zeros(batch, size, size)
        g = loss.new_zeros(batch, size)
        with torch.enable_grad() if traced else torch.no_grad():
            if traced:
                delta = loss.new_zeros(batch, size, requires_grad=True)
                evaluation = Evaluation(self, self._retract(x, delta))
            else:
                evaluation = Evaluation(self, x)
            rows, scalar_total = [], None
            for term in terms:
                value = self._weighted(term, evaluation)
                if term.kind == "scalar":
                    loss = loss + value.detach()
                    scalar_total = value if scalar_total is None else scalar_total + value
                    continue
                loss = loss + self._residual_loss(term, value).detach()
                reweighted = (value * self._irls(term, value)).flatten(1)
                if term.jac is None:
                    rows.append(reweighted)
                    continue
                with torch.no_grad():
                    jacobian = self._analytic_jacobian(term, evaluation, value)
                H = H + jacobian.mT @ jacobian
                g = g + (jacobian.mT @ reweighted.detach().unsqueeze(-1)).squeeze(-1)
            if scalar_total is not None:
                g = g + torch.autograd.grad(scalar_total.sum(), delta, retain_graph=bool(rows))[0]
            if rows:
                stacked = torch.cat(rows, dim=1)  # (B, M)
                # Batch elements are independent, so row i of every element comes from one backward pass.
                seeds = torch.eye(stacked.shape[1], dtype=stacked.dtype, device=stacked.device)
                seeds = seeds.unsqueeze(1).expand(-1, batch, -1)
                (jacobian,) = torch.autograd.grad(stacked, delta, seeds, is_grads_batched=True)
                jacobian = jacobian.permute(1, 0, 2)  # (B, M, N)
                H = H + jacobian.mT @ jacobian
                g = g + (jacobian.mT @ stacked.detach().unsqueeze(-1)).squeeze(-1)
        mask = self._free_mask()
        if mask is not None:
            g = g * mask
            H = H * mask.unsqueeze(-1) * mask.unsqueeze(-2) + torch.diag_embed((1.0 - mask).expand(batch, size))
        return Normal(H=H, g=g), loss

    # ── stepping ────────────────────────────────────────────────────────

    def retract(self, x: Values, delta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step every variable along ``delta: (B, N)`` and clamp to bounds."""
        return self._clamp(self._retract(x, delta))

    def project(self, x: Values) -> dict[str, torch.Tensor]:
        """Map ambient tensors back onto their manifolds and bounds."""
        return self._clamp({name: self.manifolds[name].project(value) for name, value in x.items()})

    def step_bounds(self, x: Values) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Allowed tangent step ``(lower, upper)``, each ``(B, N)``, or ``None`` without bounds."""
        if not self.bounds:
            return None
        first = next(iter(x.values()))
        lower = first.new_full((first.shape[0], self.tangent_dim), -float("inf"))
        upper = first.new_full((first.shape[0], self.tangent_dim), float("inf"))
        for name, (low, high) in self.bounds.items():
            lower[:, self._slices[name]], upper[:, self._slices[name]] = self.manifolds[name].tangent_bounds(
                x[name], low, high
            )
        return lower, upper

    def slices(self) -> dict[str, slice]:
        """Position of each variable in the stacked tangent vector."""
        return dict(self._slices)

    # ── internals ───────────────────────────────────────────────────────

    def _add(self, name, fn, kind, weight, kernel, jac, reads) -> None:
        self._require_new_name(name)
        self.terms[name] = Term(name, fn, kind, self._resolve_reads(name, fn, reads), weight, kernel, jac)

    def _require_new_name(self, name: str) -> None:
        if name in self.variables or name in self.data or name in self._cached or name in self.terms:
            raise ValueError(f"name {name!r} is already used in this problem")

    def _resolve_reads(self, name: str, fn: Callable[..., Any], reads: tuple[str, ...] | None) -> tuple[str, ...]:
        if reads is None:
            reads = getattr(fn, "reads", None)
        if reads is None:
            reads = tuple(inspect.signature(fn).parameters)
        known = set(self.variables) | set(self.data) | set(self._cached)
        missing = [read for read in reads if read not in known]
        if missing:
            raise KeyError(
                f"{name!r} reads {missing}, which are not variables, data or cached values. "
                f"Known names: {sorted(known)}. Register cached values before the terms that read them."
            )
        return tuple(reads)

    def _known_variables(self, mapping: Mapping[str, Any] | None, what: str) -> Mapping[str, Any]:
        mapping = mapping or {}
        unknown = set(mapping) - set(self.variables)
        if unknown:
            raise KeyError(f"{what} given for unknown variables {sorted(unknown)}")
        return mapping

    def _active(self) -> list[Term]:
        out = []
        for term in self.terms.values():
            weight = self.weights.get(term.name, term.weight)
            if isinstance(weight, torch.Tensor) or weight != 0.0:
                out.append(term)
        return out

    def _batch(self) -> int:
        return next(iter(self.variables.values())).shape[0]

    def _zeros(self) -> torch.Tensor:
        first = next(iter(self.variables.values()))
        return first.new_zeros(first.shape[0])

    def _weighted(self, term: Term, evaluation: Evaluation) -> torch.Tensor:
        """Scalar term times its weight, or residual times the square root of its weight."""
        value = term.fn(*evaluation.args(term.reads))
        weight = self.weights.get(term.name, term.weight)
        if term.kind == "scalar":
            return value * weight
        return value * (weight.sqrt() if isinstance(weight, torch.Tensor) else weight**0.5)

    @staticmethod
    def _residual_loss(term: Term, weighted: torch.Tensor) -> torch.Tensor:
        if term.kernel is None:
            return 0.5 * weighted.square().flatten(1).sum(-1)
        return term.kernel.rho(weighted.square().sum(-1)).reshape(weighted.shape[0], -1).sum(-1)

    @staticmethod
    def _irls(term: Term, weighted: torch.Tensor) -> torch.Tensor | float:
        """Square root of the robust weight of each group, broadcastable to the residual."""
        if term.kernel is None:
            return 1.0
        return term.kernel.weight(weighted.detach().square().sum(-1)).sqrt().unsqueeze(-1)

    def _analytic_jacobian(self, term: Term, evaluation: Evaluation, weighted: torch.Tensor) -> torch.Tensor:
        """Dense ``(B, m, N)`` Jacobian of the weighted, reweighted residual from ``term.jac`` blocks."""
        blocks = term.jac(*evaluation.args(term.reads))
        weight = self.weights.get(term.name, term.weight)
        row_scale = (weight.sqrt() if isinstance(weight, torch.Tensor) else weight**0.5) * self._irls(term, weighted)
        if isinstance(row_scale, torch.Tensor):
            row_scale = row_scale.expand_as(weighted).flatten(1).unsqueeze(-1)
        rows = weighted.flatten(1).shape[1]
        jacobian = weighted.new_zeros(weighted.shape[0], rows, self.tangent_dim)
        for variable, block in blocks.items():
            jacobian[:, :, self._slices[variable]] = block.detach() * row_scale
        return jacobian

    def _retract(self, x: Values, delta: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: self.manifolds[name].retract(value, delta[:, self._slices[name]]) for name, value in x.items()
        }

    def _clamp(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, (lower, upper) in self.bounds.items():
            x[name] = torch.minimum(torch.maximum(x[name], lower), upper)
        return x

    def _free_mask(self) -> torch.Tensor | None:
        """Float mask ``(B or 1, N)``: 1 for free tangent coordinates, 0 for frozen ones."""
        if self.free is None:
            return None
        first = next(iter(self.variables.values()))
        rows = max([1] + [mask.shape[0] for mask in self.free.values() if mask.ndim == 2])
        out = first.new_ones(rows, self.tangent_dim)
        for name, mask in self.free.items():
            out[:, self._slices[name]] = mask.to(first.dtype)
        return out


def check_jacobian(problem: Problem, x: Values | None = None) -> dict[str, float]:
    """Largest absolute difference between each analytic ``jac`` and autodiff of the same term.

    Returns ``{term name: error}`` for the residual terms that have an analytic Jacobian.
    """
    errors = {}
    for term in problem._active():
        if term.kind != "residual" or term.jac is None:
            continue
        only = {name: 0.0 for name in problem.terms if name != term.name}
        analytic = problem.with_(weights=only)
        autodiff = copy.copy(analytic)
        autodiff.terms = {**problem.terms, term.name: Term(**{**term.__dict__, "jac": None})}
        normal_a, _ = analytic.linearize(x)
        normal_b, _ = autodiff.linearize(x)
        errors[term.name] = max(
            float((normal_a.H - normal_b.H).abs().max()), float((normal_a.g - normal_b.g).abs().max())
        )
    return errors


__all__ = ["Evaluation", "Problem", "Term", "TermFn", "check_jacobian"]
