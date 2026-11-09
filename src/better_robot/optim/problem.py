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

from ..exceptions import ArgumentError
from .evaluation import Values, evaluate, residuals
from .kernels import RobustKernel
from .linear import ArrowNormal, BandedNormal, BandLayout, Normal
from .linearization import gradient, linearize
from .manifold import Euclidean, Manifold


@dataclass(frozen=True)
class TermFn:
    """A term function bundled with the names it reads and, optionally, its analytic Jacobian and time window.

    ``fn(*reads)`` returns the residual. ``jacobian(*reads)`` returns ``{variable: (B, m, n)}``:
    one block per variable, in that variable's tangent coordinates, with the chain rule
    through any cached value already applied. ``window`` declares a time-local term (see
    ``Problem.residual``). ``Problem.residual``, ``scalar`` and ``cached`` take a ``TermFn`` in place of
    a plain function.
    """

    fn: Callable[..., torch.Tensor]
    reads: tuple[str, ...]
    jacobian: Callable[..., dict[str, torch.Tensor]] | None = None
    window: int | None = None


@dataclass(frozen=True)
class Term:
    """One registered cost term."""

    name: str
    fn: Callable[..., torch.Tensor]
    kind: str  #: ``"residual"`` or ``"scalar"``.
    reads: tuple[str, ...]
    weight: float | torch.Tensor
    kernel: RobustKernel | None
    jacobian: Callable[..., dict[str, torch.Tensor]] | None
    window: int | None = None  #: time-local term: row ``t`` reads knots ``t .. t + window - 1``.


class Problem:
    """A batch of independent optimization problems over named tensors.

    Register terms with ``residual`` (least squares) or ``scalar`` (any ``(B,)`` loss) and shared
    intermediates with ``cached``; then minimize with ``solve`` or ``run``. A term may return
    ``(value, info)`` to report diagnostics beside its value (see ``evaluate``).

    - ``variables``: initial values, each ``(B, *event)`` with the same ``B``.
    - ``manifolds``: per-variable ``Manifold``; missing entries are ``Euclidean``.
    - ``bounds``: per-variable ``(lower, upper)`` broadcastable to the variable; enforced by
      clamping after every step.
    - ``data``: tensors that terms read but that are not optimized (targets, network outputs).
    - ``time``: names of variables shaped ``(B, T, *event)`` whose second axis is time. Terms
      registered with ``window=`` are time-local in these variables, which makes their
      Jacobians cheap. When every residual term is either time-local or reads no time
      variable, the normal equations are block-tridiagonal (``BandedNormal``), bordered by the
      variables without a time axis such as body shape or a camera (``ArrowNormal``), and solved
      in time linear in ``T``.
    - ``dense``: force dense normal equations (``Normal``) even when the banded form applies.

    ```python
    problem = Problem({"x": torch.zeros(4, 2)}, data={"target": torch.randn(4, 2)})
    problem.residual("fit", lambda x, target: x - target)
    normal, loss = problem.linearize(problem.variables)  # Normal, (4,)
    ```
    """

    def __init__(
        self,
        variables: Mapping[str, torch.Tensor],
        *,
        manifolds: Mapping[str, Manifold] | None = None,
        bounds: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        data: Mapping[str, Any] | None = None,
        time: tuple[str, ...] = (),
        dense: bool = False,
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
        self.dense = dense
        self._layouts: dict[tuple[int, int], BandLayout] = {}
        self.time = tuple(self._known_variables(dict.fromkeys(time), "time"))
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

    def cached(
        self, fn: Callable[..., Any] | TermFn, *, name: str | None = None, reads: tuple[str, ...] | None = None
    ) -> str:
        """Register a value shared by terms, computed at most once per evaluation; returns its name.

        Usable as a decorator; the name defaults to the function name. Terms read the value by that name.
        """
        if isinstance(fn, TermFn):
            fn, reads = fn.fn, reads if reads is not None else fn.reads
        name = name or fn.__name__
        self._require_new_name(name)
        self._cached[name] = (fn, self._resolve_reads(name, fn, reads))
        return name

    def residual(
        self,
        name: str,
        fn: Callable[..., torch.Tensor] | TermFn,
        *,
        weight: float | torch.Tensor = 1.0,
        kernel: RobustKernel | None = None,
        reads: tuple[str, ...] | None = None,
        jacobian: Callable[..., dict[str, torch.Tensor]] | None = None,
        window: int | None = None,
    ) -> None:
        """Add a least-squares term ``fn(...) -> (B, ..., k)``.

        Its loss is ``½ · weight · Σ r²``; with a ``kernel`` the last axis is one robust
        group and the loss is ``Σ kernel.rho(weight · ‖r‖²)`` over the other axes.
        A tensor ``weight`` has the shape of the output, with or without the batch axis.
        ``reads`` defaults to the parameter names of ``fn``. ``jacobian`` is optional: without it the
        Jacobian comes from autodiff. A ``TermFn`` brings its own ``reads``, ``jacobian`` and ``window``;
        a keyword given here replaces the bundle's.

        ``window=k`` declares a time-local term: the output is ``(B, T - k + 1, m)`` and row
        ``t`` depends only on knots ``t .. t + k - 1`` of the time variables (``k=1`` for a
        per-knot term, 2 for velocities, 3 for accelerations). Its autodiff Jacobian then costs
        ``k · m`` backward passes instead of one per row, and an analytic ``jacobian`` returns blocks
        ``(B, T - k + 1, m, k, n)`` per time variable and ``(B, T - k + 1, m, n)`` per variable
        without a time axis. The declaration is trusted: a row that reads outside its window
        gives a wrong Jacobian. ``check_jacobian`` tests it.
        """
        if isinstance(fn, TermFn):
            reads = reads if reads is not None else fn.reads
            jacobian = jacobian if jacobian is not None else fn.jacobian
            window = window if window is not None else fn.window
            fn = fn.fn
        self._add(name, fn, "residual", weight, kernel, jacobian, reads, window)

    def scalar(
        self,
        name: str,
        fn: Callable[..., torch.Tensor] | TermFn,
        *,
        weight: float | torch.Tensor = 1.0,
        reads: tuple[str, ...] | None = None,
    ) -> None:
        """Add a term ``fn(...) -> (B,)`` that is already a loss.

        First-order solvers minimize it like any other term. Gauss-Newton and LM only add its
        gradient; its curvature is ignored. A ``TermFn`` brings its ``reads``; it may not carry a
        ``jacobian`` or a ``window``.
        """
        if isinstance(fn, TermFn):
            if fn.jacobian is not None or fn.window is not None:
                raise ArgumentError(f"scalar term {name!r}: a scalar term takes no jacobian or window")
            fn, reads = fn.fn, reads if reads is not None else fn.reads
        self._add(name, fn, "scalar", weight, None, None, reads, None)

    def with_(
        self, *, weights: Mapping[str, float] | None = None, free: Mapping[str, torch.Tensor] | None = None
    ) -> "Problem":
        """Return a view of this problem with term weights overridden and/or coordinates frozen.

        ``weights[name] = 0`` switches a term off. ``free[variable]`` is a bool mask over that
        variable's tangent coordinates, ``(n,)`` or ``(B, n)``; unlisted variables stay free.
        For a time variable the mask may cover one knot; it is then repeated over time.
        """
        view = copy.copy(self)
        view.terms, view._cached, view.data = dict(self.terms), dict(self._cached), dict(self.data)
        if weights is not None:
            unknown = set(weights) - set(self.terms)
            if unknown:
                raise KeyError(f"weights for unknown terms {sorted(unknown)}; terms are {sorted(self.terms)}")
            view.weights = {**self.weights, **weights}
        if free is not None:
            view.free = {}
            for name, mask in self._known_variables(free, "free").items():
                span = self._slices[name]
                size = span.stop - span.start
                knots = self.variables[name].shape[1] if name in self.time else 1
                if mask.shape[-1] * knots != size and mask.shape[-1] != size:
                    raise ValueError(f"free mask of {name!r} has {mask.shape[-1]} entries, expected {size}")
                view.free[name] = mask.repeat(*([1] * (mask.ndim - 1)), size // mask.shape[-1])
        return view

    # ── evaluation ──────────────────────────────────────────────────────

    def evaluate(
        self, x: Values | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
        """Total loss, the loss of every active term and the terms' ``info``, from one evaluation.

        A term may return ``(value, info)`` instead of ``value``: ``info`` is a dict of diagnostics (counts,
        inlier ratios) kept beside the loss and never differentiated; its tensors are detached.

        Returns
        -------
        loss : Tensor, shape (B,)
            Differentiable w.r.t. ``x``; the sum of ``terms`` in registration order.
        terms : dict[str, Tensor]
            ``{name: (B,)}`` weighted loss of every active term.
        info : dict[str, dict]
            ``{name: info}`` for the active terms that returned one.
        """
        return evaluate(self, x)

    def losses(self, x: Values | None = None) -> dict[str, torch.Tensor]:
        """Loss of every active term. Returns ``{name: (B,)}``."""
        return evaluate(self, x)[1]

    def loss(self, x: Values | None = None) -> torch.Tensor:
        """Total loss per batch element, differentiable w.r.t. ``x``. Returns ``(B,)``."""
        return evaluate(self, x)[0]

    def residuals(self, x: Values | None = None) -> dict[str, torch.Tensor]:
        """Unweighted output of every active residual term."""
        return residuals(self, x)

    def linearize(self, x: Values | None = None) -> tuple[Normal | BandedNormal | ArrowNormal, torch.Tensor]:
        """Gauss-Newton normal equations and the loss at ``x``. Returns ``(Normal, (B,))``.

        All terms without an analytic Jacobian are differentiated together in one traced
        evaluation, so cached values are computed once and their dependence on the variables
        is kept. Robust kernels enter as iteratively reweighted least squares.
        """
        return linearize(self, x)

    def gradient(
        self, x: Values | None = None, *, create_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Loss ``(B,)``, its gradient ``(B, N)`` in tangent coordinates (frozen coordinates zeroed) and the
        detached loss of every active term ``{name: (B,)}``, from one evaluation.

        With ``create_graph=True`` the gradient stays differentiable w.r.t. data tensors.
        """
        return gradient(self, x, create_graph=create_graph)

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
                x[name], lower=low, upper=high
            )
        return lower, upper

    def slices(self) -> dict[str, slice]:
        """Position of each variable in the stacked tangent vector."""
        return dict(self._slices)

    # ── internals ───────────────────────────────────────────────────────

    def _add(self, name, fn, kind, weight, kernel, jacobian, reads, window) -> None:
        if name in self.terms:
            raise ValueError(f"a term named {name!r} already exists")
        reads = self._resolve_reads(name, fn, reads)
        self.terms[name] = Term(name, fn, kind, reads, weight, kernel, jacobian, window)

    def _require_new_name(self, name: str) -> None:
        if name in self.variables or name in self.data or name in self._cached:
            raise ValueError(f"name {name!r} is already used by a variable, data tensor or cached value")

    def _resolve_reads(self, name: str, fn: Callable[..., Any], reads: tuple[str, ...] | None) -> tuple[str, ...]:
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

    def _retract(self, x: Values, delta: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self.manifolds[name].retract(value, delta[:, self._slices[name]]) for name, value in x.items()}

    def _clamp(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, (lower, upper) in self.bounds.items():
            x[name] = torch.minimum(torch.maximum(x[name], lower), upper)
        return x


__all__ = ["Problem", "Term", "TermFn"]
