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
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import Any, final

import torch

from ..exceptions import ArgumentError, ShapeError, UnknownNameError
from .evaluation import Values, evaluate, residuals
from .kernels import RobustKernel
from .linear import ArrowNormal, BandedNormal, Normal
from .linearization import gradient, linearize
from .manifold import Euclidean, Manifold


@final
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
    _: KW_ONLY
    jacobian: Callable[..., dict[str, torch.Tensor]] | None = None
    window: int | None = None


@final
@dataclass(frozen=True)
class Term:
    """One registered cost term."""

    name: str
    fn: Callable[..., torch.Tensor]
    _: KW_ONLY
    kind: str  #: ``"residual"`` or ``"scalar"``.
    reads: tuple[str, ...]
    weight: float | torch.Tensor
    kernel: RobustKernel | None
    jacobian: Callable[..., dict[str, torch.Tensor]] | None
    window: int | None = None  #: time-local term: row ``t`` reads knots ``t .. t + window - 1``.


@final
class Problem:
    """A batch of independent optimization problems over named tensors, read-only once built.

    Register terms with ``residual`` (least squares) or ``scalar`` (any ``(B,)`` loss) and shared
    intermediates with ``cached``; then minimize with ``solve`` or ``run``. A term may return
    ``(value, info)`` to report diagnostics beside its value (see ``evaluate``). Registration only adds
    new names to the problem it is called on; everything else, and every field of a registered term, is
    fixed and read through read-only mappings. ``with_`` returns a new problem with other data, weights or
    frozen coordinates. That problem is sealed, like the views ``run``, ``solve_constrained`` and
    ``solve_layer`` make and hand to callbacks: registering on it raises ``ArgumentError``.

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

    Tensors are held by reference, never copied: the variables, data and bounds given here, the data and weight
    tensors given to ``with_``, and everything a ``with_`` result shares with its parent, free masks included. The
    read-only mappings fix the names, not the values: a tensor edited in place changes every problem that holds
    it, so build a new problem or call ``with_`` instead.

    ```python
    problem = Problem({"x": torch.zeros(4, 2)}, data={"target": torch.randn(4, 2)})
    problem.residual("fit", lambda x, target: x - target)
    normal, loss = problem.linearize(problem.variables)  # Normal, (4,)
    ```
    """

    __slots__ = (
        "_variables",
        "_data",
        "_manifolds",
        "_bounds",
        "_time",
        "_dense",
        "_terms",
        "_weights",
        "_free",
        "_cached",
        "_sealed",
        "_slices",
        "_tangent_dim",
    )

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
            raise ArgumentError("a Problem needs at least one variable")
        for name, value in variables.items():
            if value.ndim == 0:
                raise ShapeError(f"variable {name!r} is 0-d; a variable is (B, *event) with a leading batch axis")
        batch_sizes = {name: value.shape[0] for name, value in variables.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ShapeError(f"variables must share the leading batch size, got {batch_sizes}")
        self._variables: dict[str, torch.Tensor] = dict(variables)
        self._data: dict[str, Any] = dict(data or {})
        clash = set(self._variables) & set(self._data)
        if clash:
            raise ArgumentError(f"names used both as variable and data: {sorted(clash)}")
        self._manifolds: dict[str, Manifold] = {name: Euclidean() for name in self._variables}
        self._manifolds.update(self._known_variables(manifolds))
        self._bounds = dict(self._known_variables(bounds))
        self._time = tuple(self._known_variables(dict.fromkeys(time)))
        self._dense = dense
        self._terms: dict[str, Term] = {}
        self._weights: dict[str, float | torch.Tensor] = {}
        self._free: dict[str, torch.Tensor] | None = None
        self._cached: dict[str, TermFn] = {}
        self._sealed = False
        self._slices: dict[str, slice] = {}
        offset = 0
        for name, value in self._variables.items():
            size = self._manifolds[name].tangent_dim(value)
            self._slices[name] = slice(offset, offset + size)
            offset += size
        self._tangent_dim = offset

    # ── read-only fields ────────────────────────────────────────────────

    @property
    def variables(self) -> Mapping[str, torch.Tensor]:
        """Initial values ``{name: (B, *event)}``."""
        return MappingProxyType(self._variables)

    @property
    def data(self) -> Mapping[str, Any]:
        """Tensors the terms read and the solvers never change; ``with_(data=)`` adds or replaces entries."""
        return MappingProxyType(self._data)

    @property
    def manifolds(self) -> Mapping[str, Manifold]:
        """The ``Manifold`` of every variable."""
        return MappingProxyType(self._manifolds)

    @property
    def bounds(self) -> Mapping[str, tuple[torch.Tensor, torch.Tensor]]:
        """``{variable: (lower, upper)}`` box bounds on the ambient tensor."""
        return MappingProxyType(self._bounds)

    @property
    def time(self) -> tuple[str, ...]:
        """Names of the variables whose second axis is time."""
        return self._time

    @property
    def dense(self) -> bool:
        """Whether the normal equations are dense even when the banded form applies."""
        return self._dense

    @property
    def terms(self) -> Mapping[str, Term]:
        """Registered terms by name, in registration order."""
        return MappingProxyType(self._terms)

    @property
    def cached_values(self) -> Mapping[str, TermFn]:
        """Registered cached values by name, in registration order: each the function and the names it reads."""
        return MappingProxyType(self._cached)

    @property
    def weights(self) -> Mapping[str, float | torch.Tensor]:
        """Weight overrides by term name, set by ``with_(weights=)``."""
        return MappingProxyType(self._weights)

    @property
    def free(self) -> Mapping[str, torch.Tensor] | None:
        """Free-coordinate masks ``{variable: (N_variable,) or (B, N_variable)}`` in tangent coordinates, or ``None``."""
        return None if self._free is None else MappingProxyType(self._free)

    @property
    def tangent_dim(self) -> int:
        """Length ``N`` of the stacked tangent vector of all variables."""
        return self._tangent_dim

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
        self._require_unsealed(name)
        self._require_new_name(name)
        self._cached[name] = TermFn(fn=fn, reads=self._resolve_reads(name, fn, reads))
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
        self,
        *,
        data: Mapping[str, Any] | None = None,
        weights: Mapping[str, float | torch.Tensor] | None = None,
        free: Mapping[str, torch.Tensor] | None = None,
    ) -> Problem:
        """A new problem with data entries added or replaced, term weights overridden and/or coordinates frozen.

        ``data`` adds entries, or replaces those of the same name; a new name may not be a variable or a
        cached value. ``weights[name] = 0`` switches a term off. ``free[variable]``
        is a bool mask over that variable's tangent coordinates, ``(n,)`` or ``(B, n)``; unlisted variables
        stay free. For a time variable the mask may cover one knot; it is then repeated over time. This
        problem is left as it was. The new problem is sealed: register every term before calling ``with_``.
        """
        problem = self._copy()
        if data is not None:
            for name in data:
                if name not in self._data:
                    self._require_new_name(name)
            problem._data = {**self._data, **data}
        if weights is not None:
            for name in weights:
                if name not in self._terms:
                    raise UnknownNameError("term", name, list(self._terms))
            problem._weights = {**self._weights, **weights}
        if free is not None:
            problem._free = {}
            for name, mask in self._known_variables(free).items():
                span = self._slices[name]
                size = span.stop - span.start
                knots = self._variables[name].shape[1] if name in self._time else 1
                if mask.shape[-1] * knots != size and mask.shape[-1] != size:
                    raise ShapeError(f"free mask of {name!r} has {mask.shape[-1]} entries, expected {size}")
                problem._free[name] = mask.repeat(*([1] * (mask.ndim - 1)), size // mask.shape[-1])
        return problem

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
        return self._clamp(
            {name: self._manifolds[name].retract(value, delta[:, self._slices[name]]) for name, value in x.items()}
        )

    def project(self, x: Values) -> dict[str, torch.Tensor]:
        """Map ambient tensors back onto their manifolds and bounds."""
        return self._clamp({name: self._manifolds[name].project(value) for name, value in x.items()})

    def step_bounds(self, x: Values) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Allowed tangent step ``(lower, upper)``, each ``(B, N)``, or ``None`` without bounds."""
        if not self._bounds:
            return None
        first = next(iter(x.values()))
        lower = first.new_full((first.shape[0], self._tangent_dim), -float("inf"))
        upper = first.new_full((first.shape[0], self._tangent_dim), float("inf"))
        for name, (low, high) in self._bounds.items():
            lower[:, self._slices[name]], upper[:, self._slices[name]] = self._manifolds[name].tangent_bounds(
                x[name], lower=low, upper=high
            )
        return lower, upper

    def slices(self) -> dict[str, slice]:
        """Position of each variable in the stacked tangent vector."""
        return dict(self._slices)

    # ── internals ───────────────────────────────────────────────────────

    def _add(self, name, fn, kind, weight, kernel, jacobian, reads, window) -> None:
        self._require_unsealed(name)
        if name in self._terms:
            raise ArgumentError(f"a term named {name!r} already exists")
        reads = self._resolve_reads(name, fn, reads)
        self._terms[name] = Term(
            name, fn, kind=kind, reads=reads, weight=weight, kernel=kernel, jacobian=jacobian, window=window
        )

    def _copy(self) -> Problem:
        """A sealed problem sharing this one's tensors, with registries of its own."""
        problem = copy.copy(self)
        problem._terms, problem._cached = dict(self._terms), dict(self._cached)
        problem._sealed = True
        return problem

    def _require_unsealed(self, name: str) -> None:
        if self._sealed:
            raise ArgumentError(
                f"cannot register {name!r}: this problem was made by with_ (or by run, solve_constrained or "
                "solve_layer for a callback) and is sealed; register every term on the problem before solving it "
                "or deriving it with with_, and change data, weights or free masks with with_"
            )

    def _require_new_name(self, name: str) -> None:
        if name in self._variables or name in self._data or name in self._cached:
            raise ArgumentError(f"name {name!r} is already used by a variable, data tensor or cached value")

    def _resolve_reads(self, name: str, fn: Callable[..., Any], reads: tuple[str, ...] | None) -> tuple[str, ...]:
        if reads is None:
            reads = tuple(inspect.signature(fn).parameters)
        known = [*self._variables, *self._data, *self._cached]
        for read in reads:
            if read not in known:
                error = UnknownNameError("variable, data or cached value", read, known)
                error.add_note(f"{name!r} reads it; register cached values before the terms that read them")
                raise error
        return tuple(reads)

    def _known_variables(self, mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
        mapping = mapping or {}
        for name in mapping:
            if name not in self._variables:
                raise UnknownNameError("variable", name, list(self._variables))
        return mapping

    def _clamp(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, (lower, upper) in self._bounds.items():
            x[name] = torch.minimum(torch.maximum(x[name], lower), upper)
        return x


__all__ = ["Problem", "Term", "TermFn"]
