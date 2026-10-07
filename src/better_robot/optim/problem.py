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
from .linear import ArrowNormal, ArrowSink, BandedNormal, BandLayout, BandSink, DenseSink, Normal
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
    window: int | None = None

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
    window: int | None = None  #: time-local term: row ``t`` reads knots ``t .. t + window - 1``.


@dataclass(frozen=True)
class _Piece:
    """Jacobian block of one term, grouped by the columns its rows share.

    ``block[b, g, r, i]`` is the derivative of row ``r`` of group ``g`` w.r.t. column
    ``columns[g, i]``. A time-local term has one group per knot; any other term has one group.
    """

    columns: torch.Tensor  #: ``(G, n)``
    block: torch.Tensor  #: ``(B, G, m, n)``


def _accumulate(
    sink: DenseSink | BandSink | ArrowSink, g: torch.Tensor, pieces: list[_Piece], rows: torch.Tensor
) -> None:
    """Add ``JᵀJ`` of one term to ``sink`` and ``Jᵀr`` (``rows: (B, G, m)``) to ``g: (B, N)`` in place."""
    batch = g.shape[0]
    for piece in pieces:
        gradient = (piece.block.mT @ rows.unsqueeze(-1)).squeeze(-1)  # (B, G, n)
        g.scatter_add_(1, piece.columns.reshape(1, -1).expand(batch, -1), gradient.flatten(1))
    for first_index, first in enumerate(pieces):
        for second in pieces[first_index:]:
            product = first.block.mT @ second.block  # (B, G, na, nb)
            sink.add(first.columns, second.columns, product)
            if first is not second:
                sink.add(second.columns, first.columns, product.mT)


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

    Register terms with ``residual`` (least squares) or ``scalar`` (any ``(B,)`` loss) and shared
    intermediates with ``cached``; then minimize with ``solve`` or ``run``.

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
        window: int | None = None,
    ) -> None:
        """Add a least-squares term ``fn(...) -> (B, ..., k)``.

        Its loss is ``½ · weight · Σ r²``; with a ``kernel`` the last axis is one robust
        group and the loss is ``Σ kernel.rho(weight · ‖r‖²)`` over the other axes.
        A tensor ``weight`` has the shape of the output, with or without the batch axis.
        ``jac`` is optional: without it the Jacobian comes from autodiff.

        ``window=k`` declares a time-local term: the output is ``(B, T - k + 1, m)`` and row
        ``t`` depends only on knots ``t .. t + k - 1`` of the time variables (``k=1`` for a
        per-knot term, 2 for velocities, 3 for accelerations). Its autodiff Jacobian then costs
        ``k · m`` backward passes instead of one per row, and an analytic ``jac`` returns blocks
        ``(B, T - k + 1, m, k, n)`` per time variable and ``(B, T - k + 1, m, n)`` per variable
        without a time axis. The declaration is trusted: a row that reads outside its window
        gives a wrong Jacobian. ``check_jacobian`` tests it.
        """
        jac = jac if jac is not None else getattr(fn, "jac", None)
        window = window if window is not None else getattr(fn, "window", None)
        self._add(name, fn, "residual", weight, kernel, jac, reads, window)

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
        self._add(name, fn, "scalar", weight, None, None, reads, None)

    def with_(self, *, weights: Mapping[str, float] | None = None, free: Mapping[str, torch.Tensor] | None = None):
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

    def linearize(self, x: Values | None = None) -> tuple[Normal | BandedNormal | ArrowNormal, torch.Tensor]:
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
        layout = self._band_layout(terms)
        if layout is None:
            sink = DenseSink(loss, batch, size)
        else:
            sink = (BandSink if layout.shared_columns is None else ArrowSink)(loss, batch, layout)
        g = loss.new_zeros(batch, size)
        with torch.enable_grad() if traced else torch.no_grad():
            if traced:
                delta = loss.new_zeros(batch, size, requires_grad=True)
                evaluation = Evaluation(self, self._retract(x, delta))
            else:
                delta = None
                evaluation = Evaluation(self, x)
            dense_rows, scalar_total = [], None
            for term in terms:
                value = self._weighted(term, evaluation)
                if term.kind == "scalar":
                    loss = loss + value.detach()
                    scalar_total = value if scalar_total is None else scalar_total + value
                    continue
                loss = loss + self._residual_loss(term, value).detach()
                reweighted = value * self._irls(term, value)
                if term.jac is not None:
                    with torch.no_grad():
                        pieces = self._analytic_pieces(term, evaluation, value)
                elif term.window is not None:
                    pieces = self._windowed_pieces(term, reweighted, delta)
                else:
                    dense_rows.append(reweighted.flatten(1))
                    continue
                rows = reweighted.detach()
                _accumulate(sink, g, pieces, rows.flatten(2) if term.window is not None else rows.flatten(1)[:, None])
            if scalar_total is not None:
                g += torch.autograd.grad(scalar_total.sum(), delta, retain_graph=True)[0]
            if dense_rows:
                stacked = torch.cat(dense_rows, dim=1)  # (B, M)
                # Batch elements are independent, so row i of every element comes from one backward pass.
                seeds = torch.eye(stacked.shape[1], dtype=stacked.dtype, device=stacked.device)
                seeds = seeds.unsqueeze(1).expand(-1, batch, -1)
                (jacobian,) = torch.autograd.grad(stacked, delta, seeds, is_grads_batched=True)
                # Under a time layout the only terms on this route read variables without a time axis.
                columns = torch.arange(size, device=g.device) if layout is None else layout.shared_columns
                block = jacobian.permute(1, 0, 2)[:, None, :, columns]
                _accumulate(sink, g, [_Piece(columns[None], block)], stacked.detach()[:, None])
        normal = sink.finish(g)
        mask = self._free_mask()
        return (normal if mask is None else normal.masked(mask)), loss

    def gradient(self, x: Values | None = None, *, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Loss ``(B,)`` and its gradient ``(B, N)`` in tangent coordinates (frozen coordinates zeroed).

        With ``create_graph=True`` the gradient stays differentiable w.r.t. data tensors.
        """
        x = {name: value.detach() for name, value in (x if x is not None else self.variables).items()}
        with torch.enable_grad():
            delta = self._zeros().new_zeros(self._batch(), self.tangent_dim, requires_grad=True)
            loss = self.loss(self._retract(x, delta))
            (g,) = torch.autograd.grad(loss.sum(), delta, create_graph=create_graph)
        mask = self._free_mask()
        return loss.detach(), (g if mask is None else g * mask)

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

    def _add(self, name, fn, kind, weight, kernel, jac, reads, window) -> None:
        if name in self.terms:
            raise ValueError(f"a term named {name!r} already exists")
        reads = self._resolve_reads(name, fn, reads)
        self.terms[name] = Term(name, fn, kind, reads, weight, kernel, jac, window)

    def _depends(self, reads: tuple[str, ...]) -> set[str]:
        """Variables that ``reads`` depend on, following cached values."""
        out: set[str] = set()
        for read in reads:
            if read in self.variables:
                out.add(read)
            elif read in self._cached:
                out |= self._depends(self._cached[read][1])
        return out

    def _require_new_name(self, name: str) -> None:
        if name in self.variables or name in self.data or name in self._cached:
            raise ValueError(f"name {name!r} is already used by a variable, data tensor or cached value")

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
        if isinstance(weight, torch.Tensor) and weight.ndim not in (0, value.ndim - 1, value.ndim):
            raise ValueError(
                f"weight of {term.name!r} has shape {tuple(weight.shape)} but the term returns "
                f"{tuple(value.shape)}; give the weight that shape, with or without the batch axis"
            )
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

    def _band_layout(self, terms: list[Term]) -> BandLayout | None:
        """Time layout when every residual term is time-local or reads no time variable.

        Variables without a time axis become the border of an arrowhead matrix.
        """
        residuals = [t for t in terms if t.kind == "residual"]
        windows = [t.window for t in residuals if t.window is not None]
        if self.dense or not self.time or not windows:
            return None
        if any(t.window is None and self._depends(t.reads) & set(self.time) for t in residuals):
            return None
        knots = {self.variables[name].shape[1] for name in self.time}
        if len(knots) != 1:
            return None
        return self._layout(knots.pop(), max(max(windows) - 1, 1))

    def _layout(self, knots: int, group: int) -> BandLayout:
        key = (knots, group)
        if key not in self._layouts:
            device = next(iter(self.variables.values())).device
            knot_of = torch.zeros(self.tangent_dim, dtype=torch.long, device=device)
            local_of = torch.zeros(self.tangent_dim, dtype=torch.long, device=device)
            offset = 0
            time_columns, shared_columns = [], []
            for name in self.variables:
                span = self._slices[name]
                index = torch.arange(span.stop - span.start, device=device)
                if name not in self.time:
                    shared_columns.append(span.start + index)
                    continue
                n = (span.stop - span.start) // knots
                knot_of[span] = index // n
                local_of[span] = offset + index % n
                offset += n
                time_columns.append(span.start + index)
            self._layouts[key] = BandLayout(
                block_of_column=knot_of // group,
                row_of_column=(knot_of % group) * offset + local_of,
                blocks=-(-knots // group),
                block_size=group * offset,
                time_columns=torch.cat(time_columns) if shared_columns else None,
                shared_columns=torch.cat(shared_columns) if shared_columns else None,
            )
        return self._layouts[key]

    def _window_columns(self, name: str, rows: int, offset: int, device) -> torch.Tensor:
        """Columns ``(rows, n)`` of knot ``t + offset`` of time variable ``name`` for each row ``t``."""
        span = self._slices[name]
        n = (span.stop - span.start) // self.variables[name].shape[1]
        knot = torch.arange(rows, device=device) + offset
        return span.start + knot[:, None] * n + torch.arange(n, device=device)

    def _analytic_pieces(self, term: Term, evaluation: Evaluation, weighted: torch.Tensor) -> list[_Piece]:
        """Pieces of the weighted, reweighted residual from the blocks returned by ``term.jac``."""
        blocks = term.jac(*evaluation.args(term.reads))
        weight = self.weights.get(term.name, term.weight)
        scale = (weight.sqrt() if isinstance(weight, torch.Tensor) else weight**0.5) * self._irls(term, weighted)
        batch = weighted.shape[0]
        windowed = term.window is not None
        if isinstance(scale, torch.Tensor):
            scale = scale.expand_as(weighted)
            scale = (scale.flatten(2) if windowed else scale.flatten(1)[:, None]).unsqueeze(-1)  # (B, G, m, 1)
        pieces = []
        for name, raw in blocks.items():
            block = raw.detach()
            span = self._slices[name]
            if windowed and name in self.time:
                if block.ndim == 4:
                    block = block.unsqueeze(3)
                for offset in range(block.shape[3]):
                    columns = self._window_columns(name, block.shape[1], offset, block.device)
                    pieces.append(_Piece(columns, block[:, :, :, offset] * scale))
            elif windowed:
                columns = torch.arange(span.start, span.stop, device=block.device).expand(block.shape[1], -1)
                pieces.append(_Piece(columns, block * scale))
            else:
                columns = torch.arange(span.start, span.stop, device=block.device)[None]
                pieces.append(_Piece(columns, block.reshape(batch, 1, -1, span.stop - span.start) * scale))
        return pieces

    def _windowed_pieces(self, term: Term, reweighted: torch.Tensor, delta: torch.Tensor) -> list[_Piece]:
        """Autodiff pieces of a time-local term: rows whose windows do not overlap share a backward pass.

        Columns of variables without a time axis are read by every row, so they come from one
        forward-mode product per column, computed as the derivative of a backward pass.
        """
        rows_by_knot = reweighted.flatten(2)  # (B, T_out, m)
        batch, rows, width = rows_by_knot.shape
        window = term.window
        color = torch.arange(rows, device=delta.device) % window
        seeds = rows_by_knot.new_zeros(window, width, batch, rows, width)
        for c in range(window):
            for j in range(width):
                seeds[c, j, :, color == c, j] = 1.0
        (gradient,) = torch.autograd.grad(
            rows_by_knot, delta, seeds.flatten(0, 1), is_grads_batched=True, retain_graph=True
        )
        gradient = gradient.view(window, width, batch, -1)
        pieces = []
        depends = self._depends(term.reads)
        shared = [
            torch.arange(s.start, s.stop, device=delta.device)
            for n, s in self._slices.items()
            if n in depends and n not in self.time
        ]
        if shared:
            columns = torch.cat(shared)
            probe = torch.zeros_like(rows_by_knot, requires_grad=True)
            (pulled,) = torch.autograd.grad(rows_by_knot, delta, probe, create_graph=True)
            eye = torch.eye(columns.numel(), dtype=pulled.dtype, device=pulled.device)
            (block,) = torch.autograd.grad(
                pulled[:, columns], probe, eye[:, None].expand(-1, batch, -1), is_grads_batched=True, retain_graph=True
            )
            block = block.permute(1, 2, 3, 0)  # (B, T_out, m, ns)
            # An op whose backward is not itself differentiable (a custom Function, a compiled kernel)
            # silently drops out of the product above, so compare with one ordinary backward pass.
            weights = rows_by_knot.detach()
            (plain,) = torch.autograd.grad(rows_by_knot, delta, weights, retain_graph=True)
            expected = plain[:, columns]
            got = (block * weights.unsqueeze(-1)).sum((1, 2))
            if not torch.allclose(got, expected, rtol=1e-3, atol=1e-6 * float(expected.abs().max()) + 1e-30):
                raise ValueError(
                    f"term {term.name!r}: the autodiff Jacobian w.r.t. variables without a time axis is wrong, "
                    "because an operation on that path cannot be differentiated twice. Give the term a `jac`."
                )
            pieces.append(_Piece(columns.expand(rows, -1), block))
        for name in sorted(depends & set(self.time)):
            span = self._slices[name]
            knots = self.variables[name].shape[1]
            by_knot = gradient[..., span].reshape(window, width, batch, knots, -1)
            for offset in range(window):
                knot = torch.arange(rows, device=delta.device) + offset
                block = by_knot[color, :, :, knot]  # (T_out, m, B, n)
                pieces.append(_Piece(self._window_columns(name, rows, offset, delta.device), block.permute(2, 0, 1, 3)))
        return pieces

    def _retract(self, x: Values, delta: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self.manifolds[name].retract(value, delta[:, self._slices[name]]) for name, value in x.items()}

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
    """Largest absolute difference between each term's own Jacobian route and plain autodiff.

    Checks every residual term that has an analytic ``jac`` or a ``window``; the reference
    differentiates the same term row by row with no assumption. Returns ``{term name: error}``
    measured on ``H`` and ``g``, relative to the largest entry of ``H``.
    """
    errors = {}
    for term in problem._active():
        if term.kind != "residual" or (term.jac is None and term.window is None):
            continue
        only = {name: 0.0 for name in problem.terms if name != term.name}
        analytic = problem.with_(weights=only)
        autodiff = copy.copy(analytic)
        autodiff.terms = {**problem.terms, term.name: Term(**{**term.__dict__, "jac": None, "window": None})}
        normal_a, _ = analytic.linearize(x)
        normal_b, _ = autodiff.linearize(x)
        H_a = normal_a.H if isinstance(normal_a, Normal) else normal_a.dense()
        H_b = normal_b.H if isinstance(normal_b, Normal) else normal_b.dense()
        scale = max(float(H_b.abs().max()), 1e-300)
        errors[term.name] = max(float((H_a - H_b).abs().max()), float((normal_a.g - normal_b.g).abs().max())) / scale
    return errors


__all__ = ["Evaluation", "Problem", "Term", "TermFn", "check_jacobian"]
