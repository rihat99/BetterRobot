"""Gauss-Newton normal equations of a ``Problem``: Jacobian blocks by route, accumulated into one sink.

A term's Jacobian comes from its analytic ``jacobian``, from autodiff by knot colours for a time-local term
(``window=``), or from one batched backward pass shared by every other term. ``linearize`` adds each
block's ``JᵀJ`` and ``Jᵀr`` to a dense, banded or arrowhead sink; ``check_jacobian`` compares each route
with plain autodiff.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, final

import torch

from ..exceptions import ArgumentError

from .evaluation import (
    Evaluation,
    Values,
    active_terms,
    checked_values,
    depends,
    evaluate,
    irls,
    residual_loss,
    weight_of,
    weighted,
    zero_loss,
)
from .linear import ArrowNormal, ArrowSink, BandedNormal, BandLayout, BandSink, DenseSink, Normal

if TYPE_CHECKING:
    from .problem import Problem, Term


@final
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


def _stepped(problem: Problem, x: Values, delta: torch.Tensor) -> dict[str, torch.Tensor]:
    """``x`` stepped along ``delta: (B, N)`` on each variable's manifold, without the bounds clamp, so a derivative
    taken through it is that of the terms alone."""
    slices = problem.slices()
    return {name: problem.manifolds[name].retract(value, delta[:, slices[name]]) for name, value in x.items()}


def linearize(problem: Problem, x: Values | None = None) -> tuple[Normal | BandedNormal | ArrowNormal, torch.Tensor]:
    """Gauss-Newton normal equations and the loss ``(B,)`` at ``x`` (default: the problem's variables)."""
    return _normal_equations(problem, x, active_terms(problem))


def _normal_equations(
    problem: Problem, x: Values | None, terms: list[Term]
) -> tuple[Normal | BandedNormal | ArrowNormal, torch.Tensor]:
    """Normal equations of ``terms`` (registered on ``problem``, or derived from its terms) and their loss."""
    x = {name: value.detach() for name, value in checked_values(problem, x).items()}
    traced = [t for t in terms if t.kind == "scalar" or t.jacobian is None]
    loss = zero_loss(problem)
    batch, size = loss.shape[0], problem.tangent_dim
    layout = _band_layout(problem, terms)
    if layout is None:
        sink = DenseSink(loss, batch, size)
    else:
        sink = (BandSink if layout.shared_columns is None else ArrowSink)(loss, batch, layout)
    g = loss.new_zeros(batch, size)
    with torch.enable_grad() if traced else torch.no_grad():
        if traced:
            delta = loss.new_zeros(batch, size, requires_grad=True)
            evaluation = Evaluation(problem, _stepped(problem, x, delta))
        else:
            delta = None
            evaluation = Evaluation(problem, x)
        dense_rows, scalar_total = [], None
        for term in terms:
            value = weighted(problem, term, evaluation)
            if term.kind == "scalar":
                loss = loss + value.detach()
                scalar_total = value if scalar_total is None else scalar_total + value
                continue
            loss = loss + residual_loss(term, value).detach()
            reweighted = value * irls(term, value)
            if term.jacobian is not None:
                with torch.no_grad():
                    pieces = _analytic_pieces(problem, term, evaluation, value)
            elif term.window is not None:
                pieces = _windowed_pieces(problem, term, reweighted, delta)
            else:
                dense_rows.append(reweighted.flatten(1))
                continue
            rows = reweighted.detach()
            _accumulate(sink, g, pieces, rows.flatten(2) if term.window is not None else rows.flatten(1)[:, None])
        # A scalar term may read only data or cached values that do not depend on the variables.
        if scalar_total is not None and scalar_total.requires_grad:
            g += torch.autograd.grad(
                scalar_total.sum(), delta, retain_graph=True, allow_unused=True, materialize_grads=True
            )[0]
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
    mask = free_mask(problem)
    return (normal if mask is None else normal.masked(mask)), loss


def gradient(
    problem: Problem, x: Values | None = None, *, create_graph: bool = False
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Loss ``(B,)``, its gradient ``(B, N)`` in tangent coordinates (frozen ones zeroed) and the detached term losses."""
    x = {name: value.detach() for name, value in checked_values(problem, x).items()}
    with torch.enable_grad():
        zeros = zero_loss(problem)
        delta = zeros.new_zeros(zeros.shape[0], problem.tangent_dim, requires_grad=True)
        loss, terms, _ = evaluate(problem, _stepped(problem, x, delta))
        (g,) = torch.autograd.grad(loss.sum(), delta, create_graph=create_graph)
    mask = free_mask(problem)
    terms = {name: value.detach() for name, value in terms.items()}
    return loss.detach(), (g if mask is None else g * mask), terms


def free_mask(problem: Problem) -> torch.Tensor | None:
    """Float mask ``(B or 1, N)``: 1 for free tangent coordinates, 0 for frozen ones; ``None`` when all are free."""
    if problem.free is None:
        return None
    first = next(iter(problem.variables.values()))
    rows = max([1] + [mask.shape[0] for mask in problem.free.values() if mask.ndim == 2])
    out = first.new_ones(rows, problem.tangent_dim)
    slices = problem.slices()
    for name, mask in problem.free.items():
        out[:, slices[name]] = mask.to(first.dtype)
    return out


def _band_layout(problem: Problem, terms: list[Term]) -> BandLayout | None:
    """Time layout when every residual term is time-local or reads no time variable.

    Variables without a time axis become the border of an arrowhead matrix.
    """
    residual_terms = [t for t in terms if t.kind == "residual"]
    windows = [t.window for t in residual_terms if t.window is not None]
    if problem.dense or not problem.time or not windows:
        return None
    unwindowed = [t.name for t in residual_terms if t.window is None and depends(problem, t.reads) & set(problem.time)]
    if unwindowed:
        # Python's default filter shows this once per call site, not once per solver step.
        warnings.warn(
            f"time problem uses dense normal equations: residual terms {unwindowed} read a time variable "
            "without window=; declare window= for the banded solve or pass dense=True to silence this",
            stacklevel=2,
        )
        return None
    knots = {problem.variables[name].shape[1] for name in problem.time}
    if len(knots) != 1:
        return None
    spans = tuple((span.start, span.stop, name in problem.time) for name, span in problem.slices().items())
    device = next(iter(problem.variables.values())).device
    return _layout(spans, knots.pop(), max(max(windows) - 1, 1), device)


@functools.lru_cache(maxsize=64)
def _layout(spans: tuple[tuple[int, int, bool], ...], knots: int, group: int, device: torch.device) -> BandLayout:
    """Band layout of ``knots`` knots in blocks of ``group`` knots, built once per problem structure (C-STATE-4).

    ``spans`` gives each variable's tangent columns ``start:stop`` and whether it has a time axis.
    """
    size = spans[-1][1]
    knot_of = torch.zeros(size, dtype=torch.long, device=device)
    local_of = torch.zeros(size, dtype=torch.long, device=device)
    offset = 0
    time_columns, shared_columns = [], []
    for start, stop, timed in spans:
        index = torch.arange(stop - start, device=device)
        if not timed:
            shared_columns.append(start + index)
            continue
        n = (stop - start) // knots
        knot_of[start:stop] = index // n
        local_of[start:stop] = offset + index % n
        offset += n
        time_columns.append(start + index)
    return BandLayout(
        block_of_column=knot_of // group,
        row_of_column=(knot_of % group) * offset + local_of,
        blocks=-(-knots // group),
        block_size=group * offset,
        time_columns=torch.cat(time_columns) if shared_columns else None,
        shared_columns=torch.cat(shared_columns) if shared_columns else None,
    )


def _window_columns(problem: Problem, name: str, rows: int, offset: int, device: torch.device) -> torch.Tensor:
    """Columns ``(rows, n)`` of knot ``t + offset`` of time variable ``name`` for each row ``t``."""
    span = problem.slices()[name]
    n = (span.stop - span.start) // problem.variables[name].shape[1]
    knot = torch.arange(rows, device=device) + offset
    return span.start + knot[:, None] * n + torch.arange(n, device=device)


def _analytic_pieces(
    problem: Problem, term: Term, evaluation: Evaluation, weighted_value: torch.Tensor
) -> list[_Piece]:
    """Pieces of the weighted, reweighted residual from the blocks returned by ``term.jacobian``."""
    blocks = term.jacobian(*evaluation.args(term.reads))
    weight = weight_of(problem, term)
    scale = (weight.sqrt() if isinstance(weight, torch.Tensor) else weight**0.5) * irls(term, weighted_value)
    batch = weighted_value.shape[0]
    windowed = term.window is not None
    if isinstance(scale, torch.Tensor):
        scale = scale.expand_as(weighted_value)
        scale = (scale.flatten(2) if windowed else scale.flatten(1)[:, None]).unsqueeze(-1)  # (B, G, m, 1)
    slices = problem.slices()
    pieces = []
    for name, raw in blocks.items():
        block = raw.detach()
        span = slices[name]
        if windowed and name in problem.time:
            if block.ndim == 4:
                block = block.unsqueeze(3)
            for offset in range(block.shape[3]):
                columns = _window_columns(problem, name, block.shape[1], offset, block.device)
                pieces.append(_Piece(columns, block[:, :, :, offset] * scale))
        elif windowed:
            columns = torch.arange(span.start, span.stop, device=block.device).expand(block.shape[1], -1)
            pieces.append(_Piece(columns, block * scale))
        else:
            columns = torch.arange(span.start, span.stop, device=block.device)[None]
            # Explicit row count: a -1 is ambiguous when the batch is empty.
            rows = weighted_value.shape[1:].numel()
            pieces.append(_Piece(columns, block.reshape(batch, 1, rows, span.stop - span.start) * scale))
    return pieces


def _windowed_pieces(problem: Problem, term: Term, reweighted: torch.Tensor, delta: torch.Tensor) -> list[_Piece]:
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
    (gradients,) = torch.autograd.grad(
        rows_by_knot, delta, seeds.flatten(0, 1), is_grads_batched=True, retain_graph=True
    )
    gradients = gradients.view(window, width, batch, -1)
    pieces = []
    read_variables = depends(problem, term.reads)
    slices = problem.slices()
    shared = [
        torch.arange(s.start, s.stop, device=delta.device)
        for n, s in slices.items()
        if n in read_variables and n not in problem.time
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
            raise ArgumentError(
                f"term {term.name!r}: the autodiff Jacobian w.r.t. variables without a time axis is wrong, "
                "because an operation on that path cannot be differentiated twice. Give the term a `jacobian`."
            )
        pieces.append(_Piece(columns.expand(rows, -1), block))
    for name in sorted(read_variables & set(problem.time)):
        span = slices[name]
        knots = problem.variables[name].shape[1]
        by_knot = gradients[..., span].reshape(window, width, batch, knots, -1)
        for offset in range(window):
            knot = torch.arange(rows, device=delta.device) + offset
            block = by_knot[color, :, :, knot]  # (T_out, m, B, n)
            pieces.append(_Piece(_window_columns(problem, name, rows, offset, delta.device), block.permute(2, 0, 1, 3)))
    return pieces


def check_jacobian(problem: Problem, x: Values | None = None) -> dict[str, float]:
    """Largest absolute difference between each term's own Jacobian route and plain autodiff.

    Checks every residual term that has an analytic ``jacobian`` or a ``window``; the reference
    differentiates the same term row by row with no assumption. Returns ``{term name: error}``
    measured on ``H`` and ``g``, relative to the largest entry of ``H``.
    """
    x, errors = checked_values(problem, x), {}
    for term in active_terms(problem):
        if term.kind != "residual" or (term.jacobian is None and term.window is None):
            continue
        normal_a, _ = _normal_equations(problem, x, [term])
        normal_b, _ = _normal_equations(problem, x, [replace(term, jacobian=None, window=None)])
        H_a = normal_a.H if isinstance(normal_a, Normal) else normal_a.dense()
        H_b = normal_b.H if isinstance(normal_b, Normal) else normal_b.dense()
        scale = max(float(H_b.abs().max()), 1e-300)
        errors[term.name] = max(float((H_a - H_b).abs().max()), float((normal_a.g - normal_b.g).abs().max())) / scale
    return errors


__all__ = ["check_jacobian", "free_mask", "gradient", "linearize"]
