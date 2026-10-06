"""Normal equations of a batch of independent least-squares problems."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Normal:
    """Gauss-Newton model ``loss(delta) ≈ loss + g·delta + ½ delta·H·delta`` per batch element."""

    H: torch.Tensor  #: ``(B, N, N)`` symmetric positive semi-definite.
    g: torch.Tensor  #: ``(B, N)`` gradient of the loss in tangent coordinates.
    lower: torch.Tensor | None = None  #: ``(B, N)`` most negative allowed step, or ``None``.
    upper: torch.Tensor | None = None  #: ``(B, N)`` most positive allowed step, or ``None``.

    def restrict(self, lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-9) -> Normal:
        """Box-constrain the step to ``[lower, upper]`` (projected Newton, one active-set pass).

        A coordinate that sits on a bound and whose gradient pushes it outward is frozen: its
        gradient entry becomes 0 and its row and column of ``H`` become the identity. The step
        of the other coordinates is clamped to the box by :meth:`solve`.
        """
        blocked = ((lower >= -tolerance) & (self.g > 0)) | ((upper <= tolerance) & (self.g < 0))
        out = self.masked(~blocked)
        return Normal(H=out.H, g=out.g, lower=lower, upper=upper)

    def masked(self, keep: torch.Tensor) -> Normal:
        """Freeze columns where ``keep: (B or 1, N)`` is 0: zero gradient, identity row and column."""
        keep = keep.expand(self.g.shape[0], -1).to(self.g.dtype)
        H = self.H * keep.unsqueeze(-1) * keep.unsqueeze(-2) + torch.diag_embed(1.0 - keep)
        return Normal(H=H, g=self.g * keep, lower=self.lower, upper=self.upper)

    def solve(
        self, damping: torch.Tensor | float = 0.0, *, floor: float = 1e-9, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Solve ``(H + damping · diag(H)) delta = -g``. Returns ``delta: (B, N)``.

        Columns are scaled to unit diagonal and the factorization runs in float64, because a
        float32 Cholesky fails on the rank-deficient ``H`` of redundant robots. A batch element
        whose factorization still fails gets ``delta = 0`` (a rejected step for LM).
        ``g`` replaces the stored gradient; the result is differentiable w.r.t. it.
        """
        H, g = self.H.double(), (self.g if g is None else g).double()
        diagonal = H.diagonal(dim1=-2, dim2=-1)
        scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
        scaled = H * scale.unsqueeze(-1) * scale.unsqueeze(-2)
        shift = torch.as_tensor(damping, dtype=H.dtype, device=H.device).expand(H.shape[0]) + floor
        scaled = scaled + torch.diag_embed(shift.unsqueeze(-1).expand_as(diagonal))
        factor, info = torch.linalg.cholesky_ex(scaled)
        delta = torch.cholesky_solve(-(g * scale).unsqueeze(-1), factor).squeeze(-1) * scale
        ok = (info == 0).unsqueeze(-1) & torch.isfinite(delta).all(dim=-1, keepdim=True)
        delta = torch.where(ok, delta, torch.zeros_like(delta)).to(self.g.dtype)
        if self.lower is not None:
            delta = torch.minimum(torch.maximum(delta, self.lower), self.upper)
        return delta

    def predicted_decrease(self, delta: torch.Tensor) -> torch.Tensor:
        """Loss decrease the quadratic model predicts for ``delta: (B, N)``. Returns ``(B,)``."""
        curvature = (delta.unsqueeze(-2) @ self.H @ delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return -(self.g * delta).sum(-1) - 0.5 * curvature


@dataclass(frozen=True)
class BandLayout:
    """Where each tangent column sits in a block-tridiagonal arrangement of the time axis.

    Knots are grouped into super-blocks of ``group`` consecutive knots, so a term whose window
    spans at most ``group + 1`` knots only couples neighbouring super-blocks.
    """

    block_of_column: torch.Tensor  #: ``(N,)`` super-block index of each column.
    row_of_column: torch.Tensor  #: ``(N,)`` position of each column inside its super-block.
    blocks: int
    block_size: int
    time_columns: torch.Tensor | None = None  #: ``(Nt,)`` columns with a time axis; ``None`` means all.
    shared_columns: torch.Tensor | None = None  #: ``(Ns,)`` columns of variables shared over time.

    @property
    def position(self) -> torch.Tensor:
        """``(Nt,)`` index of each time column in the banded ordering."""
        position = self.block_of_column * self.block_size + self.row_of_column
        return position if self.time_columns is None else position[self.time_columns]


@dataclass(frozen=True)
class BandedNormal:
    """Normal equations of a trajectory problem stored as a block-tridiagonal matrix.

    Same interface as :class:`Normal`; ``g`` and every ``delta`` use the problem's column order.
    Padding rows (when the horizon is not a multiple of the group size) carry an identity diagonal.
    """

    diagonal: torch.Tensor  #: ``(B, K, S, S)`` diagonal super-blocks.
    upper: torch.Tensor  #: ``(B, K - 1, S, S)`` blocks coupling super-block ``i`` to ``i + 1``.
    g: torch.Tensor  #: ``(B, N)``
    layout: BandLayout
    lower: torch.Tensor | None = None
    upper_bound: torch.Tensor | None = None

    def _to_band(self, vector: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        """``(B, N, ...)`` in problem order to ``(B, K, S, ...)`` in banded order."""
        batch, rest = vector.shape[0], vector.shape[2:]
        out = vector.new_full((batch, self.layout.blocks * self.layout.block_size, *rest), fill)
        out[:, self.layout.position] = vector
        return out.view(batch, self.layout.blocks, self.layout.block_size, *rest)

    def _from_band(self, banded: torch.Tensor) -> torch.Tensor:
        return banded.flatten(1)[:, self.layout.position]

    def masked(self, keep: torch.Tensor) -> BandedNormal:
        """Freeze columns where ``keep: (B or 1, N)`` is 0: zero gradient, identity row and column."""
        keep = keep.expand(self.g.shape[0], -1).to(self.g.dtype)
        k = self._to_band(keep, fill=1.0)
        diagonal = self.diagonal * k.unsqueeze(-1) * k.unsqueeze(-2) + torch.diag_embed(1.0 - k)
        upper = self.upper * k[:, :-1].unsqueeze(-1) * k[:, 1:].unsqueeze(-2)
        return BandedNormal(diagonal, upper, self.g * keep, self.layout, self.lower, self.upper_bound)

    def restrict(self, lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-9) -> BandedNormal:
        """Box-constrain the step; see :meth:`Normal.restrict`."""
        blocked = ((lower >= -tolerance) & (self.g > 0)) | ((upper <= tolerance) & (self.g < 0))
        out = self.masked(~blocked)
        return BandedNormal(out.diagonal, out.upper, out.g, self.layout, lower, upper)

    def solve(
        self, damping: torch.Tensor | float = 0.0, *, floor: float = 1e-9, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Solve ``(H + damping · diag(H)) delta = -g`` by block-tridiagonal Cholesky in float64.

        ``g`` replaces the stored gradient; the result is differentiable w.r.t. it.
        """
        g = self.g if g is None else g
        scale, factorization, failed = self._factor(damping, floor)
        rhs = -self._to_band(g.double()) * scale
        delta = self._from_band(self._substitute(factorization, rhs.unsqueeze(-1)).squeeze(-1) * scale)
        ok = ~failed.unsqueeze(-1) & torch.isfinite(delta).all(dim=-1, keepdim=True)
        delta = torch.where(ok, delta, torch.zeros_like(delta)).to(self.g.dtype)
        if self.lower is not None:
            delta = torch.minimum(torch.maximum(delta, self.lower), self.upper_bound)
        return delta

    def _factor(self, damping: torch.Tensor | float, floor: float):
        """Block Cholesky of the scaled, damped matrix. Returns ``(scale (B, K, S), factorization, failed (B,))``."""
        D, U = self.diagonal.double(), self.upper.double()
        batch, blocks, size, _ = D.shape
        diagonal = D.diagonal(dim1=-2, dim2=-1)
        scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
        shift = torch.as_tensor(damping, dtype=D.dtype, device=D.device).expand(batch) + floor
        D = D * scale.unsqueeze(-1) * scale.unsqueeze(-2) + torch.diag_embed(
            shift.view(batch, 1, 1).expand(batch, blocks, size)
        )
        U = U * scale[:, :-1].unsqueeze(-1) * scale[:, 1:].unsqueeze(-2)
        factors, links = [], []
        failed = torch.zeros(batch, dtype=torch.bool, device=D.device)
        schur = D[:, 0]
        for index in range(blocks):
            factor, info = torch.linalg.cholesky_ex(schur)
            failed |= info != 0
            factors.append(factor)
            if index + 1 < blocks:
                links.append(torch.linalg.solve_triangular(factor, U[:, index], upper=False))
                schur = D[:, index + 1] - links[-1].mT @ links[-1]
        return scale, (factors, links), failed

    @staticmethod
    def _substitute(factorization, rhs: torch.Tensor) -> torch.Tensor:
        """Solve the factored system for ``rhs: (B, K, S, R)``. Returns ``(B, K, S, R)``."""
        factors, links = factorization
        blocks = len(factors)
        forward = []
        for index in range(blocks):
            right = rhs[:, index] if index == 0 else rhs[:, index] - links[index - 1].mT @ forward[-1]
            forward.append(torch.linalg.solve_triangular(factors[index], right, upper=False))
        solution = [None] * blocks
        for index in reversed(range(blocks)):
            right = forward[index] if index + 1 == blocks else forward[index] - links[index] @ solution[index + 1]
            solution[index] = torch.linalg.solve_triangular(factors[index].mT, right, upper=True)
        return torch.stack(solution, dim=1)

    def predicted_decrease(self, delta: torch.Tensor) -> torch.Tensor:
        """Loss decrease the quadratic model predicts for ``delta: (B, N)``. Returns ``(B,)``."""
        d = self._to_band(delta)
        product = (self.diagonal @ d.unsqueeze(-1)).squeeze(-1)
        cross = (self.upper @ d[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, K - 1, S)
        curvature = (d * product).sum((-1, -2)) + 2.0 * (d[:, :-1] * cross).sum((-1, -2))
        return -(self.g * delta).sum(-1) - 0.5 * curvature

    def dense(self) -> torch.Tensor:
        """The full ``(B, N, N)`` matrix in problem order (for tests and small problems)."""
        batch, blocks, size, _ = self.diagonal.shape
        full = self.diagonal.new_zeros(batch, blocks * size, blocks * size)
        for index in range(blocks):
            rows = slice(index * size, (index + 1) * size)
            full[:, rows, rows] = self.diagonal[:, index]
            if index + 1 < blocks:
                cols = slice((index + 1) * size, (index + 2) * size)
                full[:, rows, cols] = self.upper[:, index]
                full[:, cols, rows] = self.upper[:, index].mT
        position = self.layout.position
        return full[:, position[:, None], position[None, :]]


@dataclass(frozen=True)
class ArrowNormal:
    """Block-tridiagonal normal equations bordered by variables shared over time (an arrowhead matrix).

    Same interface as :class:`Normal`. The time part is eliminated by the banded factorization
    and the shared part by a dense Schur complement, so cost stays linear in the horizon.
    """

    band: BandedNormal  #: time columns; its vectors are ``(B, Nt)`` in ``layout.time_columns`` order.
    border: torch.Tensor  #: ``(B, K, S, Ns)`` coupling of banded rows to shared columns.
    corner: torch.Tensor  #: ``(B, Ns, Ns)``
    g: torch.Tensor  #: ``(B, N)``
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None

    def masked(self, keep: torch.Tensor) -> ArrowNormal:
        """Freeze columns where ``keep: (B or 1, N)`` is 0: zero gradient, identity row and column."""
        keep = keep.expand(self.g.shape[0], -1).to(self.g.dtype)
        layout = self.band.layout
        keep_time, keep_shared = keep[:, layout.time_columns], keep[:, layout.shared_columns]
        border = self.border * self.band._to_band(keep_time, fill=1.0).unsqueeze(-1) * keep_shared[:, None, None]
        corner = self.corner * keep_shared.unsqueeze(-1) * keep_shared.unsqueeze(-2) + torch.diag_embed(
            1.0 - keep_shared
        )
        return ArrowNormal(self.band.masked(keep_time), border, corner, self.g * keep, self.lower, self.upper)

    def restrict(self, lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-9) -> ArrowNormal:
        """Box-constrain the step; see :meth:`Normal.restrict`."""
        blocked = ((lower >= -tolerance) & (self.g > 0)) | ((upper <= tolerance) & (self.g < 0))
        out = self.masked(~blocked)
        return ArrowNormal(out.band, out.border, out.corner, out.g, lower, upper)

    def solve(
        self, damping: torch.Tensor | float = 0.0, *, floor: float = 1e-9, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Solve ``(H + damping · diag(H)) delta = -g`` in float64. Returns ``delta: (B, N)``.

        ``g`` replaces the stored gradient; the result is differentiable w.r.t. it.
        """
        g = (self.g if g is None else g).double()
        layout = self.band.layout
        batch = g.shape[0]
        scale, factorization, failed = self.band._factor(damping, floor)
        C = self.corner.double()
        diagonal = C.diagonal(dim1=-2, dim2=-1)
        scale_shared = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), torch.ones_like(diagonal))
        shift = torch.as_tensor(damping, dtype=C.dtype, device=C.device).expand(batch) + floor
        C = C * scale_shared.unsqueeze(-1) * scale_shared.unsqueeze(-2) + torch.diag_embed(
            shift.unsqueeze(-1).expand_as(diagonal)
        )
        border = self.border.double() * scale.unsqueeze(-1) * scale_shared[:, None, None]
        rhs_time = -self.band._to_band(g[:, layout.time_columns]) * scale
        rhs_shared = -g[:, layout.shared_columns] * scale_shared
        solved = self.band._substitute(factorization, torch.cat([border, rhs_time.unsqueeze(-1)], dim=-1))
        flat_border, solved = border.flatten(1, 2), solved.flatten(1, 2)  # (B, K·S, Ns), (B, K·S, Ns + 1)
        schur = C - flat_border.mT @ solved[..., :-1]
        factor, info = torch.linalg.cholesky_ex(schur)
        right = rhs_shared.unsqueeze(-1) - flat_border.mT @ solved[..., -1:]
        shared = torch.cholesky_solve(right, factor)  # (B, Ns, 1)
        time = (solved[..., -1:] - solved[..., :-1] @ shared).view_as(rhs_time) * scale
        delta = g.new_zeros(g.shape)
        delta = delta.index_copy(1, layout.time_columns, self.band._from_band(time))
        delta = delta.index_copy(1, layout.shared_columns, shared.squeeze(-1) * scale_shared)
        ok = ~(failed | (info != 0)).unsqueeze(-1) & torch.isfinite(delta).all(dim=-1, keepdim=True)
        delta = torch.where(ok, delta, torch.zeros_like(delta)).to(self.g.dtype)
        if self.lower is not None:
            delta = torch.minimum(torch.maximum(delta, self.lower), self.upper)
        return delta

    def predicted_decrease(self, delta: torch.Tensor) -> torch.Tensor:
        """Loss decrease the quadratic model predicts for ``delta: (B, N)``. Returns ``(B,)``."""
        layout = self.band.layout
        time, shared = delta[:, layout.time_columns], delta[:, layout.shared_columns]
        cross = (self.band._to_band(time) * (self.border @ shared[:, None, :, None]).squeeze(-1)).sum((-1, -2))
        corner = (shared.unsqueeze(-2) @ self.corner @ shared.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        gradient = (self.g[:, layout.shared_columns] * shared).sum(-1)
        return self.band.predicted_decrease(time) - gradient - cross - 0.5 * corner

    def dense(self) -> torch.Tensor:
        """The full ``(B, N, N)`` matrix in problem order (for tests and small problems)."""
        layout = self.band.layout
        time, shared = layout.time_columns, layout.shared_columns
        full = self.g.new_zeros(*self.g.shape, self.g.shape[1])
        full[:, time[:, None], time[None, :]] = self.band.dense()
        border = self.border.flatten(1, 2)[:, layout.position]  # (B, Nt, Ns)
        full[:, time[:, None], shared[None, :]] = border
        full[:, shared[:, None], time[None, :]] = border.mT
        full[:, shared[:, None], shared[None, :]] = self.corner
        return full


class DenseSink:
    """Accumulates Jacobian products into a dense ``H: (B, N, N)``."""

    def __init__(self, reference: torch.Tensor, batch: int, size: int) -> None:
        self.H = reference.new_zeros(batch, size, size)
        self.size = size

    def add(self, columns_a: torch.Tensor, columns_b: torch.Tensor, values: torch.Tensor) -> None:
        """Add ``values: (B, G, na, nb)`` where group ``g`` touches ``columns_a[g] × columns_b[g]``."""
        if columns_a.shape[0] == 1:
            self.H[:, columns_a[0][:, None], columns_b[0][None, :]] += values[:, 0]
            return
        index = (columns_a[:, :, None] * self.size + columns_b[:, None, :]).reshape(1, -1)
        self.H.view(values.shape[0], -1).scatter_add_(1, index.expand(values.shape[0], -1), values.flatten(1))

    def finish(self, g: torch.Tensor) -> Normal:
        return Normal(H=self.H, g=g)


class BandSink:
    """Accumulates Jacobian products into block-tridiagonal storage."""

    def __init__(self, reference: torch.Tensor, batch: int, layout: BandLayout) -> None:
        self.layout = layout
        size = layout.block_size
        # One spare slot at the end swallows the lower-triangle entries, which the symmetric call stores.
        self.diagonal = reference.new_zeros(batch, layout.blocks * size * size + 1)
        self.upper = reference.new_zeros(batch, max(layout.blocks - 1, 0) * size * size + 1)

    def add(self, columns_a: torch.Tensor, columns_b: torch.Tensor, values: torch.Tensor) -> None:
        """Add ``values: (B, G, na, nb)`` where group ``g`` touches ``columns_a[g] × columns_b[g]``."""
        layout, size = self.layout, self.layout.block_size
        block_a = layout.block_of_column[columns_a][:, :, None]
        block_b = layout.block_of_column[columns_b][:, None, :]
        inner = layout.row_of_column[columns_a][:, :, None] * size + layout.row_of_column[columns_b][:, None, :]
        index = block_a * size * size + inner
        batch = values.shape[0]
        flat = values.flatten(1)
        same = torch.where(block_a == block_b, index, self.diagonal.shape[1] - 1).reshape(1, -1)
        self.diagonal.scatter_add_(1, same.expand(batch, -1), flat)
        above = torch.where(block_b == block_a + 1, index, self.upper.shape[1] - 1).reshape(1, -1)
        self.upper.scatter_add_(1, above.expand(batch, -1), flat)

    def finish(self, g: torch.Tensor) -> BandedNormal:
        layout, size, batch = self.layout, self.layout.block_size, g.shape[0]
        diagonal = self.diagonal[:, :-1].reshape(batch, layout.blocks, size, size)
        used = g.new_zeros(layout.blocks * size)
        used[layout.position] = 1.0
        diagonal = diagonal + torch.diag_embed(1.0 - used.view(layout.blocks, size))
        upper = self.upper[:, :-1].reshape(batch, max(layout.blocks - 1, 0), size, size)
        return BandedNormal(diagonal=diagonal, upper=upper, g=g, layout=layout)


class ArrowSink:
    """Accumulates Jacobian products into a band, a border and a dense corner.

    Every call must pass columns that are all time columns or all shared columns.
    """

    def __init__(self, reference: torch.Tensor, batch: int, layout: BandLayout) -> None:
        self.layout = layout
        self.band = BandSink(reference, batch, layout)
        shared = layout.shared_columns
        self.count = shared.numel()
        self.shared_index = torch.full_like(layout.block_of_column, -1)
        self.shared_index[shared] = torch.arange(self.count, device=shared.device)
        self.border = reference.new_zeros(batch, layout.blocks * layout.block_size * self.count)
        self.corner = reference.new_zeros(batch, self.count * self.count)

    def add(self, columns_a: torch.Tensor, columns_b: torch.Tensor, values: torch.Tensor) -> None:
        """Add ``values: (B, G, na, nb)`` where group ``g`` touches ``columns_a[g] × columns_b[g]``."""
        shared_a, shared_b = self.shared_index[columns_a], self.shared_index[columns_b]
        a_is_shared, b_is_shared = bool(shared_a[0, 0] >= 0), bool(shared_b[0, 0] >= 0)
        batch = values.shape[0]
        if not a_is_shared and not b_is_shared:
            self.band.add(columns_a, columns_b, values)
        elif a_is_shared and b_is_shared:
            index = (shared_a[:, :, None] * self.count + shared_b[:, None, :]).reshape(1, -1)
            self.corner.scatter_add_(1, index.expand(batch, -1), values.flatten(1))
        elif not a_is_shared:  # the mirrored call (shared × time) carries the same numbers
            layout = self.layout
            position = layout.block_of_column[columns_a] * layout.block_size + layout.row_of_column[columns_a]
            index = (position[:, :, None] * self.count + shared_b[:, None, :]).reshape(1, -1)
            self.border.scatter_add_(1, index.expand(batch, -1), values.flatten(1))

    def finish(self, g: torch.Tensor) -> ArrowNormal:
        layout, batch = self.layout, g.shape[0]
        band = self.band.finish(g[:, layout.time_columns])
        border = self.border.view(batch, layout.blocks, layout.block_size, self.count)
        return ArrowNormal(band=band, border=border, corner=self.corner.view(batch, self.count, self.count), g=g)


__all__ = ["ArrowNormal", "ArrowSink", "BandLayout", "BandSink", "BandedNormal", "DenseSink", "Normal"]
