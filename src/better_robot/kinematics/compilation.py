"""`small_matmul`: batched products of small matrices in the form each execution mode runs fastest.

Inductor leaves batched products of small matrices (``3x4 @ 4x4``, ``3x3 @ 3x1``, ``6x6 @ 6x1``) as extern batched
matrix products, about half of a compiled batch-4096 call on CPU. Written as a multiply and a sum under
``torch.compile``, they fuse with their neighbours; in eager mode that form is slower, so ``@`` stays.
"""

from __future__ import annotations

import torch


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for batches of small matrices, ``(..., m, k) @ (..., k, n) -> (..., m, n)``.

    Under ``torch.compile`` a multiply and a sum over ``k``, which inductor fuses into the neighbouring kernels;
    eagerly ``@``, as the multiply and sum would materialise a ``(..., m, k, n)`` product.
    """
    if torch.compiler.is_compiling():
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)
    return a @ b


__all__: list[str] = []
