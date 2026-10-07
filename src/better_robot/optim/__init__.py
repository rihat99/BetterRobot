"""Optimization over named tensors: plain-function terms, torch.optim, Gauss-Newton and LM.

A ``Problem`` holds variables, each a named tensor ``(B, *event)``; the ``B`` problems of a batch are independent.
A term is a plain function whose parameter names are the variables, ``data`` tensors or ``cached`` values it reads;
``TermFn`` adds an analytic ``jac`` or a time ``window``. ``Problem.loss`` is an ordinary differentiable ``(B,)``
tensor, so any ``torch.optim`` optimizer can minimize it (``TorchOptim``). ``Problem.linearize`` builds the
Gauss-Newton normal equations from the same terms: ``Normal`` (dense), ``BandedNormal`` (block-tridiagonal, for
trajectories declared with ``Problem(time=...)``) or ``ArrowNormal`` (banded, bordered by variables shared over
time); all three solve the same way. Solvers (``LevenbergMarquardt``, ``GaussNewton``, ``LBFGS``) are
``init(problem, x) -> State`` and ``step(problem, state) -> State``. ``solve`` runs one solver, ``run`` chains
``Phase``s with weight overrides and frozen coordinates, ``solve_layer`` makes the solution differentiable w.r.t.
the data, and ``solve_constrained`` adds hard constraints (augmented Lagrangian). ``check_jacobian`` tests every
analytic ``jac`` and ``window`` against autodiff. Robust kernels (``Huber``, ``Cauchy``, ...) down-weight outliers.

This package imports only torch and knows nothing about robots; robot terms live in ``better_robot.terms``.

```python
import torch
from better_robot.optim import LevenbergMarquardt, Problem, solve

problem = Problem({"x": torch.zeros(4, 2)}, data={"target": torch.randn(4, 2)})
problem.residual("fit", lambda x, target: x - target)
result = solve(problem, LevenbergMarquardt(), iters=20)  # result.x["x"]: (4, 2)
```
"""

from __future__ import annotations

from .kernels import Cauchy, GemanMcClure, Huber, Tukey
from .constrained import solve_constrained
from .linear import ArrowNormal, BandedNormal, Normal
from .manifold import Euclidean, Manifold
from .phases import Phase, Result, run, solve, solve_layer
from .problem import Problem, TermFn, check_jacobian
from .solvers import LBFGS, GaussNewton, LevenbergMarquardt, Solver, State, TorchOptim

__all__ = [
    "ArrowNormal",
    "BandedNormal",
    "Cauchy",
    "Euclidean",
    "GaussNewton",
    "GemanMcClure",
    "Huber",
    "LBFGS",
    "LevenbergMarquardt",
    "Manifold",
    "Normal",
    "Phase",
    "Problem",
    "Result",
    "Solver",
    "State",
    "TermFn",
    "TorchOptim",
    "Tukey",
    "check_jacobian",
    "run",
    "solve",
    "solve_constrained",
    "solve_layer",
]
