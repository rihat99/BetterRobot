"""Optimization over named tensors: plain-function terms, torch.optim, Gauss-Newton and LM.

This package knows nothing about robots. Robot terms live in ``better_robot.terms``.
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
