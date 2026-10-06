"""Optimization over named tensors: plain-function terms, torch.optim, Gauss-Newton and LM.

This package knows nothing about robots. Robot terms live in ``better_robot.terms``.
"""

from __future__ import annotations

from .kernels import Cauchy, GemanMcClure, Huber, Tukey
from .linear import Normal
from .manifold import Euclidean, Manifold
from .phases import Phase, Result, run, solve
from .problem import Problem, TermFn, check_jacobian
from .solvers import GaussNewton, LevenbergMarquardt, Solver, State, TorchOptim

__all__ = [
    "Cauchy",
    "Euclidean",
    "GaussNewton",
    "GemanMcClure",
    "Huber",
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
]
