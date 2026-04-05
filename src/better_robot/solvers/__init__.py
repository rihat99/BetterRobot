"""Solvers layer: optimization backends for robotics problems."""
from .problem import Problem
from .base import Solver
from .registry import SOLVERS, Registry

# Import solver classes to trigger registry registration
from .levenberg_marquardt import LevenbergMarquardt
from .levenberg_marquardt_pypose import PyposeLevenbergMarquardt
from .gauss_newton import GaussNewton
from .adam import AdamSolver
from .lbfgs import LBFGSSolver

__all__ = [
    "Problem", "Solver", "SOLVERS", "Registry",
    "LevenbergMarquardt", "PyposeLevenbergMarquardt",
    "GaussNewton", "AdamSolver", "LBFGSSolver",
]
