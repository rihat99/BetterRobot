"""Solver layer: swappable optimization backends."""

from ._base import CostTerm as CostTerm, Problem as Problem, Solver as Solver
from ._levenberg_marquardt import LevenbergMarquardt as LevenbergMarquardt
from ._gauss_newton import GaussNewton as GaussNewton
from ._adam import AdamSolver as AdamSolver
from ._lbfgs import LBFGSSolver as LBFGSSolver

SOLVER_REGISTRY: dict[str, type[Solver]] = {
    "lm": LevenbergMarquardt,
    "gn": GaussNewton,
    "adam": AdamSolver,
    "lbfgs": LBFGSSolver,
}

__all__ = [
    "CostTerm", "Problem", "Solver",
    "LevenbergMarquardt", "GaussNewton", "AdamSolver", "LBFGSSolver",
    "SOLVER_REGISTRY",
]
