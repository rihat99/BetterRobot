"""Solver layer: swappable optimization backends."""

from ._base import CostTerm as CostTerm, Problem as Problem, Solver as Solver
from ._lm import LevenbergMarquardt as LevenbergMarquardt
from ._levenberg_marquardt import LevenbergMarquardt as PyposeLevenbergMarquardt
from ._gauss_newton import GaussNewton as GaussNewton
from ._adam import AdamSolver as AdamSolver
from ._lbfgs import LBFGSSolver as LBFGSSolver

SOLVER_REGISTRY: dict[str, type[Solver]] = {
    "lm": LevenbergMarquardt,               # our LM (autodiff or analytic via jacobian_fn)
    "lm_pypose": PyposeLevenbergMarquardt,  # PyPose LM (kept for comparison)
    "gn": GaussNewton,
    "adam": AdamSolver,
    "lbfgs": LBFGSSolver,
}

__all__ = [
    "CostTerm", "Problem", "Solver",
    "LevenbergMarquardt", "PyposeLevenbergMarquardt",
    "GaussNewton", "AdamSolver", "LBFGSSolver",
    "SOLVER_REGISTRY",
]
