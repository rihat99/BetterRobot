"""BetterRobot: PyTorch-native robot kinematics and optimization.

Quick start:
    import better_robot as br
    robot = br.Robot.from_urdf(urdf)
    solution = br.solve_ik(robot, target_link="panda_hand", target_pose=pose)
"""

from .core._robot import Robot as Robot
from .tasks._ik import solve_ik as solve_ik
from .tasks._trajopt import solve_trajopt as solve_trajopt
from .tasks._retarget import retarget as retarget
from . import collision as collision
from . import solvers as solvers
from . import costs as costs
from . import viewer as viewer

__version__ = "0.1.0"

__all__ = [
    "Robot",
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "collision",
    "solvers",
    "costs",
    "viewer",
]
