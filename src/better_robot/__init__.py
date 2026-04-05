"""BetterRobot: PyTorch-native robot kinematics and optimization.

Quick start:
    import better_robot as br
    robot = br.Robot.from_urdf(urdf)

    # Fixed base — single or multiple targets
    cfg = br.solve_ik(robot, targets={"panda_hand": pose})

    # Floating base (humanoid whole-body IK)
    base_pose, cfg = br.solve_ik(
        robot,
        targets={"left_rubber_hand": p_lh, "right_rubber_hand": p_rh},
        initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
    )
"""

from .core._robot import Robot as Robot
from .tasks._config import IKConfig as IKConfig
from .tasks._ik import solve_ik as solve_ik
from .tasks._trajopt import solve_trajopt as solve_trajopt
from .tasks._retarget import retarget as retarget
from .viewer._visualizer import Visualizer as Visualizer
from . import collision as collision
from . import solvers as solvers
from . import costs as costs
from . import viewer as viewer

__version__ = "0.1.0"

__all__ = [
    "Robot",
    "IKConfig",
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "Visualizer",
    "collision",
    "solvers",
    "costs",
    "viewer",
]
