"""BetterRobot: PyTorch-native robot kinematics and optimization.

Quick start::

    import better_robot as br
    import yourdfpy

    urdf = yourdfpy.URDF.load("robot.urdf")
    model = br.load_urdf(urdf)

    # Fixed base — single or multiple targets
    cfg = br.solve_ik(model, targets={"panda_hand": pose})

    # Floating base (humanoid whole-body IK)
    base_pose, cfg = br.solve_ik(
        model,
        targets={"left_rubber_hand": p_lh, "right_rubber_hand": p_rh},
        initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
    )
"""

# Model loading
from .models import RobotModel, RobotData, load_urdf
from .models import JointInfo, LinkInfo

# Algorithms (convenience re-exports)
from .algorithms.kinematics import forward_kinematics, compute_jacobian

# Tasks (high-level API)
from .tasks.ik import solve_ik, IKConfig
from .tasks.trajopt import solve_trajopt, TrajOptConfig
from .tasks.retarget import retarget, RetargetConfig

# Visualization
from .viewer import Visualizer

# Submodule access
from . import models, algorithms, math, costs, solvers, tasks, viewer

__version__ = "0.1.0"

__all__ = [
    # Model loading
    "RobotModel",
    "RobotData",
    "load_urdf",
    "JointInfo",
    "LinkInfo",
    # Algorithms
    "forward_kinematics",
    "compute_jacobian",
    # Tasks
    "solve_ik",
    "IKConfig",
    "solve_trajopt",
    "TrajOptConfig",
    "retarget",
    "RetargetConfig",
    # Visualization
    "Visualizer",
    # Submodules
    "models",
    "algorithms",
    "math",
    "costs",
    "solvers",
    "tasks",
    "viewer",
]
