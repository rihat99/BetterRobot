# BetterRobot Restructuring: Target Architecture

## 1. New Package Structure

```
src/better_robot/
│
├── __init__.py                    # Thin public API (Robot, solve_ik, etc.)
│
├── models/                        # ======= ROBOT MODEL (immutable structure) =======
│   ├── __init__.py                # Exports: RobotModel, JointType, JointInfo, LinkInfo
│   ├── robot_model.py             # RobotModel dataclass (frozen=True)
│   ├── joint_info.py              # JointInfo, JointType enum
│   ├── link_info.py               # LinkInfo dataclass
│   └── parsers/                   # Model loading from various formats
│       ├── __init__.py
│       ├── urdf.py                # load_urdf(path_or_urdf) -> RobotModel
│       └── _urdf_impl.py         # Internal URDF parsing (current _urdf_parser.py logic)
│       # Future: mjcf.py, sdf.py, etc.
│
├── math/                          # ======= MATH UTILITIES =======
│   ├── __init__.py                # Exports: se3, so3, spatial
│   ├── se3.py                     # SE3 operations: compose, inverse, log, exp, identity
│   ├── so3.py                     # SO3 operations: from_matrix, to_matrix, log, exp
│   ├── spatial.py                 # Spatial algebra: adjoint, cross products, wrenches
│   └── transforms.py             # Conversion utilities: quat_to_matrix, euler_to_quat, etc.
│
├── algorithms/                    # ======= ROBOTICS ALGORITHMS (free functions) =======
│   ├── __init__.py                # Exports all algorithm submodules
│   ├── kinematics/
│   │   ├── __init__.py
│   │   ├── forward.py             # forward_kinematics(model, q, base_pose=None) -> LinkPoses
│   │   ├── jacobian.py            # compute_jacobian(model, q, link_idx) -> (6, n) Tensor
│   │   ├── chain.py               # get_chain(model, link_idx) -> list[int]
│   │   └── frames.py             # frame utilities: link-to-world, world-to-link
│   │   # Future: differential_kinematics.py, velocity_kinematics.py
│   │
│   ├── dynamics/                  # Future: dynamics algorithms
│   │   ├── __init__.py
│   │   ├── rnea.py                # inverse_dynamics(model, q, qd, qdd) -> tau
│   │   ├── aba.py                 # forward_dynamics(model, q, qd, tau) -> qdd
│   │   ├── crba.py                # mass_matrix(model, q) -> M
│   │   └── energy.py              # kinetic_energy, potential_energy, gravity_compensation
│   │
│   └── geometry/                  # Geometric algorithms (collision, distance)
│       ├── __init__.py
│       ├── primitives.py          # Sphere, Capsule, Box, HalfSpace, Heightmap
│       ├── distance.py            # Pairwise distance dispatcher (PyRoki pattern)
│       ├── distance_pairs.py      # All geometry-pair distance functions
│       ├── robot_collision.py     # RobotCollisionModel: sphere decomposition for a robot
│       └── gjk.py                 # Future: GJK/EPA for convex meshes
│
├── costs/                         # ======= COST/RESIDUAL FUNCTIONS =======
│   ├── __init__.py                # Exports: CostTerm, pose_cost, limit_cost, etc.
│   ├── cost_term.py               # CostTerm dataclass (residual_fn, weight, kind)
│   ├── pose.py                    # pose_cost(model, q, link_idx, target, ...) -> residual
│   ├── limits.py                  # limit_cost, velocity_cost, acceleration_cost, jerk_cost
│   ├── regularization.py          # rest_cost, smoothness_cost
│   ├── collision.py               # self_collision_cost, world_collision_cost
│   ├── manipulability.py          # manipulability_cost (Yoshikawa measure)
│   └── trajectory.py             # Future: waypoint_cost, endpoint_cost
│
├── solvers/                       # ======= OPTIMIZATION SOLVERS =======
│   ├── __init__.py                # Exports: Solver, Problem, solve, registry
│   ├── problem.py                 # Problem dataclass (variables, costs, bounds)
│   ├── base.py                    # Solver ABC
│   ├── registry.py                # SolverRegistry with register/get pattern
│   ├── levenberg_marquardt.py     # LM solver (our default)
│   ├── gauss_newton.py            # Future: GN solver
│   ├── adam.py                    # Future: Adam solver
│   └── lbfgs.py                   # Future: L-BFGS solver
│
├── tasks/                         # ======= HIGH-LEVEL TASK APIs =======
│   ├── __init__.py                # Exports: solve_ik, solve_trajopt, retarget
│   ├── ik/
│   │   ├── __init__.py            # Exports: solve_ik, IKConfig
│   │   ├── config.py              # IKConfig dataclass
│   │   ├── solver.py              # solve_ik() -- unified fixed + floating base
│   │   └── variable.py            # IKVariable: wraps cfg + optional base_pose
│   ├── trajopt/                   # Future: trajectory optimization
│   │   ├── __init__.py
│   │   ├── config.py              # TrajOptConfig
│   │   └── solver.py              # solve_trajopt()
│   └── retarget/                  # Future: motion retargeting
│       ├── __init__.py
│       ├── config.py              # RetargetConfig
│       └── solver.py              # retarget()
│
├── viewer/                        # ======= VISUALIZATION =======
│   ├── __init__.py
│   ├── visualizer.py              # Visualizer class (viser-based)
│   └── helpers.py                 # Quaternion conversion, mesh utils
│
└── _compat.py                     # Temporary: backward-compatible aliases during migration
```

---

## 2. Core Design Patterns

### 2.1 Model/Data Separation (from Pinocchio)

```python
# models/robot_model.py
from dataclasses import dataclass

@dataclass(frozen=True)
class RobotModel:
    """Immutable robot kinematic (and future dynamic) structure.

    This is the 'Model' in the Model/Data pattern. It contains all
    static robot parameters and is safe to share across threads.
    """
    joints: JointInfo           # Joint tree: names, types, axes, limits, origins
    links: LinkInfo             # Link tree: names, parent joints
    name: str = ""              # Robot name (from URDF)

    # Precomputed FK data (set during construction)
    _fk_joint_parent_link: tuple[int, ...] = ()
    _fk_joint_child_link: tuple[int, ...]  = ()
    _fk_joint_origins: torch.Tensor | None = None  # (num_joints, 7)
    _fk_joint_types: tuple[str, ...]       = ()
    _fk_cfg_indices: tuple[int, ...]       = ()
    _fk_joint_axes: torch.Tensor | None    = None   # (num_joints, 3)
    _root_link_idx: int = 0
    _default_cfg: torch.Tensor | None = None

    @property
    def num_joints(self) -> int:
        return self.joints.num_joints

    @property
    def num_actuated(self) -> int:
        return self.joints.num_actuated_joints

    @property
    def num_links(self) -> int:
        return self.links.num_links

    def link_index(self, name: str) -> int:
        """Return link index by name. Raises ValueError if not found."""
        ...
```

**Why frozen?** A frozen dataclass is hashable, immutable, and thread-safe. Multiple
IK solvers, trajectory optimizers, or parallel evaluations can share the same model
without copies or locks.

**What about `_default_cfg`?** Default config is model metadata (midpoint of limits).
It's set once at construction and never changes.

### 2.2 Algorithms as Free Functions (from Pinocchio)

```python
# algorithms/kinematics/forward.py

def forward_kinematics(
    model: RobotModel,
    q: torch.Tensor,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute world poses of all links.

    Args:
        model: Robot model (immutable structure).
        q: Joint configuration (*batch, num_actuated).
        base_pose: Optional base transform (*batch, 7).

    Returns:
        Link poses (*batch, num_links, 7) in [tx, ty, tz, qx, qy, qz, qw].
    """
    ...
```

**Why free functions?**
- The Robot class currently has `forward_kinematics()` as a method. This means
  adding `inverse_dynamics()` requires modifying the Robot class.
- With free functions, each algorithm is independently testable and importable.
- Users can `from better_robot.algorithms.kinematics import forward_kinematics`
  without pulling in the solver or cost machinery.

**Convenience alias:** `RobotModel` can still have a thin wrapper:
```python
class RobotModel:
    def forward_kinematics(self, q, base_pose=None):
        """Convenience method. Delegates to algorithms.kinematics.forward."""
        from ..algorithms.kinematics.forward import forward_kinematics
        return forward_kinematics(self, q, base_pose)
```

### 2.3 Unified Variable System (fix floating-base duplication)

```python
# tasks/ik/variable.py

@dataclass
class IKVariable:
    """Wraps joint config and optional base pose as a single optimization variable."""
    cfg: torch.Tensor           # (num_actuated,)
    base_pose: torch.Tensor | None = None  # (7,) or None for fixed base

    def to_flat(self) -> torch.Tensor:
        """Flatten to single vector for the solver."""
        if self.base_pose is None:
            return self.cfg
        # base tangent (6,) + cfg (n,) = (n+6,)
        return torch.cat([torch.zeros(6), self.cfg])

    @staticmethod
    def from_flat(flat: torch.Tensor, num_actuated: int,
                  has_base: bool, current_base: torch.Tensor | None) -> "IKVariable":
        """Reconstruct from flat solver output."""
        if not has_base:
            return IKVariable(cfg=flat)
        base_tangent = flat[:6]
        cfg = flat[6:]
        new_base = se3_compose(se3_exp(base_tangent), current_base)
        return IKVariable(cfg=cfg, base_pose=new_base)
```

**Why?** Currently `_floating_base_ik.py` duplicates the entire LM loop to handle
base + joints. With a unified variable system, the same solver handles both cases.
The floating-base path becomes just a different variable wrapping, not a different solver.

### 2.4 Registry Pattern (from IsaacLab)

```python
# solvers/registry.py

class Registry:
    """Generic registry for named components."""

    def __init__(self, name: str):
        self._name = name
        self._entries: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a class."""
        def wrapper(cls):
            self._entries[name] = cls
            return cls
        return wrapper

    def get(self, name: str):
        if name not in self._entries:
            raise KeyError(f"'{name}' not in {self._name} registry. "
                          f"Available: {list(self._entries.keys())}")
        return self._entries[name]

    def list(self) -> list[str]:
        return list(self._entries.keys())

# Usage:
SOLVERS = Registry("solvers")

@SOLVERS.register("lm")
class LevenbergMarquardt(Solver):
    ...
```

**Apply to:** Solvers, model parsers (URDF, MJCF, SDF), cost functions, geometry distance pairs.

### 2.5 Config Dataclass Pattern (from IsaacLab)

```python
# Every task has a typed config

@dataclass
class IKConfig:
    pos_weight: float = 1.0
    ori_weight: float = 0.1
    pose_weight: float = 1.0
    limit_weight: float = 0.1
    rest_weight: float = 0.01
    solver: str = "lm"                    # Solver name from registry
    solver_params: dict = field(default_factory=dict)  # Solver-specific params
    jacobian: Literal["autodiff", "analytic"] = "autodiff"
    max_iter: int = 100

@dataclass
class TrajOptConfig:
    num_waypoints: int = 20
    smoothness_weight: float = 1.0
    collision_weight: float = 10.0
    solver: str = "lm"
    ...
```

---

## 3. Dependency Graph

```
Layer 0: math/              (zero internal deps -- pure PyTorch + PyPose)
           |
Layer 1: models/            (depends on: math/)
           |
Layer 2: algorithms/        (depends on: models/, math/)
           |
Layer 3: costs/             (depends on: algorithms/, models/, math/)
           |
Layer 4: solvers/           (depends on: costs/ for types only)
           |
Layer 5: tasks/             (depends on: solvers/, costs/, algorithms/, models/)
           |
Layer 6: viewer/            (depends on: models/, algorithms/kinematics)
```

**Key rule:** No upward dependencies. `math/` never imports from `models/`.
`algorithms/` never imports from `costs/` or `solvers/`. This is what makes the
library modular -- users can use `algorithms/` without pulling in the optimization stack.

---

## 4. Public API Design

### Top-Level Imports (what users see)

```python
import better_robot as br

# Load a robot
model = br.load_urdf("path/to/robot.urdf")
# or: model = br.load_urdf(yourdfpy.URDF.load("path"))

# Forward kinematics
poses = br.forward_kinematics(model, q)

# Jacobian
J = br.compute_jacobian(model, q, link_idx)

# Inverse kinematics
cfg = br.solve_ik(model, targets={"hand": target_pose})

# Floating-base IK
base, cfg = br.solve_ik(model, targets={...}, initial_base_pose=bp)

# Trajectory optimization (future)
traj = br.solve_trajopt(model, start_cfg, goal_cfg, config=br.TrajOptConfig())

# Visualization
vis = br.Visualizer(model)
vis.update(q)
```

### Module-Level Imports (for power users)

```python
from better_robot.models import RobotModel, load_urdf
from better_robot.algorithms.kinematics import forward_kinematics, compute_jacobian
from better_robot.algorithms.geometry import Sphere, Capsule, distance
from better_robot.costs import CostTerm, pose_cost, limit_cost
from better_robot.solvers import LevenbergMarquardt, Problem
from better_robot.tasks.ik import solve_ik, IKConfig
from better_robot.math import se3, so3
```

### Key API Changes from Current

| Current | New | Reason |
|---------|-----|--------|
| `Robot.from_urdf(urdf)` | `br.load_urdf(path)` or `RobotModel.from_urdf(urdf)` | Cleaner entry point; accepts path or URDF object |
| `robot.forward_kinematics(q)` | `br.forward_kinematics(model, q)` | Algorithm as free function |
| `robot.get_link_index(name)` | `model.link_index(name)` | Cleaner name |
| `robot.get_chain(idx)` | `get_chain(model, idx)` | Free function in algorithms/ |
| `from costs import pose_residual` | `from costs import pose_cost` | Consistent naming (cost, not residual) |
| `CostTerm(residual_fn=partial(...))` | `pose_cost(model, link_idx, target, ...)` returns `CostTerm` | Factory functions create ready-to-use cost terms |
| `SOLVER_REGISTRY["lm"]()` | `SOLVERS.get("lm")()` | Generic registry pattern |
| Separate fixed/floating IK paths | Unified via IKVariable | No code duplication |

---

## 5. File-by-File Migration Map

| Current File | New Location | Changes |
|-------------|-------------|---------|
| `core/_robot.py` | `models/robot_model.py` + `algorithms/kinematics/forward.py` | Split: model data vs FK algorithm |
| `core/_lie_ops.py` | `math/se3.py` + `math/so3.py` + `math/spatial.py` | Split by topic; cleaner imports |
| `core/_urdf_parser.py` | `models/parsers/_urdf_impl.py` | Same logic, new home |
| `costs/_pose.py` | `costs/pose.py` | Remove underscore; return CostTerm directly |
| `costs/_limits.py` | `costs/limits.py` | Remove underscore; add velocity/accel/jerk |
| `costs/_regularization.py` | `costs/regularization.py` | Remove underscore |
| `costs/_jacobian.py` | `algorithms/kinematics/jacobian.py` | Jacobians are algorithms, not costs |
| `costs/_collision.py` | `costs/collision.py` | Implement using algorithms/geometry |
| `costs/_manipulability.py` | `costs/manipulability.py` | Implement using algorithms/kinematics |
| `solvers/_base.py` | `solvers/problem.py` + `solvers/base.py` | Split Problem and Solver ABC |
| `solvers/_lm.py` | `solvers/levenberg_marquardt.py` | Remove underscore; add registry decorator |
| `solvers/_levenberg_marquardt.py` | `solvers/levenberg_marquardt_pypose.py` | Rename for clarity |
| `tasks/_ik.py` + `_floating_base_ik.py` | `tasks/ik/solver.py` | Unify via IKVariable |
| `tasks/_config.py` | `tasks/ik/config.py` | Same content, better location |
| `collision/_geometry.py` | `algorithms/geometry/primitives.py` | Geometry is an algorithm concern |
| `collision/_robot_collision.py` | `algorithms/geometry/robot_collision.py` | Same |
| `viewer/_visualizer.py` | `viewer/visualizer.py` | Remove underscore |
| `viewer/_helpers.py` | `viewer/helpers.py` | Remove underscore |

---

## 6. Convention Changes

### Naming

| Convention | Current | New |
|-----------|---------|-----|
| Internal modules | `_module.py` (underscore prefix) | `module.py` (no prefix; use `__all__` for API control) |
| Public classes | Mixed | PascalCase, descriptive: `RobotModel`, `CostTerm`, `IKConfig` |
| Algorithm functions | `pose_residual`, `limit_residual` | `pose_cost`, `limit_cost` (they return cost terms) |
| Private helpers | Mixed | Prefix with `_` only for truly private functions within a module |

### SE3 Convention (unchanged)

`[tx, ty, tz, qx, qy, qz, qw]` -- PyPose native, scalar last. This is fundamental
and should not change.

### Import Convention

```python
# Prefer:
from better_robot.algorithms.kinematics import forward_kinematics

# Over:
import better_robot.algorithms.kinematics.forward as fk_mod
fk_mod.forward_kinematics(...)
```

Each `__init__.py` re-exports the public API of its subpackage. Users should never
need to import from a specific file -- always from the package.
