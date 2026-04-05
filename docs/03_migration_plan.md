# BetterRobot Restructuring: Step-by-Step Migration Plan

## Guiding Principles for Migration

1. **Green tests at every step.** Each phase ends with all existing tests passing.
2. **No logic changes during restructuring.** Move code, don't rewrite it. Logic improvements come later.
3. **Backward compatibility bridge.** Old imports work during transition via `_compat.py`.
4. **One module at a time.** Complete each phase before starting the next.

---

## Phase 0: Preparation

### 0.1 Create the new directory skeleton

```bash
mkdir -p src/better_robot/{models/parsers,math,algorithms/{kinematics,dynamics,geometry},costs,solvers,tasks/{ik,trajopt,retarget},viewer}
```

Create empty `__init__.py` in each new directory.

### 0.2 Add `_compat.py` for backward compatibility

```python
# src/better_robot/_compat.py
"""Temporary backward-compatible aliases. Remove after migration is complete."""
import warnings

def _deprecated_import(old_name, new_module, new_name):
    warnings.warn(
        f"{old_name} has moved to {new_module}.{new_name}. "
        f"Update your imports.",
        DeprecationWarning, stacklevel=3,
    )
```

### 0.3 Snapshot current test results

Run `uv run pytest tests/ -v` and save output. This is the baseline -- all tests
must continue passing after each phase.

---

## Phase 1: Extract `math/` from `core/_lie_ops.py`

**Goal:** Move all Lie group operations into `math/`, split by topic.

### Steps

1. Create `math/se3.py` with all SE3 functions from `_lie_ops.py`:
   - `se3_compose`, `se3_inverse`, `se3_log`, `se3_exp`, `se3_identity`, `se3_apply_base`

2. Create `math/so3.py` with SO3-specific helpers (if any are extracted).

3. Create `math/spatial.py` with `adjoint_se3` and `skew_symmetric`.

4. Create `math/transforms.py` with quaternion conversion utilities (from `viewer/_helpers.py`).

5. Create `math/__init__.py` re-exporting everything.

6. Update `core/_lie_ops.py` to re-export from `math/` (backward compat):
   ```python
   # core/_lie_ops.py -- backward compatibility
   from ..math.se3 import *  # noqa
   from ..math.spatial import *  # noqa
   ```

7. Run tests. All must pass.

**Files changed:** New `math/` files created. `core/_lie_ops.py` becomes a thin re-export layer.

---

## Phase 2: Extract `models/` from `core/`

**Goal:** Create the immutable `RobotModel` and model loading infrastructure.

### Steps

1. Create `models/joint_info.py`:
   - Move `JointInfo` dataclass from `core/_urdf_parser.py`
   - Add `JointType` enum: `REVOLUTE, CONTINUOUS, PRISMATIC, FIXED`

2. Create `models/link_info.py`:
   - Move `LinkInfo` dataclass from `core/_urdf_parser.py`

3. Create `models/robot_model.py`:
   - Create `RobotModel` dataclass (frozen or near-frozen)
   - Move all model-level data from current `Robot.__init__` and `Robot.from_urdf`
   - Keep convenience methods: `link_index()`, `default_cfg` property
   - Add `forward_kinematics()` as a convenience method that delegates to `algorithms/`

4. Create `models/parsers/urdf.py`:
   - Public function: `load_urdf(path_or_urdf) -> RobotModel`
   - Handles both `str` (file path) and `yourdfpy.URDF` objects
   - Returns fully-constructed `RobotModel`

5. Create `models/parsers/_urdf_impl.py`:
   - Move `RobotURDFParser.parse()` logic from `core/_urdf_parser.py`
   - Move FK data structure building from `Robot.from_urdf()`

6. Create `models/__init__.py`:
   ```python
   from .robot_model import RobotModel
   from .joint_info import JointInfo, JointType
   from .link_info import LinkInfo
   from .parsers import load_urdf
   ```

7. Update `core/_robot.py` to be a backward-compat wrapper:
   ```python
   # Backward compatibility
   from ..models import RobotModel as Robot
   ```

8. Run tests. All must pass.

**Key decision:** `RobotModel` should be `frozen=True` for the Model/Data pattern.
However, the current code sets many private attributes during `from_urdf()`. Solution:
use `object.__setattr__` during construction (standard pattern for frozen dataclasses),
or use a builder that constructs the model then freezes it.

---

## Phase 3: Extract `algorithms/kinematics/` from `core/_robot.py`

**Goal:** FK, Jacobian, and chain computation become free functions.

### Steps

1. Create `algorithms/kinematics/forward.py`:
   - Extract `forward_kinematics(model, q, base_pose=None) -> Tensor` as free function
   - Extract `_revolute_transform()` and `_prismatic_transform()` as module-level helpers
   - Uses `RobotModel` fields instead of `self._fk_*` attributes

2. Create `algorithms/kinematics/jacobian.py`:
   - Move `pose_jacobian()`, `limit_jacobian()`, `rest_jacobian()` from `costs/_jacobian.py`
   - These are mathematical algorithms, not costs
   - Rename: `compute_jacobian(model, q, link_idx, ...)` for the geometric Jacobian
   - Keep `limit_jacobian` and `rest_jacobian` as utilities

3. Create `algorithms/kinematics/chain.py`:
   - Move `get_chain()` from `Robot.get_chain()` to free function
   - `get_chain(model, link_idx) -> list[int]`

4. Create `algorithms/kinematics/__init__.py`:
   ```python
   from .forward import forward_kinematics
   from .jacobian import compute_jacobian, limit_jacobian, rest_jacobian
   from .chain import get_chain
   ```

5. Update `RobotModel` convenience methods to delegate:
   ```python
   def forward_kinematics(self, q, base_pose=None):
       from ..algorithms.kinematics import forward_kinematics
       return forward_kinematics(self, q, base_pose)
   ```

6. Run tests. All must pass.

---

## Phase 4: Move `collision/` into `algorithms/geometry/`

**Goal:** Collision geometry becomes part of the algorithms layer.

### Steps

1. Create `algorithms/geometry/primitives.py`:
   - Move `CollGeom`, `Sphere`, `Capsule`, `Box`, `HalfSpace`, `Heightmap` from `collision/_geometry.py`

2. Create `algorithms/geometry/robot_collision.py`:
   - Move `RobotCollision` from `collision/_robot_collision.py`
   - Rename to `RobotCollisionModel` for consistency

3. Create `algorithms/geometry/distance.py`:
   - Dispatcher function: `compute_distance(geom_a, geom_b) -> Tensor`
   - Dict-based dispatch on `(type(a), type(b))` (PyRoki pattern)

4. Create `algorithms/geometry/distance_pairs.py`:
   - Implement distance functions for each geometry pair
   - Start with: sphere-sphere, sphere-capsule, capsule-capsule, halfspace-sphere, halfspace-capsule

5. Update `collision/__init__.py` to re-export from new location (backward compat).

6. Run tests. All must pass.

---

## Phase 5: Restructure `costs/`

**Goal:** Clean cost module with factory functions that return `CostTerm` objects.

### Steps

1. Create `costs/cost_term.py`:
   - Move `CostTerm` from `solvers/_base.py` to here (it's a cost concept, not a solver concept)

2. Refactor `costs/pose.py` (rename from `_pose.py`):
   - Keep `pose_residual()` as the raw residual function
   - Add `pose_cost(model, link_idx, target, pos_weight, ori_weight, ...) -> CostTerm`
     factory that returns a ready-to-use `CostTerm` with `functools.partial` already applied

3. Similarly for `costs/limits.py`, `costs/regularization.py`:
   - Keep raw residual functions
   - Add factory functions: `limit_cost(model, weight) -> CostTerm`, `rest_cost(model, weight) -> CostTerm`

4. Implement `costs/collision.py`:
   - `self_collision_cost(model, collision_model, margin, weight) -> CostTerm`
   - `world_collision_cost(model, collision_model, world_geoms, margin, weight) -> CostTerm`
   - Uses `algorithms/geometry/distance.py` internally

5. Implement `costs/manipulability.py`:
   - `manipulability_cost(model, link_idx, weight) -> CostTerm`
   - Uses `algorithms/kinematics/jacobian.py` internally

6. Update `costs/__init__.py`:
   ```python
   from .cost_term import CostTerm
   from .pose import pose_cost, pose_residual
   from .limits import limit_cost, limit_residual
   from .regularization import rest_cost, smoothness_cost
   from .collision import self_collision_cost, world_collision_cost
   from .manipulability import manipulability_cost
   ```

7. Run tests. All must pass.

**Key insight:** The "factory function returning CostTerm" pattern means users don't
need to know about `functools.partial`. They just call `pose_cost(model, ...)` and
get back a ready-to-use cost term. Power users can still use the raw residual functions.

---

## Phase 6: Restructure `solvers/`

**Goal:** Clean solver module with proper registry and separated Problem/Solver.

### Steps

1. Create `solvers/problem.py`:
   - Move `Problem` from `_base.py`
   - Update to import `CostTerm` from `costs/cost_term.py`

2. Create `solvers/base.py`:
   - Move `Solver` ABC from `_base.py`

3. Create `solvers/registry.py`:
   - Generic `Registry` class
   - `SOLVERS = Registry("solvers")`

4. Rename `solvers/_lm.py` -> `solvers/levenberg_marquardt.py`:
   - Add `@SOLVERS.register("lm")` decorator
   - No logic changes

5. Rename `solvers/_levenberg_marquardt.py` -> `solvers/levenberg_marquardt_pypose.py`:
   - Add `@SOLVERS.register("lm_pypose")` decorator
   - No logic changes

6. Create `solvers/__init__.py`:
   ```python
   from .problem import Problem
   from .base import Solver
   from .registry import SOLVERS
   from .levenberg_marquardt import LevenbergMarquardt
   # Import stubs to trigger registration
   from . import gauss_newton, adam, lbfgs  # noqa
   ```

7. Run tests. All must pass.

---

## Phase 7: Restructure `tasks/`

**Goal:** Each task in its own subpackage with config + solver.

### Steps

1. Create `tasks/ik/config.py`:
   - Move `IKConfig` from `tasks/_config.py`
   - Add `solver` and `solver_params` fields

2. Create `tasks/ik/variable.py`:
   - Implement `IKVariable` that unifies fixed and floating base

3. Create `tasks/ik/solver.py`:
   - Unified `solve_ik()` using `IKVariable`
   - No more separate `_floating_base_ik.py`
   - Both paths use the same solver through the variable abstraction

4. Create `tasks/ik/__init__.py`:
   ```python
   from .solver import solve_ik
   from .config import IKConfig
   ```

5. Create placeholder `tasks/trajopt/` and `tasks/retarget/` with proper structure.

6. Update `tasks/__init__.py`:
   ```python
   from .ik import solve_ik, IKConfig
   from .trajopt import solve_trajopt, TrajOptConfig
   from .retarget import retarget, RetargetConfig
   ```

7. Run tests. All must pass.

---

## Phase 8: Update Top-Level `__init__.py`

**Goal:** Clean public API.

```python
# src/better_robot/__init__.py
"""BetterRobot: PyTorch-native robot kinematics and optimization."""

# Model loading
from .models import RobotModel, load_urdf

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
```

---

## Phase 9: Update Tests

### Steps

1. Update all test imports from old paths to new paths.
2. Replace `Robot` with `RobotModel` in test fixtures.
3. Replace `robot.forward_kinematics(q)` with `br.forward_kinematics(model, q)` or `model.forward_kinematics(q)`.
4. Verify all 34 tests pass.
5. Add new tests:
   - `test_models.py`: RobotModel construction, immutability, link_index
   - `test_algorithms.py`: FK as free function, Jacobian as free function
   - `test_registry.py`: Solver registry register/get/list

---

## Phase 10: Cleanup

### Steps

1. Remove `core/` directory entirely (all code moved to `models/`, `math/`, `algorithms/`).
2. Remove `collision/` directory entirely (moved to `algorithms/geometry/`).
3. Remove `_compat.py` if no external users depend on old imports.
4. Remove old `CLAUDE.md` files that reference the old structure.
5. Update `CLAUDE.md` at project root with new architecture.
6. Update `README.md` with new import patterns.

---

## Summary Timeline

| Phase | What | Risk | Estimated Effort |
|-------|------|------|-----------------|
| 0 | Preparation | None | Small |
| 1 | Extract `math/` | Low -- pure extraction | Small |
| 2 | Extract `models/` | Medium -- frozen dataclass design | Medium |
| 3 | Extract `algorithms/kinematics/` | Medium -- FK refactor | Medium |
| 4 | Move collision to `algorithms/geometry/` | Low -- mostly stubs | Small |
| 5 | Restructure `costs/` | Low -- add factory pattern | Medium |
| 6 | Restructure `solvers/` | Low -- rename + registry | Small |
| 7 | Restructure `tasks/` | High -- unify floating base | Large |
| 8 | Update top-level API | Low | Small |
| 9 | Update tests | Medium | Medium |
| 10 | Cleanup | Low | Small |

**Total: ~10 phases, each independently testable.**

---

## What NOT to Change

1. **SE3 convention** -- `[tx, ty, tz, qx, qy, qz, qw]` stays.
2. **PyPose as Lie group backend** -- The `math/` module still uses PyPose internally.
3. **LM solver logic** -- The damping, rejection, and projection logic is correct.
4. **Residual function signatures** -- `(x: Tensor) -> Tensor` pattern stays.
5. **Test fixtures** -- Panda URDF testing stays.
6. **IKConfig default weights** -- `ori_weight=0.1` is intentional, keep it.
7. **Limit residual clamping** -- `torch.clamp(min=0)` is critical, never remove.

---

## Future Extensions Enabled by This Architecture

Once the restructuring is complete, these become straightforward additions:

| Feature | Where to Add | What to Implement |
|---------|-------------|-------------------|
| **MJCF parser** | `models/parsers/mjcf.py` | New parser, returns same `RobotModel` |
| **Inverse dynamics (RNEA)** | `algorithms/dynamics/rnea.py` | Free function, uses `RobotModel` + inertia data |
| **Forward dynamics (ABA)** | `algorithms/dynamics/aba.py` | Free function |
| **Mass matrix (CRBA)** | `algorithms/dynamics/crba.py` | Free function |
| **Gravity compensation** | `algorithms/dynamics/energy.py` | Uses RNEA internally |
| **GJK collision** | `algorithms/geometry/gjk.py` | New distance algorithm |
| **Mesh collision** | `algorithms/geometry/mesh.py` | Uses trimesh + GJK |
| **RRT planner** | `tasks/planning/rrt.py` | New task, uses collision + FK |
| **PRM planner** | `tasks/planning/prm.py` | New task |
| **Impedance control** | `tasks/control/impedance.py` | Uses dynamics + kinematics |
| **Contact dynamics** | `algorithms/dynamics/contact.py` | Uses geometry + dynamics |
| **CasADi backend** | `math/backends/casadi.py` | Alternative to PyPose (codegen) |
| **Batched GPU solving** | `solvers/batched_lm.py` | Vectorized solver variant |
| **Robot description assets** | `assets/` top-level | Pre-configured URDFs with configs |

Each of these is a single file or subpackage addition. None requires modifying existing code.
That is the hallmark of a well-structured library.
