# BetterRobot Restructuring: Analysis of Current State & Reference Projects

## 1. Current BetterRobot Architecture

### What We Have

```
src/better_robot/
  core/        _robot.py, _lie_ops.py, _urdf_parser.py
  costs/       _pose.py, _limits.py, _regularization.py, _jacobian.py, _collision.py (stub), _manipulability.py (stub)
  solvers/     _base.py, _lm.py, _levenberg_marquardt.py, _gauss_newton.py (stub), _adam.py (stub), _lbfgs.py (stub)
  tasks/       _ik.py, _floating_base_ik.py, _config.py, _trajopt.py (stub), _retarget.py (stub)
  collision/   _geometry.py, _robot_collision.py (stub)
  viewer/      _visualizer.py, _helpers.py
```

### What Works Well

1. **Lie group centralization** -- All SE3 math in `_lie_ops.py` makes backend swapping possible.
2. **Functional residual pattern** -- Pure `(Tensor) -> Tensor` functions are composable and testable.
3. **Dual Jacobian paths** -- Autodiff and analytical options give users flexibility.
4. **Fixed + floating base unified API** -- Single `solve_ik()` dispatches correctly.
5. **Good test coverage** -- 34 tests covering FK, costs, solvers, IK, Jacobians.

### Pain Points (Why Restructure)

| Problem | Impact | Example |
|---------|--------|---------|
| **Flat module hierarchy** | Hard to find things; `costs/` mixes residuals, Jacobians, and collision stubs | `_jacobian.py` lives in `costs/` but is really a math utility |
| **Robot class is a god object** | `Robot` stores FK data, URDF data, default configs, chain computation. Hard to extend. | Adding dynamics requires modifying the same class |
| **No Model/Data separation** | State (cfg, FK results) and structure (joint tree, limits) are interleaved | Can't share one model across parallel solvers |
| **Config is IK-only** | `IKConfig` is hardcoded for IK; no config pattern for trajopt, retarget, dynamics | Adding trajectory optimization means another bespoke config |
| **Collision is orphaned** | `collision/` defines geometry but `costs/_collision.py` has stubs; no clear integration path | Two half-implemented subsystems |
| **No registry/discovery pattern** | Solvers use a dict registry, but nothing else does | Adding a new solver type requires editing `__init__.py` |
| **Floating-base is special-cased** | `_floating_base_ik.py` duplicates solver logic instead of generalizing the variable system | Every new task needs its own floating-base variant |
| **No clear extension mechanism** | Third-party code has no documented way to add new joint types, costs, or solvers | Researchers can't plug in custom components without forking |
| **Mixed abstraction levels in tasks/** | `tasks/` contains both the public API (`solve_ik`) and implementation details (`_floating_base_ik`) | Confusing for contributors |
| **Private underscore conventions used inconsistently** | Most files start with `_` but some are public | Hard to know what's API vs internal |

### Planned Features (from stubs and TODOs)

- Collision avoidance (self + world)
- Trajectory optimization
- Motion retargeting
- Additional solvers (GN, Adam, L-BFGS)
- Manipulability metrics
- Velocity/acceleration/jerk constraints
- (Future) Dynamics, contact, planning, control

---

## 2. Lessons from Reference Projects

### Pinocchio (C++ -- target scope)

**Key pattern: Model/Data separation**

```
Model  = immutable robot structure (joints, links, inertias, limits)
Data   = mutable computation buffers (FK results, Jacobians, forces)
```

- Algorithms are **free functions**: `rnea(model, data, q, v, a)` -- not methods on Model.
- Each algorithm in its own file: `aba.hpp`, `rnea.hpp`, `crba.hpp`, `kinematics.hpp`.
- Templates on scalar type enable autodiff without code duplication.
- Python bindings mirror C++ structure exactly.

**What to adopt:**
- Separate robot structure from computation state
- Algorithms as standalone functions, not Robot methods
- Per-algorithm modules with clear interfaces
- Plan for analytical derivatives alongside algorithms

### IsaacLab (Python -- target modularity)

**Key pattern: Config-driven + Manager composition**

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(robot=..., ground=...)
    actions = ActionsCfg(...)
    observations = ObservationsCfg(...)
    rewards = RewardsCfg(...)
```

- **Configclass decorator**: Wraps dataclass with `to_dict()`, `copy()`, `replace()`.
- **Registry pattern**: Components discovered by name, instantiated from configs.
- **Manager pattern**: Each MDP component (observations, rewards, actions) has its own manager.
- **Extension template**: Standard way for third parties to add tasks/robots.

**What to adopt:**
- Configuration as first-class objects (not just IKConfig)
- Registry pattern for robots, solvers, cost terms
- Clear extension template
- Separation of "what" (config) from "how" (implementation)

### MjLab / dm_control (Python -- target simplicity)

**Key pattern: Entity + Composable Terms**

```python
# MjLab: modular terms compose into complex behaviors
reward_cfg = RewardsCfg(
    alive_bonus={"func": is_alive, "weight": 0.1},
    task_reward={"func": track_target, "weight": 1.0},
)
```

- **Entity pattern**: Self-contained robot/object with `_build()` lifecycle.
- **Observable abstraction**: Sensor access decoupled from state.
- **Zero-copy tensor abstraction**: Bridges simulation and PyTorch.
- **Task registration**: `register_task()` for plugin-like extensibility.

**What to adopt:**
- Functional term composition (already partially done)
- Task registration for extensibility
- Clean entity lifecycle

### PyRoki (Python -- closest relative)

**Key pattern: JAX pytree dataclass + jaxls costs**

```python
@jdc.pytree_dataclass
class Robot:
    joints: JointInfo
    links: LinkInfo
```

- **Pytree dataclass**: Immutable, JIT-compatible, differentiable.
- **Cost factory**: `Cost.factory(residual_fn)` wraps residuals into solver-ready objects.
- **Collision dispatcher**: Dict mapping `(GeomType, GeomType)` to distance functions.
- **Analytical Jacobian**: Custom Jacobian via `jac_custom_with_cache_fn`.

**Strengths to keep:**
- Functional residuals (BetterRobot already does this)
- Geometry collision dispatch pattern

**Weaknesses to avoid:**
- No dynamics at all
- Static JAX shapes require padding
- Limited joint types
- No Model/Data separation

---

## 3. Cross-Project Comparison Matrix

| Feature | BetterRobot | PyRoki | IsaacLab | Pinocchio | MjLab |
|---------|------------|--------|----------|-----------|-------|
| Model/Data separation | No | No | Partial (cfg vs env) | **Yes** | Partial |
| Config-driven design | Minimal (IKConfig) | No | **Yes** (configclass) | No | **Yes** |
| Registry pattern | Solvers only | No | **Yes** (assets, envs) | No | **Yes** (tasks) |
| Extension mechanism | None | None | **Yes** (template) | Headers + bindings | Task registration |
| Algorithm as free functions | No (Robot methods) | No (Robot methods) | N/A (managers) | **Yes** | N/A (managers) |
| Analytical Jacobians | Yes | Yes | N/A | **Yes** (all algos) | N/A |
| Collision system | Stub | **Yes** | **Yes** | **Yes** (FCL) | **Yes** |
| Dynamics | No | No | **Yes** (PhysX) | **Yes** (RNEA, ABA, CRBA) | **Yes** (MuJoCo) |
| Trajectory optimization | Stub | Yes (jaxls) | No (uses ext RL) | No (uses Crocoddyl) | No (uses ext RL) |
| Autodiff support | PyTorch autograd | JAX | N/A | Templates (CppAD, CasADi) | JAX |
| Batched computation | Manual broadcasting | JAX vmap | GPU vectorized | Loop-based + parallel | GPU vectorized |

---

## 4. Design Principles for the Restructure

Based on analysis of all projects, these principles should guide BetterRobot's restructuring:

1. **Model/Data separation** (from Pinocchio): Robot structure is immutable. Computation state is separate.

2. **Algorithms as free functions** (from Pinocchio): `forward_kinematics(model, q)` not `robot.forward_kinematics(q)`. Enables composition and testing.

3. **Config-driven composition** (from IsaacLab): Every task, solver, and cost is configured via typed dataclasses.

4. **Registry + plugin pattern** (from IsaacLab/MjLab): Components register themselves. Third parties can add new ones without modifying core.

5. **Layered API** (from all projects): Low-level (algorithms), mid-level (costs/solvers), high-level (tasks). Users choose their level.

6. **Functional residuals** (keep from BetterRobot): Pure functions are the right choice. Enhance, don't replace.

7. **Unified variable system** (fix floating-base): One variable abstraction handles joint configs, base poses, trajectories.

8. **Collision as first-class citizen** (from PyRoki/Pinocchio): Integrated with the geometry system from the start, not bolted on.

9. **Minimal dependencies, maximal composability**: Each module works independently. Users import only what they need.

10. **Documentation as architecture** (from all good projects): The directory structure should tell you the story. File names should be self-explanatory.
