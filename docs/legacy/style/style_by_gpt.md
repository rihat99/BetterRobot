# Robotics Toolbox Engineering Style Guide (ARCHIVED)

> ⚠️ **ARCHIVED — do not follow this draft.**
> The normative coding-style guide is
> [conventions/19_STYLE.md](../conventions/19_STYLE.md). This file is
> kept for historical reference only. It contradicts the current spec
> on quaternion ordering, tensor library (we are PyTorch-first, not
> NumPy), unit-suffixed identifiers (we use the `<entity>_<quantity>_<frame>`
> pattern from [13 §1.1](../conventions/13_NAMING.md) instead), and
> docstring style (NumPy, not Google).

---

# Robotics Toolbox Engineering Style Guide

This document defines the architectural rules, code-writing patterns, and quality standards for the robotics toolbox.

It is intended to keep the codebase professional, maintainable, testable, and pleasant to use.

---

## 1. Core Principles

1. The package **must have a small, intentional public API**.
2. The codebase **must separate pure robotics logic from side effects**.
3. Standard scientific Python types **should be preferred over custom container classes**.
4. Units, frames, timestamps, and conventions **must be explicit**.
5. Optional integrations such as ROS, simulators, plotting, and hardware drivers **must not pollute the core package**.
6. Imports **should stay cheap**.
7. Testing, typing, linting, documentation, and CI **are part of the product**, not afterthoughts.

---

## 2. Package Architecture

### 2.1 Recommended Layout

```text
robotics_toolbox/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── robotics_toolbox/
│       ├── __init__.py
│       ├── _version.py
│       ├── exceptions.py
│       ├── types.py
│       ├── frames.py
│       ├── units.py
│       ├── math/
│       ├── models/
│       ├── kinematics/
│       ├── dynamics/
│       ├── planning/
│       ├── control/
│       ├── estimation/
│       ├── io/
│       ├── sim/
│       ├── hardware/
│       ├── ros/
│       ├── plugins/
│       ├── cli/
│       └── data/
├── tests/
├── docs/
├── examples/
└── benchmarks/
```

### 2.2 Layering Rules

The codebase should be mentally divided into two zones:

**Core domain**
- `math`
- `models`
- `kinematics`
- `dynamics`
- `planning`
- `control`
- `estimation`

**Adapters / edges**
- `io`
- `sim`
- `hardware`
- `ros`
- `cli`
- `plugins`

### 2.3 Dependency Direction

Dependencies should flow inward.

- Core modules may depend on other core modules.
- Adapter modules may depend on core modules.
- Core modules must not depend on ROS, simulator APIs, plotting backends, device SDKs, or CLI code.

This rule keeps the library testable, reusable, and easy to install.

---

## 3. Public API Design

### 3.1 Keep the Top Level Small

The package root should expose only stable, well-supported entry points.

Good:

```python
from robotics_toolbox.models import RobotModel
from robotics_toolbox.kinematics import fk, ik
from robotics_toolbox.planning import rrt_connect
```

Bad:

```python
from robotics_toolbox.everything import *
```

### 3.2 Stability Rules

- Names exported from `robotics_toolbox.__init__` are considered public.
- Internal helpers should remain in submodules and may use a leading underscore when appropriate.
- Avoid leaking implementation details into the public surface.
- Prefer adding new public functions deliberately instead of re-exporting entire modules.

### 3.3 API Shape

Public APIs should be:
- easy to discover,
- hard to misuse,
- consistent in naming,
- explicit about units and frames,
- stable across releases.

---

## 4. Object Design: Composition over Giant Classes

### 4.1 Avoid God Objects

Do not build a single `Robot` class that does everything:
- parses URDF,
- computes FK and IK,
- talks to hardware,
- publishes ROS messages,
- logs data,
- renders plots,
- runs planners,
- stores simulation state.

That design becomes brittle and impossible to maintain.

### 4.2 Prefer Composition

Use focused components with narrow responsibilities.

Examples:
- `RobotModel`
- `IKSolver`
- `TrajectoryPlanner`
- `Controller`
- `StateEstimator`
- `Driver`
- `SimulatorSession`

### 4.3 Use Interfaces for Capabilities

Model behavior around capabilities, not inheritance pyramids.

Use `Protocol` or abstract base classes for swappable behaviors like:
- inverse kinematics,
- planning,
- control,
- estimation,
- serialization,
- device communication.

Example:

```python
from typing import Protocol
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

class IKSolver(Protocol):
    def solve(self, target_pose, seed: Vec): ...
```

Prefer narrow interfaces over large “manager” classes.

---

## 5. Data Modeling

### 5.1 Use Dataclasses for Structured Data

Use `@dataclass` for:
- configs,
- state snapshots,
- results,
- diagnostics,
- trajectory samples,
- metadata-rich values.

Example:

```python
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

@dataclass(frozen=True)
class JointState:
    q: Vec
    qd: Vec | None = None
    frame: str = "base_link"

@dataclass(frozen=True)
class IKResult:
    q: Vec
    converged: bool
    iterations: int
    residual: float
```

### 5.2 Prefer Standard Numerical Types

Prefer:
- `numpy.ndarray`
- `numpy.typing.NDArray`
- built-in Python scalars
- simple dataclasses around arrays

Avoid inventing custom vector or matrix classes unless they provide real invariants or safety that cannot be achieved otherwise.

### 5.3 When Custom Classes Are Justified

Custom classes are acceptable when they represent:
- a robot model,
- a stateful controller,
- a hardware session,
- a simulator connection,
- a trajectory object with strong invariants,
- a domain type whose misuse would otherwise be common.

If a class is only wrapping a NumPy array without adding meaningful behavior, it usually should not exist.

---

## 6. Functional Core, Imperative Shell

### 6.1 Pure Logic First

Core algorithms should be mostly pure functions.

Good:

```python
def fk(q, model) -> Pose: ...
def jacobian(q, model) -> Matrix: ...
def integrate_dynamics(state, torque, dt) -> State: ...
```

### 6.2 Keep Side Effects at the Edges

Side effects belong in adapter modules.

Examples:
- sending torque commands,
- reading encoders,
- publishing ROS messages,
- reading files,
- communicating with simulators,
- writing logs to disk.

### 6.3 Benefits

This separation improves:
- testability,
- determinism,
- reproducibility,
- portability,
- performance profiling,
- easier debugging.

---

## 7. Explicit Units, Frames, and Conventions

This is non-negotiable for robotics code.

### 7.1 Units

Every public API must be explicit about units.

Examples:
- angles: radians by default,
- distances: meters by default,
- velocities: meters per second or radians per second,
- accelerations: SI units,
- torques: N·m.

Never silently mix:
- radians and degrees,
- meters and millimeters,
- seconds and milliseconds.

### 7.2 Frames

Every spatial quantity should clearly identify its frame when ambiguity is possible.

Examples:
- `base_link`
- `tool0`
- `world`
- `odom`
- `map`

### 7.3 Timestamps and Time Base

Time-bearing data should make timestamp meaning obvious:
- wall-clock time,
- monotonic time,
- simulation time,
- synchronized sensor time.

### 7.4 Covariance and Ordering

If covariance is used, document:
- variable ordering,
- units,
- whether covariance is in local or world frame,
- whether values are full covariance or diagonal approximations.

### 7.5 No Ambiguous Arrays in Public APIs

Avoid accepting anonymous arrays where meaning is unclear.

Bad:

```python
solve(np.array([1, 2, 3]))
```

Good:

```python
solve(target_position_m=np.array([1.0, 2.0, 3.0]))
```

Even better when appropriate:

```python
@dataclass(frozen=True)
class Position3D:
    xyz_m: Vec
    frame: str
```

---

## 8. Typing Standards

### 8.1 Type Hints Are Required

All public functions, methods, and dataclasses must use type hints.

Internal code should also use type hints unless there is a strong reason not to.

### 8.2 Use Aliases for Common Types

Create shared type aliases in `types.py`.

Example:

```python
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Mat = npt.NDArray[np.float64]
```

### 8.3 Prefer Protocols for Behavioral Contracts

Use `Protocol` for pluggable strategies.

Use ABCs when you need:
- shared default implementation,
- registration,
- runtime inheritance semantics.

### 8.4 Avoid Overly Clever Types

Type hints should improve clarity, not make signatures unreadable.

Prefer simple, understandable annotations.

---

## 9. Naming Conventions

### 9.1 General Rules

- Modules: short, lowercase, descriptive.
- Classes: `PascalCase`.
- Functions and methods: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Internal helpers: prefix with `_` when private.

### 9.2 Domain Names Should Be Precise

Prefer names that expose robotics meaning.

Good:
- `joint_positions`
- `end_effector_pose`
- `body_twist`
- `covariance_world`
- `target_frame`

Bad:
- `data`
- `thing`
- `obj`
- `vals`
- `misc`

### 9.3 Naming Units in Variables

Use unit-bearing suffixes when helpful.

Examples:
- `angle_rad`
- `distance_m`
- `latency_s`
- `velocity_mps`
- `torque_nm`

This is especially encouraged at public boundaries.

---

## 10. Import Rules

### 10.1 Imports Should Stay Cheap

`import robotics_toolbox` should not import heavy optional systems by default.

Do not pull in ROS, plotting, simulators, hardware SDKs, or large datasets at package import time.

### 10.2 `__init__.py` Must Stay Lightweight

Use the package root only for:
- version,
- public re-exports,
- lightweight constants.

### 10.3 Delay Heavy Imports

Use local imports in adapter code when necessary.

Example:

```python
def connect_ros_bridge(...):
    import rclpy
    ...
```

Only do this when it materially improves optionality or import cost.

---

## 11. Dependency Management

### 11.1 Keep Core Dependencies Small

The core package should depend only on what is necessary.

### 11.2 Use Optional Extras

Optional integrations must be installable as extras.

Examples:
- `robotics_toolbox[ros]`
- `robotics_toolbox[sim]`
- `robotics_toolbox[vision]`
- `robotics_toolbox[docs]`
- `robotics_toolbox[dev]`

### 11.3 Dependency Policy

Before adding a dependency, ask:
- Is it mature?
- Is it maintained?
- Is it widely useful?
- Can the same result be achieved with existing dependencies?
- Does it belong in the core or an optional extra?

---

## 12. Plugin and Extension Design

### 12.1 Build for Extensibility

Pluggable systems are preferred over hard-coding every backend into the core.

Good plugin targets:
- robot model loaders,
- planner backends,
- controller backends,
- simulator adapters,
- hardware drivers,
- file format import/export.

### 12.2 Rules for Plugins

- Plugin interfaces should be narrow.
- The failure mode for a missing plugin should be clear.
- Plugins must not silently modify global behavior.
- Plugin registration should be explicit and documented.

---

## 13. Error Handling

### 13.1 Use a Clear Exception Hierarchy

All custom exceptions should derive from a project base class.

Example:

```python
class RoboticsError(Exception):
    """Base class for toolbox exceptions."""

class FrameError(RoboticsError):
    pass

class IKNoConvergence(RoboticsError):
    pass

class DriverTimeout(RoboticsError):
    pass
```

### 13.2 Raise Exceptions for Failures

Use exceptions for actual failures, not for regular control flow.

### 13.3 Result Objects for Algorithm Output

When an algorithm can validly succeed or fail in nuanced ways, prefer a structured result object.

Example fields:
- `converged`
- `iterations`
- `residual`
- `status_code`
- `message`

### 13.4 Error Messages Must Help the User Recover

Bad:

```python
raise ValueError("Invalid input")
```

Good:

```python
raise ValueError(
    "Expected joint vector of shape (6,), got shape (5,) for robot UR5e."
)
```

---

## 14. Warnings and Deprecation

### 14.1 Do Not Break Public APIs Abruptly

Use warnings before removal.

### 14.2 Deprecation Policy

When deprecating something:
- emit a deprecation warning,
- explain what to use instead,
- document the expected removal version,
- keep the migration path simple.

Example:

```python
import warnings

warnings.warn(
    "ik_solve() is deprecated; use ik() instead. Will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

## 15. Logging Standards

### 15.1 Use the Logging Module

Library code must use the standard logging system.

```python
import logging

logger = logging.getLogger(__name__)
```

### 15.2 Do Not Print from Library Code

`print()` is for:
- CLI tools,
- examples,
- explicit user-facing scripts.

It is not for library internals.

### 15.3 Log Levels

- `debug`: internal diagnostics
- `info`: notable but expected milestones
- `warning`: recoverable problem or suspicious condition
- `error`: operation failed
- `exception`: operation failed with traceback in exception handlers

### 15.4 Logging Must Not Leak Sensitive Data

Be careful with:
- hardware serial numbers,
- file paths if sensitive,
- network credentials,
- raw calibration secrets,
- personally identifiable user data.

---

## 16. Documentation Standards

### 16.1 Every Public API Needs Documentation

Public functions, methods, classes, and modules must have docstrings.

### 16.2 Prefer Numpydoc-Style Docstrings

Example:

```python
def fk(q: Vec, model: RobotModel) -> Pose:
    """Compute forward kinematics.

    Parameters
    ----------
    q : Vec
        Joint positions in radians.
    model : RobotModel
        Robot kinematic model.

    Returns
    -------
    Pose
        End-effector pose expressed in the base frame.
    """
```

### 16.3 Public Docs Must State

- parameter meanings,
- units,
- frames,
- expected shapes,
- return value semantics,
- possible exceptions,
- examples when helpful.

### 16.4 Examples Matter

Prefer small, executable examples over vague prose.

---

## 17. Testing Standards

### 17.1 Testing Is Mandatory

Every feature should ship with tests appropriate to its risk.

### 17.2 Test Categories

The project should contain:
- unit tests,
- property-based tests,
- integration tests,
- regression tests,
- performance benchmarks for hot paths.

### 17.3 Unit Tests

Use for:
- math helpers,
- transforms,
- parser behavior,
- controller computations,
- shape validation,
- error conditions.

### 17.4 Property-Based Tests

Use property-based testing for mathematical invariants.

Examples:
- transform inverse correctness,
- compose/invert consistency,
- quaternion normalization properties,
- serialization round-trips,
- path constraints staying within bounds.

### 17.5 Integration Tests

Use for:
- simulator bridges,
- ROS adapters,
- file I/O,
- hardware abstraction layers.

Hardware tests should be separated so they do not break normal CI.

### 17.6 Regression Tests

When a bug is fixed, add a test that would have caught it.

### 17.7 Test Readability

A test should make clear:
- what scenario is being exercised,
- what behavior is expected,
- why the result matters.

---

## 18. CI and Quality Gates

Every pull request should run automated checks.

Minimum recommended checks:
- formatting,
- linting,
- type checking,
- test suite,
- docs build,
- packaging build.

Recommended tools:
- Ruff
- mypy
- pytest
- Hypothesis
- Sphinx

A change that fails quality gates is not ready to merge.

---

## 19. Numerical and Scientific Code Practices

### 19.1 Validate Shapes Early

At public boundaries, validate array shapes and assumptions.

### 19.2 Be Explicit About Numerical Behavior

Document or encode:
- tolerances,
- convergence criteria,
- iteration limits,
- singularity behavior,
- failure modes,
- assumptions about normalization.

### 19.3 Avoid Silent Numerical Magic

Do not silently:
- normalize inputs unless clearly documented,
- clamp values without telling the caller,
- reorder joints unexpectedly,
- guess frames.

### 19.4 Determinism

When randomness is involved, support deterministic seeding.

Example targets:
- planners,
- sampling-based algorithms,
- randomized tests,
- synthetic data generation.

---

## 20. Performance Rules

### 20.1 Optimize After Measurement

Do not micro-optimize blindly.

Measure first.

### 20.2 Focus on Hot Paths

Typical hot paths:
- kinematics kernels,
- collision queries,
- dynamics integration,
- planning inner loops,
- estimation updates.

### 20.3 Keep APIs Friendly Even When Internals Are Fast

Readable public APIs are usually more valuable than exposing internal performance hacks.

### 20.4 Add Benchmarks for Critical Paths

If performance matters, benchmark it and track regressions.

---

## 21. Resource and File Handling

### 21.1 Package Resources Must Be Loaded Robustly

Use package-resource mechanisms for bundled assets.

Examples:
- URDFs,
- meshes,
- calibration templates,
- sample trajectories,
- reference configs.

### 21.2 Avoid Fragile Relative Paths

Do not assume the current working directory.

### 21.3 Separate Parsing from Domain Logic

- `io/` should parse files and convert them into domain objects.
- Core modules should operate on structured in-memory data.

---

## 22. Configuration Standards

### 22.1 Prefer Explicit Config Objects

Use typed configuration objects over large untyped dictionaries.

Bad:

```python
def run_controller(config: dict):
    ...
```

Good:

```python
@dataclass(frozen=True)
class ControllerConfig:
    kp: float
    kd: float
    max_torque_nm: float
```

### 22.2 Validate Configs Early

Configuration errors should fail fast with clear messages.

---

## 23. Packaging Standards

### 23.1 Use Modern Packaging

The project must use `pyproject.toml`.

### 23.2 Use a `src/` Layout

All package code should live under `src/`.

### 23.3 Declare Supported Python Versions

Set `requires-python` explicitly and maintain a clear support policy.

### 23.4 Keep Optional Features Optional

Optional integrations belong in optional dependency groups, not core dependencies.

---

## 24. Versioning and Releases

### 24.1 Use Semantic Versioning

- MAJOR: incompatible API changes
- MINOR: backward-compatible features
- PATCH: backward-compatible fixes

### 24.2 Release Notes Must Be Clear

Each release should document:
- added features,
- behavior changes,
- fixes,
- deprecations,
- removals,
- migration notes.

### 24.3 Breaking Changes Need Migrations

If a public API changes, document how users should update their code.

---

## 25. Review Standards

### 25.1 What Reviewers Should Look For

Reviewers should evaluate:
- architecture,
- API clarity,
- test coverage,
- type quality,
- naming,
- documentation,
- failure handling,
- maintainability.

### 25.2 Questions to Ask in Review

- Does this belong in the core or at the edge?
- Does this create unnecessary coupling?
- Is the public API easy to understand?
- Are units and frames explicit?
- Are failure modes documented?
- Is the new dependency justified?
- Is this easy to test?
- Will this still make sense in a year?

---

## 26. Anti-Patterns

Avoid the following:

- giant all-in-one classes,
- `from module import *`,
- hidden side effects in math code,
- ambiguous units and frames,
- custom numerical wrappers with no value,
- printing from library internals,
- undocumented public APIs,
- untyped interfaces,
- circular imports,
- broad `except Exception:` blocks without reason,
- silent fallback behavior that masks errors,
- adding heavy dependencies to the core for niche features.

---

## 27. Example Design Pattern

A preferred pattern is:

1. model domain data with typed dataclasses,
2. define capabilities with protocols,
3. implement algorithms in pure functions or slim service objects,
4. isolate I/O and integration logic in adapters,
5. expose only stable public entry points.

Example:

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

@dataclass(frozen=True)
class JointState:
    q: Vec
    qd: Vec | None = None

@dataclass(frozen=True)
class IKResult:
    q: Vec
    converged: bool
    iterations: int
    residual: float

class IKSolver(Protocol):
    def solve(self, target_pose, seed: Vec) -> IKResult: ...


def fk(q: Vec, model) -> object:
    """Pure kinematics computation."""
    ...
```

---

## 28. Enforcement Summary

The most important rules are:

1. **Pure core, messy edges.**
2. **Small public API.**
3. **Composition over giant classes.**
4. **Dataclasses for structured data.**
5. **Protocols for interchangeable behavior.**
6. **Standard arrays first.**
7. **Explicit units, frames, timestamps, and conventions.**
8. **Optional integrations stay optional.**
9. **No `print()` in library code.**
10. **Type hints, tests, docs, and CI are required.**

If a design decision conflicts with these rules, the burden is on the change to justify itself.

---

## 29. Recommended Tooling Baseline

Recommended baseline stack:

- Packaging: `pyproject.toml`
- Test runner: `pytest`
- Property testing: `hypothesis`
- Linting: `ruff`
- Type checking: `mypy`
- Docs: `sphinx` + numpydoc-style docstrings
- Benchmarks: `pytest-benchmark` or dedicated benchmark suite

Suggested optional extras:
- `dev`
- `docs`
- `ros`
- `sim`
- `vision`
- `hardware`

---

## 30. Final Standard

This toolbox should feel:
- explicit,
- modular,
- predictable,
- well-documented,
- easy to test,
- easy to extend,
- hard to misuse.

Professional robotics software is rarely defined by cleverness.
It is defined by clarity, discipline, and consistency.
