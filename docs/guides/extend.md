# Extend BetterRobot

Five kinds of brick plug into BetterRobot: a joint kind, a term, a parser function, a viewer layer and a solver.
None of them is registered anywhere (record 0025): you build the object and pass it where it is used. Each section
below gives the contract and a program that runs; the docs tests run every fence of this page, in order, in one
namespace. The normative rules are in [contracts/08-extension-points](../contracts/08-extension-points.md).

## A joint kind

A joint kind subclasses the abstract `JointModel` as a final frozen dataclass (records 0013 and 0025); the built-in
kinds in `src/better_robot/model/joints/` (`revolute.py`, `prismatic.py`, `planar.py`, ...) are the reference
implementations, one file per kind.

| Member | Contract |
|--------|----------|
| `nq`, `nv` | class variables: the widths of the joint's `q` and `v` slices |
| `manifold` | class variable: the `JointManifold` whose `integrate`/`difference` step the coordinates |
| `axis` | the joint axis in the joint frame as a tuple of floats (a dataclass field), or the class variable `None` |
| `placement(q)` | `(B..., nq) -> (B..., 7)` pose of the child in the joint frame |
| `motion_subspace(q)` | `(B..., nq) -> (B..., 6, nv)`; the child's twist in its own frame is `S(q) @ v` |
| `neutral()` | `(nq,)` configuration with the identity placement |
| `random(generator, *, lower, upper)` | `(nq,)` configuration inside the limits |

A manifold is a `JointManifold(name=, nq=, nv=, integrate=, difference=)` whose maps are the group exponential and
logarithm, so that `difference(q0, integrate(q0, v)) == v` and the derivative of `placement(integrate(q, t v))` at
`t = 0` is `motion_subspace(q) @ v`. `better_robot.testing.check_joint_model(joint, dtype=, generator=)` checks both
and the shapes; run it on every new kind. Pass the kind to `ModelBuilder.add_joint(kind=...)`: the built model keeps
that exact object, forward kinematics evaluates it through `placement`, and the viewer draws it as a sphere.

```python
from dataclasses import dataclass
from typing import ClassVar, final

import torch

import better_robot as br
from better_robot.io import build_model
from better_robot.lie import se3
from better_robot.model import JointManifold, JointModel
from better_robot.testing import check_joint_model

LINE = JointManifold(name="line", nq=1, nv=1, integrate=lambda q, v: q + v, difference=lambda q0, q1: q1 - q0)


@final
@dataclass(frozen=True)
class JointDrop(JointModel):
    """A translation along -z of the joint frame."""

    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[JointManifold] = LINE
    axis: ClassVar[None] = None

    def placement(self, q):  # (B..., 1) -> (B..., 7)
        return se3.from_translation(torch.cat((q.new_zeros(*q.shape[:-1], 2), -q), dim=-1))

    def motion_subspace(self, q):  # (B..., 1) -> (B..., 6, 1)
        return q.new_tensor([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]).expand(*q.shape[:-1], 6).unsqueeze(-1)

    def neutral(self):
        return torch.zeros(1)

    def random(self, generator, *, lower, upper):
        return lower + (upper - lower) * torch.rand(1, generator=generator)


check_joint_model(JointDrop(), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
builder = br.ModelBuilder("drop")
builder.add_body("base")
builder.add_body("tip")
builder.add_joint("drop", kind=JointDrop(), parent="base", child="tip", lower=0.0, upper=0.5)
model = build_model(builder.finalize())
print(br.forward_kinematics(model, torch.tensor([0.2])).frame_pose_of("tip"))
```
```output
tensor([ 0.0000,  0.0000, -0.2000,  0.0000,  0.0000,  0.0000,  1.0000])
```

Add a Pinocchio parity test when Pinocchio has the same joint (see [Test suite](../reference/tests.md#pinocchio-parity)).

## A term

A term is a plain function: it reads named variables and data by its argument names and returns a residual with the
problem's leading batch axes and a flat event axis. [Write a term](write-a-term.md) is the full guide,
with an analytic Jacobian. Robot-aware terms go in `better_robot.terms`; `better_robot.optim` imports torch and
`better_robot.exceptions` only, and must not learn about robots.

```python
from better_robot.optim import LevenbergMarquardt, Problem, solve


def to_target(x, target):  # x: (B, 3), target: (3,) -> (B, 3)
    return x - target


x0 = torch.zeros(4, 3, dtype=torch.float64)
problem = Problem({"x": x0}, data={"target": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)})
problem.residual("to_target", to_target)
print(solve(problem, LevenbergMarquardt(), iterations=10).x["x"][0])
```
```output
tensor([1.0000, 2.0000, 3.0000], dtype=torch.float64)
```

## A parser function

A parser is a function from a path to an `IRModel`, the intermediate model of `better_robot.io`; pass its result to
`build_model`, as `load` does for URDF and MJCF. `load` knows those two formats by suffix, and your format lives in
your function. [Loading and parsing](../concepts/io.md) describes the intermediate model.

```python
import tempfile
from pathlib import Path

from better_robot.io import IRBody, IRJoint, IRModel
from better_robot.model import JointRevolute


def load_chain(path):
    """One revolute joint per line, `name ax ay az length`: each link `length` metres above its parent."""
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    bodies = [IRBody(name="base"), *(IRBody(name=f"{name}_link", mass=1.0) for name, *_ in rows)]
    joints = [
        IRJoint(
            name=name,
            parent_body=parent.name,
            child_body=child.name,
            joint=JointRevolute(axis=(float(ax), float(ay), float(az))),
            origin=se3.from_translation(torch.tensor([0.0, 0.0, float(length)])),
        )
        for (name, ax, ay, az, length), parent, child in zip(rows, bodies, bodies[1:])
    ]
    return IRModel(name=Path(path).stem, bodies=bodies, joints=joints)


with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "arm.chain"
    path.write_text("shoulder 0 0 1 0.1\nelbow 0 1 0 0.3\n")
    model = build_model(load_chain(path))
print(model.nq, model.joint_names)
```
```output
2 ('universe', 'root_joint', 'shoulder', 'elbow')
```

## A viewer layer

A layer subclasses `better_robot.viewer.Layer`. It holds its own `(T, ...)` arrays, creates scene nodes in
`build(scene, root)` (register each handle with `_track` so visibility and disposal work), and writes one step in
`set_step(step)`. Override `build_gui`, `apply_theme`, `world_points` and `dispose` only when needed. See
[Viewer](../concepts/viewer.md) for the built-in layers.

```python
import numpy as np

from better_robot.viewer import Layer


class BouncingBall(Layer):
    """One red point at `heights: (T,)` metres above the origin."""

    def __init__(self, name, heights):
        super().__init__(name, nsteps=len(heights))
        self.heights = heights

    def build(self, scene, root):
        self.ball = self._track(scene.add_point_cloud(f"{root}/ball", np.zeros((1, 3)), (255, 0, 0), point_size=0.05))

    def set_step(self, step):
        self.ball.points = np.array([[0.0, 0.0, self.heights[step]]])


viewer = br.Viewer()
ball = BouncingBall("ball", np.abs(np.sin(np.linspace(0.0, 6.0, 60))))
viewer.add([ball])
viewer.show(30)
print(ball.ball.points)
```
```output
[[0.        0.        0.0906207]]
```

## A solver

A solver subclasses `better_robot.optim.Solver` with `init(problem, x) -> State` and
`step(problem, state) -> State`. Both are pure: every step returns a new `State` and writes nothing into the problem.
`solve` and `run` call `init` once and `step` up to the iteration budget; `State.done` stops a batch element early.

```python
from better_robot.optim import Solver, State


class GradientDescent(Solver):
    """`x <- project(x - rate * d loss / d x)`, the smallest solver."""

    def __init__(self, rate):
        self.rate = rate

    def init(self, problem, x):
        loss = problem.evaluate(x)[0]
        return State(x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), solver=type(self))

    def step(self, problem, state):
        x = {name: value.detach().requires_grad_(True) for name, value in state.x.items()}
        gradients = torch.autograd.grad(problem.evaluate(x)[0].sum(), list(x.values()))
        with torch.no_grad():
            moved = problem.project({name: x[name] - self.rate * grad for name, grad in zip(x, gradients)})
            loss = problem.evaluate(moved)[0]
        return State(x=moved, loss=loss, done=state.done, solver=type(self), iteration=state.iteration + 1)


print(solve(problem, GradientDescent(rate=0.25), iterations=40).x["x"][0])
```
```output
tensor([1.0000, 2.0000, 3.0000], dtype=torch.float64)
```

## Rules for every extension

- Imports point down the package graph; `tests/contract/test_layer_dependencies.py` enforces it.
  Nothing in the computational core imports `viewer`.
- Optional dependencies are imported lazily inside the function that needs them and raise
  `OptionalDependencyError` when missing (`tests/contract/test_optional_imports.py`).
- Add an abstraction only when at least two in-tree callers need it.
- Follow the [design contracts](../contracts/00-index.md) for names, shapes and validation; the per-extension-point
  rules are in [08-extension-points](../contracts/08-extension-points.md).

## Where next

- [Run the tests](run-the-tests.md) and the [test suite](../reference/tests.md): the parity, contract and docs tests
  every extension must pass.
- [Write a term](write-a-term.md): a term with an analytic Jacobian, step by step.
