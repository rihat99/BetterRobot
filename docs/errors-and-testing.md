# Errors and testing

Every error BetterRobot raises is a class of `better_robot.exceptions`, under one root, `BetterRobotError`. Each
class also derives from the builtin you would expect, so `except ValueError` and `except KeyError` keep working.
`lie.checks` checks the tensor inputs of your own functions the same way, and `testing` holds checks for your own
test suite.

The names you will use:

| Name | What it does |
|---|---|
| `BetterRobotError` | the root: catch it to catch any BetterRobot error |
| `ShapeError`, `DtypeMismatchError`, `DeviceMismatchError` | a tensor of the wrong shape, dtype or device (`ValueError`) |
| `UnknownNameError` | a joint, frame or body name the model does not have (`KeyError`); the message lists valid names |
| `ModelError`, `ArgumentError`, `OptionalDependencyError` | a bad robot file, a bad argument, a missing optional package |
| `lie.checks.check_tensor`, `check_index_tensor`, `check_dtype`, `check_broadcast` | check a tensor input, an integer table, a dtype or batch axes, with these errors and messages |
| `testing.check_joint_model(joint, dtype=, generator=)` | checks a joint kind of your own against the joint contract |
| `testing.FakeServer` | a viser server without sockets, for testing viewer code |

## Errors say what went wrong

```python
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot.exceptions import BetterRobotError, ShapeError, UnknownNameError

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
try:
    model.frame_id("panda_head")
except UnknownNameError as error:
    print(type(error).__name__, isinstance(error, KeyError))

try:
    br.forward_kinematics(model, torch.zeros(model.nq + 1, dtype=torch.float64))
except ShapeError as error:
    print(type(error).__name__, isinstance(error, BetterRobotError))
```
```output
UnknownNameError True
ShapeError True
```

## Check the inputs of your own functions

A library built on BetterRobot checks its tensor inputs with `lie.checks`, so its errors read like BetterRobot's.

```python
from better_robot.lie import checks


def scale(points: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """``points (..., 3)`` times ``factor (...)``, the inputs checked as BetterRobot checks its own."""
    checks.check_tensor("points", points, shape=(3,))
    checks.check_tensor("factor", factor, shape=(), dtype=points.dtype, device=points.device, owner="points")
    checks.check_broadcast(points=points.shape[:-1], factor=factor.shape)
    return points * factor[..., None]


try:
    scale(torch.zeros(4, 3), torch.ones(5))
except ShapeError as error:
    print(error)
```
```output
points has batch axes (4,) and factor (5,), which do not broadcast; give both the same leading axes (size 1 repeats)
```

## Check a joint kind of your own

A joint kind is a small frozen class under `model.JointModel`. `check_joint_model` runs the checks every built-in
kind passes: shapes, the identity at the neutral configuration, and that `integrate` and `difference` agree.

```python
from better_robot import testing
from better_robot.model import JointRevolute

testing.check_joint_model(JointRevolute(axis=(0.0, 0.0, 1.0)), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
print("passed")
```
```output
passed
```

The pytest plugin `better_robot.testing.no_skips` turns every skip and xfail into a failure. Add it to
`pytest_plugins` in your `conftest.py` if you want the same rule.

See also: [Model and io](model-and-io.md) for the joint kinds that ship.
