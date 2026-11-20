# Model and io

A `Model` is your robot as a frozen set of tensors: its joints, bodies and frames, their placements, inertias and
limits. You never change a model in place; you make a new one. `io` builds models from URDF or MJCF files, or from
a robot you write in code.

The names you will use:

| Name | What it does |
|---|---|
| `br.load(path)` | reads a URDF or MJCF file into a `Model` |
| `br.Model` | the robot: `nq`, `nv`, `joint_names`, `frame_names`, `q_neutral`, limits, `frame_id(name)` |
| `br.JointFreeFlyer` | a floating base: pass `root_joint=br.JointFreeFlyer()` to `load` |
| `br.ModelBuilder` | writes a robot in code; `io.build_model` turns it into a `Model` |
| `io.load_geometry(path, model)` | the visual and collision shapes of the file |
| `model.to(device, dtype)`, `model.with_values(...)` | a copy on another device, or with some tensors replaced |

## Load a robot

```python
import torch
from robot_descriptions import panda_description

import better_robot as br

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
print(model.name, "nq =", model.nq, "nv =", model.nv)
print(model.joint_names[:4])
print("hand frame id:", model.frame_id("panda_hand"))
print(model.upper_position_limit)
```
```output
panda nq = 8 nv = 8
('universe', 'root_joint', 'panda_joint1', 'panda_joint2')
hand frame id: 10
tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400],
       dtype=torch.float64)
```

`nq` is the size of a configuration and `nv` the size of a velocity. Joints, bodies and frames have names and
integer ids; every function takes either. A file ending in `.xml` or `.mjcf` loads through MuJoCo's compiler.

## Add a floating base

A humanoid or a mobile robot moves its base freely. Its base is one more joint, a `JointFreeFlyer`, whose
configuration is a pose `[x, y, z, qx, qy, qz, qw]` and whose velocity has six entries. So `nq` and `nv` differ.

```python
from robot_descriptions import g1_description

g1 = br.load(g1_description.URDF_PATH, root_joint=br.JointFreeFlyer(), dtype=torch.float64)
print("nq =", g1.nq, "nv =", g1.nv)
print(g1.q_neutral[:7])
```
```output
nq = 36 nv = 35
tensor([0., 0., 0., 0., 0., 0., 1.], dtype=torch.float64)
```

## Write a robot in code

`ModelBuilder` names bodies and joints; `io.build_model` turns the result into a model. Here is a one-link
pendulum that swings about `y`:

```python
builder = br.ModelBuilder("pendulum")
builder.add_body("base", mass=0.0)
builder.add_body("link", mass=2.0, com=torch.tensor([0.3, 0.0, 0.0]), inertia=torch.diag(torch.tensor([0.2, 0.3, 0.4])))
builder.add_revolute("shoulder", axis=(0.0, 1.0, 0.0), parent="base", child="link", lower=-2.0, upper=2.0)
pendulum = br.io.build_model(builder.finalize(), dtype=torch.float64)
print(pendulum.nq, pendulum.joint_names)
```
```output
1 ('universe', 'root_joint', 'shoulder')
```

See also: [Kinematics](kinematics.md) for what you do with a model.
