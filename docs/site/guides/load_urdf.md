# Load a robot from URDF or MJCF

`br.load` dispatches by suffix:

```python
import better_robot as br

# URDF (text path or yourdfpy.URDF object)
model = br.load("panda.urdf")

# MJCF
model = br.load("g1.xml")

# Programmatic builder
from better_robot.io import ModelBuilder, build_model
b = ModelBuilder("toy")
b.add_body("base")
b.add_body("arm", mass=1.0)
b.add_revolute_y("hinge", parent="base", child="arm")
model = build_model(b.finalize())
```

Floating-base robots: pass `free_flyer=True`. BetterRobot replaces
the inserted root joint with `JointFreeFlyer` (`nq=7, nv=6`).

## Optional dependencies

URDF parsing requires `yourdfpy`; MJCF requires `mujoco`. Install with
`pip install better-robot[urdf]` or `[mjcf]`. If they're missing, the
loader raises `BackendNotAvailableError` with a pointer to the right
extra.
