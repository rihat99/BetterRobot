# models/ — Robot Model and URDF Loading

Immutable robot kinematic structure. Depends only on `math/`.

## Public API

```python
from better_robot.models import RobotModel, JointInfo, LinkInfo, load_urdf
import yourdfpy

urdf = yourdfpy.URDF.load("robot.urdf")
model = load_urdf(urdf)           # yourdfpy.URDF → RobotModel
idx = model.link_index("panda_hand")
fk = model.forward_kinematics(cfg)
chain = model.get_chain(idx)
```

## Files

### `robot_model.py` — `RobotModel`

Central class. All FK data is set once in `from_urdf()` and never mutated.

**Immutability guard**: `RobotModel` raises `AttributeError` on any attribute assignment after `from_urdf()` completes. Implemented via `__setattr__` checking `_frozen`. Once `_frozen = True` is set at the end of `from_urdf()`, all further mutation attempts raise an error. This ensures the model is truly immutable at runtime.

```python
model._frozen  # True after from_urdf()
model.some_attr = x  # → raises AttributeError: RobotModel is immutable
```

**Public interface:**
| Member | Type | Description |
|--------|------|-------------|
| `joints` | `JointInfo` | Joint metadata (names, limits, axes, origins) |
| `links` | `LinkInfo` | Link metadata (names, parent joint indices) |
| `forward_kinematics(cfg, base_pose=None)` | method | Thin delegate to `algorithms/kinematics/forward.py`; returns `(*batch, N_links, 7)` SE3 poses |
| `link_index(name)` | method | Returns int index; raises `ValueError` if not found |
| `get_chain(link_idx)` | method | Thin delegate to `algorithms/kinematics/chain.py`; returns actuated joint indices root→EE |
| `_default_cfg` | `Tensor` | Midpoint of joint limits (set by `from_urdf`) |
| `from_urdf(urdf)` | classmethod | Construct from `yourdfpy.URDF` |

**Private FK attributes** (set once in `from_urdf`, read by `algorithms/kinematics/`):
- `_fk_joint_parent_link`, `_fk_joint_child_link` — parent/child link indices per joint
- `_fk_joint_origins` — fixed SE3 transforms from parent to joint frame
- `_fk_joint_types` — `"revolute"`, `"continuous"`, `"prismatic"`, `"fixed"`, etc.
- `_fk_cfg_indices` — index into `cfg` tensor for each joint (`-1` if fixed)
- `_fk_joint_axes` — joint axis in local frame
- `_fk_joint_order` — BFS traversal order
- `_root_link_idx` — index of root link

### `joint_info.py` — `JointInfo` dataclass

Stores per-joint arrays (all tensors):
- `names`: list of joint names (BFS order, all joints including fixed)
- `lower_limits`, `upper_limits`: `(num_actuated_joints,)` tensors
- `twists`: `(num_joints, 6)` screw axes
- `parent_transforms`: `(num_joints, 7)` fixed SE3 from parent frame to joint origin
- `num_actuated_joints`: int

### `link_info.py` — `LinkInfo` dataclass

- `names`: list of link names (BFS order)
- `parent_joint_indices`: int list (`-1` for root)
- `num_links`: int

### `parsers/urdf.py` — `load_urdf`

```python
def load_urdf(urdf: yourdfpy.URDF) -> RobotModel
```

Public entry point. Calls `RobotURDFParser.parse()` internally.

### `parsers/_urdf_impl.py` — `RobotURDFParser`

Internal. Converts `yourdfpy.URDF` joints/links into `JointInfo` and `LinkInfo` via BFS traversal.
