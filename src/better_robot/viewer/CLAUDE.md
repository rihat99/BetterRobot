# viewer/ — Viser Visualization Helpers

## Public API

| Function | Description |
|----------|-------------|
| `wxyz_pos_to_se3(wxyz, pos)` | Convert viser handle pose to SE3 tensor |
| `qxyzw_to_wxyz(q)` | Convert PyPose quaternion to viser tuple |
| `build_cfg_dict(robot, cfg)` | Build joint_name→angle dict for ViserUrdf |

## Helper functions (`_helpers.py`)

```python
from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict

# Viser handle → SE3 [tx, ty, tz, qx, qy, qz, qw]
se3 = wxyz_pos_to_se3(handle.wxyz, handle.position)

# PyPose [qx, qy, qz, qw] tensor → viser (w, x, y, z) tuple
handle.wxyz = qxyzw_to_wxyz(fk[link_idx, 3:7].detach())

# Build dict for ViserUrdf.update_cfg (actuated joints only, BFS order)
urdf_vis.update_cfg(build_cfg_dict(robot, cfg))
```

## Quaternion convention

PyPose SE3 stores quaternion as `[qx, qy, qz, qw]` (scalar last).
Viser expects `(w, x, y, z)` (scalar first). `qxyzw_to_wxyz` handles the swap.

## `Visualizer` class

Also exported: `Visualizer` — a higher-level viser wrapper (see `_visualizer.py`).
