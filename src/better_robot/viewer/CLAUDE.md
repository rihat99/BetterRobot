# viewer/ — Viser Visualization Helpers

Thin wrapper around [viser](https://viser.studio) for interactive robot visualization.

## Public API

```python
from better_robot.viewer import Visualizer, wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict
# or:
from better_robot import Visualizer
```

## `Visualizer` (`visualizer.py`)

High-level viser wrapper. Manages the server, URDF mesh, draggable target handles, and GUI.

```python
vis = Visualizer(urdf, model, port=8080, floating_base=False)
```

**Setup methods:**
| Method | Description |
|--------|-------------|
| `add_target(link_name, scale=0.12)` | Add draggable transform-control handle |
| `add_grid(size=4.0)` | Add ground-plane grid |
| `add_timing_display()` | Add read-only elapsed-time number in GUI |
| `add_restart_button()` | Add Restart button; consumed via `restart_requested` |

**Per-frame API:**
| Method | Description |
|--------|-------------|
| `restart_requested` | Property; `True` once after button click, then `False` |
| `reset_targets(model, cfg, base_pose=None)` | Snap handles to robot's FK poses |
| `get_targets()` | Returns `{link_name: (7,) SE3 tensor}` |
| `update(cfg, base_pose=None)` | Update URDF mesh to new joint config |
| `set_timing(elapsed_ms)` | EMA-smoothed timing display update |

**Typical loop (fixed-base):**
```python
vis = Visualizer(urdf, model)
vis.add_target("panda_hand")
vis.add_restart_button()
vis.reset_targets(model, cfg)

while True:
    if vis.restart_requested:
        cfg = model._default_cfg.clone()
        vis.reset_targets(model, cfg)
    cfg = solve_ik(model, targets=vis.get_targets(), initial_cfg=cfg)
    vis.update(cfg)
```

**Typical loop (floating-base):**
```python
vis = Visualizer(urdf, model, floating_base=True)
vis.add_grid()
vis.add_target("left_rubber_hand")
vis.add_timing_display()
vis.add_restart_button()
vis.reset_targets(model, cfg, base_pose)

while True:
    if vis.restart_requested:
        cfg, base_pose = initial_cfg, initial_base
        vis.reset_targets(model, cfg, base_pose)
    t0 = time.perf_counter()
    base_pose, cfg = solve_ik(model, targets=vis.get_targets(), initial_base_pose=base_pose, ...)
    vis.set_timing((time.perf_counter() - t0) * 1000)
    vis.update(cfg, base_pose=base_pose)
```

## Helper Functions (`helpers.py`)

### `wxyz_pos_to_se3(wxyz, pos) → (7,) Tensor`

Convert viser handle pose to SE3 `[tx, ty, tz, qx, qy, qz, qw]`.

```python
se3 = wxyz_pos_to_se3(handle.wxyz, handle.position)
```

### `qxyzw_to_wxyz(q) → tuple`

Convert PyPose quaternion `[qx, qy, qz, qw]` tensor to viser `(w, x, y, z)` tuple.

```python
handle.wxyz = qxyzw_to_wxyz(fk[link_idx, 3:7].detach())
```

### `build_cfg_dict(model, cfg) → dict[str, float]`

Build `joint_name → float` dict for `ViserUrdf.update_cfg`. Filters to actuated joints (revolute, continuous, prismatic) in BFS order.

```python
urdf_vis.update_cfg(build_cfg_dict(model, cfg))
```

## Quaternion Convention

PyPose SE3 stores quaternion as `[qx, qy, qz, qw]` (scalar last).
Viser expects `(w, x, y, z)` (scalar first). `qxyzw_to_wxyz` handles the swap.
