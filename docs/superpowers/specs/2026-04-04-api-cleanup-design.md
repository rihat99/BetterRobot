# API Cleanup Design: Unified IK, IKConfig, Viewer Helpers

**Goal:** Replace three IK functions with one, expose all tuning parameters through a single `IKConfig` dataclass, and extract shared viser utilities into `viewer/_helpers.py`.

**Architecture:** Single `solve_ik` dispatches internally on `initial_base_pose` — fixed-base path uses the existing `Problem`/LM solver; floating-base path uses `_FloatingBaseIKModule`. `IKConfig` replaces the `weights` dict and hardcoded `pos_weight`/`ori_weight`. Viewer helpers eliminate duplication across both examples.

**Tech Stack:** PyTorch, PyPose, viser

---

## Files Changed / Created

| File | Action | What changes |
|------|--------|-------------|
| `src/better_robot/tasks/_config.py` | Create | `IKConfig` dataclass |
| `src/better_robot/tasks/_ik.py` | Rewrite | Unified `solve_ik`; drop `solve_ik_multi` |
| `src/better_robot/tasks/_floating_base_ik.py` | Modify | Remove `solve_ik_floating_base`; keep `_FloatingBaseIKModule` |
| `src/better_robot/tasks/__init__.py` | Modify | Export `IKConfig`, `solve_ik` only; drop old exports |
| `src/better_robot/__init__.py` | Modify | Export `IKConfig`, `solve_ik` only; drop old exports; update docstring |
| `src/better_robot/viewer/_helpers.py` | Create | `wxyz_pos_to_se3`, `qxyzw_to_wxyz`, `build_cfg_dict` |
| `src/better_robot/viewer/__init__.py` | Modify | Export helpers |
| `examples/01_basic_ik.py` | Modify | New API + import helpers |
| `examples/02_g1_ik.py` | Modify | New API + import helpers; add `--profile` flag |
| `tests/test_ik.py` | Modify | Update all tests to new API |
| `tests/test_imports.py` | Modify | Update smoke tests |
| `src/better_robot/tasks/CLAUDE.md` | Modify | Document unified API |
| `src/better_robot/viewer/CLAUDE.md` | Create | Document helpers |
| `CLAUDE.md` | Modify | Update IK usage pattern + floating-base section |

---

## Component Details

### 1. `IKConfig` (`tasks/_config.py`)

```python
from dataclasses import dataclass

@dataclass
class IKConfig:
    pos_weight: float = 1.0    # position component of pose error
    ori_weight: float = 0.1    # orientation component of pose error
    pose_weight: float = 1.0   # overall pose cost scale
    limit_weight: float = 0.1  # soft joint limit penalty
    rest_weight: float = 0.01  # pull toward default config
```

All five fields replace: the `weights={"pose", "limits", "rest"}` dict pattern and the
hardcoded `pos_weight=1.0, ori_weight=0.1` inside `_ik.py` and `_floating_base_ik.py`.

### 2. Unified `solve_ik` (`tasks/_ik.py`)

```python
def solve_ik(
    robot: Robot,
    targets: dict[str, torch.Tensor],        # {link_name: (7,) SE3}
    cfg: IKConfig = IKConfig(),
    initial_cfg: torch.Tensor | None = None,
    initial_base_pose: torch.Tensor | None = None,  # None → fixed base
    max_iter: int = 100,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
```

**Dispatch logic:**

- `initial_base_pose is None` → fixed-base path:
  - Builds one `CostTerm(pose_residual(...))` per target (same as old `solve_ik_multi`)
  - Adds shared `limit_residual` and `rest_residual` `CostTerm`s
  - Uses `Problem` + `SOLVER_REGISTRY["lm"]`
  - Returns `Tensor` shape `(num_actuated_joints,)`

- `initial_base_pose is not None` → floating-base path:
  - Constructs `_FloatingBaseIKModule` from `_floating_base_ik.py`
  - Runs PyPose LM loop with joint clamping and quaternion renorm
  - Returns `tuple[Tensor, Tensor]` → `(base_pose (7,), cfg (n,))`

The solver selector (`"lm"/"gn"/"adam"/"lbfgs"`) is removed — always LM.
The collision parameters (`robot_coll`, `world_coll`) are removed — stubs not worth carrying.

**Passing `IKConfig` to `pose_residual`:**
```python
functools.partial(
    pose_residual,
    robot=robot,
    target_link_index=link_idx,
    target_pose=target_pose,
    pos_weight=cfg.pos_weight,
    ori_weight=cfg.ori_weight,
)
# CostTerm weight = cfg.pose_weight
```

### 3. `_FloatingBaseIKModule` (`tasks/_floating_base_ik.py`)

Remove `solve_ik_floating_base`. Keep `_FloatingBaseIKModule` unchanged — it becomes a
private implementation detail called by `solve_ik`. Its constructor gains `IKConfig` instead
of individual weight floats:

```python
class _FloatingBaseIKModule(nn.Module):
    def __init__(self, robot, target_link_indices, target_poses,
                 initial_base, initial_cfg, rest_cfg, ik_cfg: IKConfig):
        ...
        self._pos_weight = ik_cfg.pos_weight
        self._ori_weight = ik_cfg.ori_weight
        self._pose_weight = ik_cfg.pose_weight
        self._limit_weight = ik_cfg.limit_weight
        self._rest_weight = ik_cfg.rest_weight
```

### 4. Viewer helpers (`viewer/_helpers.py`)

```python
import torch
from ..core._robot import Robot


def wxyz_pos_to_se3(wxyz: tuple, pos: tuple | list) -> torch.Tensor:
    """Convert viser wxyz + position to SE3 [tx, ty, tz, qx, qy, qz, qw]."""
    w, x, y, z = wxyz
    return torch.tensor([pos[0], pos[1], pos[2], x, y, z, w], dtype=torch.float32)


def qxyzw_to_wxyz(q: torch.Tensor) -> tuple:
    """Convert [qx, qy, qz, qw] tensor to viser (w, x, y, z) tuple."""
    return (q[3].item(), q[0].item(), q[1].item(), q[2].item())


def build_cfg_dict(robot: Robot, cfg: torch.Tensor) -> dict[str, float]:
    """Build joint_name→angle dict suitable for ViserUrdf.update_cfg."""
    names = [
        name for name, jtype in zip(robot.joints.names, robot._fk_joint_types)
        if jtype in ("revolute", "continuous", "prismatic")
    ]
    return {name: float(v) for name, v in zip(names, cfg.detach().cpu())}
```

Exported from `viewer/__init__.py`:
```python
from ._helpers import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict
```

### 5. Updated examples

Both examples import from `better_robot.viewer`:
```python
from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict
```

`01_basic_ik.py` new IK call:
```python
cfg = br.solve_ik(
    robot=robot,
    targets={"panda_hand": target_pose},
    cfg=br.IKConfig(rest_weight=0.001),
    initial_cfg=cfg,
    max_iter=20,
)
```

`02_g1_ik.py` new IK call:
```python
base_pose, cfg = br.solve_ik(
    robot=robot,
    targets=targets,
    initial_base_pose=base_pose,
    initial_cfg=cfg,
    max_iter=20,
)
```

`--profile` flag: when passed, wraps the first 10 IK calls with `torch.profiler.profile`
and prints `key_averages().table(sort_by="cpu_time_total", row_limit=15)` then exits.
Add to `02_g1_ik.py` only (it's the slow one).

### 6. Updated tests (`tests/test_ik.py`)

All tests use the new API. Key changes:

```python
# Old
solve_ik(panda, target_link="panda_hand", target_pose=target)
# New
solve_ik(panda, targets={"panda_hand": target})

# Old
solve_ik(..., weights={"pose": 2.0, "limits": 0.5, "rest": 0.001})
# New
solve_ik(..., cfg=IKConfig(pose_weight=2.0, limit_weight=0.5, rest_weight=0.001))

# Old (imported from private module)
from better_robot.tasks._ik import solve_ik_multi
solve_ik_multi(panda, targets=targets, ...)
# New (multi-target is just solve_ik with a dict)
solve_ik(panda, targets=targets, ...)

# Old (imported from private module)
from better_robot.tasks._floating_base_ik import solve_ik_floating_base
base_pose, cfg = solve_ik_floating_base(panda, targets=..., initial_cfg=...)
# New
base_pose, cfg = solve_ik(panda, targets=..., initial_base_pose=identity, initial_cfg=...)
```

For floating-base tests the `initial_base_pose` is `torch.tensor([0., 0., 0., 0., 0., 0., 1.])` (identity SE3: zero translation, unit quaternion).

`test_imports.py` smoke test replaces `solve_ik_multi` and `solve_ik_floating_base` assertions with `IKConfig`.

---

## Testing

No new test files. All 33 existing tests must pass (updated for new API).
Run: `uv run pytest tests/ -v`

---

## Constraints

- `solve_ik_multi` and `solve_ik_floating_base` are **removed from the public API** — not deprecated, just gone. Only one example file uses them and tests are updated.
- `_FloatingBaseIKModule` stays private (`_` prefix) — not exported.
- `IKConfig` uses dataclass defaults so `IKConfig()` gives the same behavior as before.
- `--profile` mode exits after profiling; it does not affect normal operation.
