# Analytical Jacobian & Custom LM Optimizer Design

## Goal

Add a custom Levenberg-Marquardt solver that supports both autodiff and analytical (geometric) Jacobians for fixed-base and floating-base IK. Keep the PyPose LM for comparison. Add a benchmark to validate correctness and speed.

## Architecture

```
solvers/
  _base.py                  — Problem gains optional jacobian_fn field
  _lm.py                    — NEW: our LM (autodiff via jacrev OR user-provided jacobian_fn)
  _levenberg_marquardt.py   — UNCHANGED: PyPose-based LM kept for comparison

costs/
  _jacobian.py              — NEW: pose_jacobian(), limit_jacobian(), rest_jacobian()

core/
  _robot.py                 — add Robot.get_chain(link_idx) -> list[int]
  _lie_ops.py               — add adjoint_se3(T_7) -> Tensor (6×6)

tasks/
  _config.py                — add jacobian: Literal["autodiff","analytic"] = "autodiff"
  _ik.py                    — build jacobian_fn when analytic; pass to Problem
  _floating_base_ik.py      — add analytic path alongside existing PyPose path

tests/
  test_lm_benchmark.py      — NEW: correctness + timing benchmark (3 modes × 2 robots)
```

## Tech Stack

PyTorch (torch.func.jacrev for autodiff mode), PyPose (Lie ops only), existing BetterRobot infrastructure.

---

## Component Designs

### 1. `Problem` extension (`solvers/_base.py`)

Add one optional field:

```python
@dataclass
class Problem:
    variables: Tensor
    costs: list[CostTerm]
    lower_bounds: Tensor | None
    upper_bounds: Tensor | None
    jacobian_fn: Callable[[Tensor], Tensor] | None = None
    # None  → our LM computes J via torch.func.jacrev(total_residual)(x)
    # not None → our LM calls jacobian_fn(x) directly → (m, n) Tensor
```

The existing `_levenberg_marquardt.py` (PyPose LM) ignores `jacobian_fn` — it always uses autograd internally.

---

### 2. Our LM Solver (`solvers/_lm.py`)

Parameters mirror PyPose's `LevenbergMarquardt`:

| Our param | PyPose equivalent | Default |
|-----------|-------------------|---------|
| `damping` | `Adaptive(damping=...)` | `1e-4` |
| `factor`  | internal scale factor | `2.0` |
| `reject`  | `reject=16` | `16` |
| `max_iter`| passed to `.solve()` | `100` |

Algorithm (adaptive LM):

```python
class LevenbergMarquardt(Solver):
    def __init__(self, damping=1e-4, factor=2.0, reject=16):
        ...

    def solve(self, problem: Problem, max_iter: int = 100) -> Tensor:
        x = problem.variables.clone().float()
        lam = self.damping
        lo, hi = problem.lower_bounds, problem.upper_bounds

        for _ in range(max_iter):
            r = problem.total_residual(x)                    # (m,)
            J = self._compute_jacobian(problem, x)           # (m, n)
            JtJ = J.T @ J                                    # (n, n)
            Jtr = J.T @ r                                    # (n,)

            for _ in range(self.reject):
                I_n = torch.eye(J.shape[1], ...)
                delta = torch.linalg.solve(JtJ + lam * I_n, -Jtr)
                x_new = (x + delta)
                if lo is not None:
                    x_new = x_new.clamp(lo, hi)
                if problem.total_residual(x_new).norm() <= r.norm():
                    x = x_new
                    lam = max(lam / self.factor, 1e-7)
                    break
                lam = min(lam * self.factor, 1e7)

        return x

    def _compute_jacobian(self, problem: Problem, x: Tensor) -> Tensor:
        if problem.jacobian_fn is not None:
            return problem.jacobian_fn(x)
        return torch.func.jacrev(problem.total_residual)(x)
```

Registered in `SOLVER_REGISTRY` as `"lm"` (replacing the PyPose one). PyPose LM registered as `"lm_pypose"`.

---

### 3. Analytical Jacobian Functions (`costs/_jacobian.py`)

#### `pose_jacobian`

```python
def pose_jacobian(
    cfg: Tensor,
    robot: Robot,
    target_link_index: int,
    target_pose: Tensor,
    pos_weight: float,
    ori_weight: float,
    base_pose: Tensor | None = None,
) -> Tensor:   # (6, n_joints)
```

For each joint `j` in `robot.get_chain(target_link_index)`:
- Run FK to get world-frame joint poses and EE pose
- **Revolute**: `J[:,j] = [ω_j × (p_ee − p_j) ; ω_j]` (world frame)
- **Prismatic**: `J[:,j] = [d_j ; 0]`

Apply `pos_weight` / `ori_weight` to the 3+3 rows.

**jlog approximation**: Use identity (first-order). Valid for errors < ~30°; documented in CLAUDE.md. Exact jlog is a future improvement.

#### `limit_jacobian`

```python
def limit_jacobian(cfg: Tensor, robot: Robot) -> Tensor:  # (2n, n)
```

Diagonal ±1 for violated limits (same logic as `limit_residual`), 0 inside limits.

#### `rest_jacobian`

```python
def rest_jacobian(cfg: Tensor, rest_pose: Tensor) -> Tensor:  # (n, n)
```

Returns `torch.eye(len(cfg))` — rest residual is `cfg − rest`, J is identity.

---

### 4. `Robot.get_chain` (`core/_robot.py`)

```python
def get_chain(self, link_idx: int) -> list[int]:
    """Joint indices on the path from root to link_idx (in traversal order)."""
```

Walks `_fk_joint_child_link` / `_fk_joint_parent_link` backwards from `link_idx` to root. Returns only actuated joints (cfg_idx >= 0). ~15 lines.

---

### 5. `adjoint_se3` (`core/_lie_ops.py`)

```python
def adjoint_se3(T: Tensor) -> Tensor:  # T: (7,) → (6, 6)
```

For SE3 `T = [px, py, pz, qx, qy, qz, qw]`:

```
Ad(T) = [[R,          skew(p) @ R],
          [zeros(3,3), R          ]]
```

Where se3 tangent convention is `[tx, ty, tz, rx, ry, rz]` (PyPose native — translation first).

`skew(p)` is the 3×3 skew-symmetric matrix of vector p.

---

### 6. `IKConfig` change (`tasks/_config.py`)

```python
@dataclass
class IKConfig:
    pos_weight: float = 1.0
    ori_weight: float = 0.1
    pose_weight: float = 1.0
    limit_weight: float = 0.1
    rest_weight: float = 0.01
    jacobian: Literal["autodiff", "analytic"] = "autodiff"
```

---

### 7. Fixed-base IK dispatch (`tasks/_ik.py`)

When `ik_cfg.jacobian == "analytic"`, build `jacobian_fn` and attach to `Problem`:

```python
def _build_jacobian_fn(robot, targets, ik_cfg):
    """Returns (cfg,) -> (m, n) full Jacobian for the IK problem."""
    target_specs = [(robot.get_link_index(name), pose)
                    for name, pose in targets.items()]
    rest = robot._default_cfg.clone()

    def jacobian_fn(cfg):
        rows = []
        for link_idx, target_pose in target_specs:
            J = pose_jacobian(cfg, robot, link_idx, target_pose,
                              ik_cfg.pos_weight, ik_cfg.ori_weight)
            rows.append(J * ik_cfg.pose_weight)
        rows.append(limit_jacobian(cfg, robot) * ik_cfg.limit_weight)
        rows.append(rest_jacobian(cfg, rest) * ik_cfg.rest_weight)
        return torch.cat(rows, dim=0)

    return jacobian_fn
```

Use `SOLVER_REGISTRY["lm"]` for both modes. Our LM handles the dispatch internally via `problem.jacobian_fn`.

---

### 8. Floating-base analytic path (`tasks/_floating_base_ik.py`)

When `ik_cfg.jacobian == "analytic"`, use a separate ~35-line loop instead of `_FloatingBaseIKModule`:

```python
def _run_floating_base_lm_analytic(robot, targets, ik_cfg, initial_cfg, initial_base_pose, max_iter):
    base = initial_base_pose.clone().float()
    cfg = initial_cfg.clone().float()
    lam = 1e-4
    ...
    for _ in range(max_iter):
        r = _compute_residuals(base, cfg, ...)   # (m,)
        J = _build_floating_jacobian(base, cfg, ...)  # (m, 6+n)
        # J = [J_joints | J_base] per EE, stacked with limit/rest rows
        # J_base = adjoint_se3(se3_inverse(T_ee_local))  ← smart trick
        delta = linalg.solve(J.T@J + lam*I, -J.T@r)
        base = se3_compose(se3_exp(delta[:6]), base)
        cfg = (cfg + delta[6:]).clamp(lo, hi)
        # adaptive damping ...
    return base.detach(), cfg.detach()
```

**Base Jacobian smart trick:**
```
T_ee_local = se3_compose(se3_inverse(base), T_ee_world[link_idx])
J_base = adjoint_se3(se3_inverse(T_ee_local))   # (6, 6), no extra FK call
```

The Adjoint maps a body-frame perturbation of the base to the resulting EE perturbation in world frame. `T_ee_local` is derived from the already-computed world FK — no second FK pass needed.

**Limit/rest rows for floating base**: limit and rest apply only to `cfg` (the joint columns). The base columns (first 6) are zero for these rows.

---

### 9. Benchmark (`tests/test_lm_benchmark.py`)

Three configurations, same Panda IK problem (same random reachable target):

| Mode | Solver | jacobian |
|------|--------|----------|
| Baseline | PyPose LM (`lm_pypose`) | autograd (internal) |
| Our autodiff | Our LM (`lm`) | `jacrev` |
| Our analytic | Our LM (`lm`) | `pose_jacobian` |

Asserts:
- All three reach position error < 5 cm (correctness)
- Our analytic ≤ our autodiff in wall-clock time (speed check)

Reports: ms/call, final position error, iterations to convergence.

Also one floating-base scenario (G1 with 2 EE targets) for all three modes.

---

## CLAUDE.md Updates

- `solvers/CLAUDE.md`: document our LM, `jacobian_fn` in Problem, `SOLVER_REGISTRY` keys
- `costs/CLAUDE.md`: document `_jacobian.py` functions + jlog approximation note
- `core/CLAUDE.md`: document `get_chain`, `adjoint_se3`
- Root `CLAUDE.md`: update solver notes section

---

## What Is Not In Scope

- Exact jlog (left Jacobian of SE3 log) — approximated as identity; noted in CLAUDE.md
- Analytical Jacobian for other cost terms (manipulability, collision) — stubs only
- LBFGS / GN solver implementations — still stubs
