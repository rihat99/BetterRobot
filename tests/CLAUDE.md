# tests/

All tests use real robot data (Panda URDF via `robot_descriptions`). No mocking of FK or URDF parsing.

## Running

```bash
uv run pytest tests/ -v          # all 57 tests
uv run pytest tests/test_ik.py   # IK tests only
uv run pytest tests/test_lm_benchmark.py -v -s  # benchmarks with timing output
```

## Test Files

| File | Covers |
|------|--------|
| `test_imports.py` | Smoke tests: all public APIs importable, registry correct, geometry types present |
| `test_robot.py` | RobotModel construction, FK shape/quaternions/batch, `link_index`, `get_chain`, `adjoint_se3` |
| `test_costs.py` | `pose_residual` shape/zero/gradient/base_pose, `limit_residual`, `rest_residual` |
| `test_jacobian.py` | `compute_jacobian` shape + finite-diff validation, `limit_jacobian`, `rest_jacobian` |
| `test_solvers.py` | LM converges on quadratic, `Problem.total_residual/constraint_residual`, autodiff vs PyPose LM, analytic Jacobian IK |
| `test_ik.py` | IK shape, joint limits, convergence, custom weights, multi-target, floating-base (all modes) |
| `test_lm_benchmark.py` | Timing comparison: autodiff vs analytic vs PyPose LM (fixed + floating base) |

## Fixtures

`panda` fixture (in `test_robot.py`, `test_costs.py`, `test_ik.py`, `test_jacobian.py`, `test_solvers.py`) returns a `RobotModel` loaded from the real Panda URDF. Session/module scoped — URDF is parsed once per test run.

```python
@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return load_urdf(urdf)
```

## Conventions

- Target poses use format `[tx, ty, tz, qx, qy, qz, qw]` (PyPose convention)
- Use `model._default_cfg` (midpoint of joint limits) as the default warm start
- IK convergence tests use FK of the default config as target — guarantees reachability
- Position error tolerance: 5 cm for convergence tests
- Finite-diff Jacobian check: `eps=1e-3` (float32 — smaller values cause cancellation in `se3_log`)

## Key API (new — no old `Robot` or `get_link_index`)

```python
from better_robot import load_urdf, solve_ik, IKConfig
from better_robot.algorithms.kinematics import compute_jacobian, limit_jacobian, rest_jacobian
from better_robot.costs import CostTerm, pose_residual, limit_residual, rest_residual
from better_robot.solvers import LevenbergMarquardt, PyposeLevenbergMarquardt, Problem

model = load_urdf(urdf)
idx = model.link_index("panda_hand")    # not get_link_index()
J = compute_jacobian(model, cfg, idx, target, 1.0, 0.1)   # model first, cfg second
```
