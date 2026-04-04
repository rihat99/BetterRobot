# tests/

All tests use real robot data (Panda URDF via `robot_descriptions`). No mocking of FK or URDF parsing.

## Running

```bash
uv run pytest tests/ -v          # all 24 tests
uv run pytest tests/test_ik.py   # IK tests only
```

## Test Files

| File | Covers |
|------|--------|
| `test_imports.py` | Smoke tests: all public APIs importable |
| `test_robot.py` | FK shape, unit quaternions, batch FK, link index |
| `test_costs.py` | Pose residual shape/zero/gradient, limit residual, rest residual |
| `test_solvers.py` | LM converges on quadratic, shape, `total_residual`, `constraint_residual` |
| `test_ik.py` | IK shape, joint limits respected, convergence, custom weights |

## Fixtures

`panda` fixture (in `test_robot.py`, `test_costs.py`, `test_ik.py`) returns a `Robot` loaded from the real Panda URDF. It's session-scoped — URDF is parsed once per test run.

## Conventions

- Target poses use format `[tx, ty, tz, qx, qy, qz, qw]` (PyPose convention)
- Use `robot._default_cfg` (midpoint of joint limits) as the default warm start
- IK convergence test uses the FK of the default config as the target — this guarantees reachability
- Position error tolerance: 5 cm for convergence tests
