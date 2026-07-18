# BetterRobot — Agent Guide

BetterRobot is a PyTorch-native robotics library for batched kinematics,
dynamics, residual modeling, and nonlinear least squares. Fixed- and
floating-base robots use the same `Model`/`Data` APIs; a floating base is a
`JointFreeFlyer` in the model topology.

## Start here

- Reader documentation: `docs/index.md`
- Architecture and contracts: `docs/concepts/architecture.md` and
  `docs/conventions/contracts.md`
- Public roadmap: `docs/reference/roadmap.md`
- Package-specific rules: the nearest `CLAUDE.md` under `src/better_robot/`

## Commands

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"
uv run ruff check src/ tests/
uv run sphinx-build -b html docs docs/_build/html
uv run python examples/01_basic_ik.py
```

## Architecture

Imports point down the checked package DAG:

```text
viewer -> tasks -> optim -> residuals
                              |
io   kinematics   dynamics   collision
 \       |           |          /
          data_model -> spatial -> lie
```

Kinematics and dynamics share an architectural rank; dynamics may reuse raw
FK helpers. `io` builds `data_model` objects and does not depend on solvers.
Nothing in the computational core imports `viewer`. The dependency contract is
enforced by `tests/contract/test_layer_dependencies.py`.

Public wrappers validate shapes, dtype, device, and model attachment. Raw
tensor passes below them trust validated inputs. `ModelStructure` holds static
topology, `ModelValues` holds differentiable tensors, and mutable `Data` belongs
to one query. Torch is the reference implementation. Optional acceleration is
pass-specific; only forward kinematics currently has an opt-in Warp lane.

## Numerical contracts

- Pose: `[tx, ty, tz, qx, qy, qz, qw]`; quaternion scalar is last.
- Tangent/twist: `[linear(3), angular(3)]`.
- Public tensors use trailing event axes and arbitrary leading batch axes.
- Frame Jacobians default to `LOCAL_WORLD_ALIGNED`; do not apply a full
  adjoint when only a rotation into the local frame is required.
- `nq` and `nv` can differ. Use `Model.integrate` and `Model.difference`, not
  configuration-space addition or subtraction.
- Preserve dtype/device and PyTorch broadcasting; never silently cast, move,
  squeeze, or change process-wide Torch settings.

## Optimization

`better_robot.optim` has one supported problem representation: named variables
with manifolds, bounds, masks, providers, and fixed-width residuals. It is
consumed by batched LM/GN or `run_first_order`. `solve_ik` and `solve_trajopt`
are recipes over that same surface. Finite differences are an explicit debug
strategy; normal differentiation uses analytic blocks or `torch.func`.

Solver `update` must remain fixed-shape and free of host synchronization.
Eager `run` may check whether every batch element is terminal once per
iteration. Provider caches live for one evaluation only. See
`src/better_robot/optim/CLAUDE.md` before changing solver behavior.

## Working rules

- Keep Lie and spatial operations as direct Torch formulas.
- Keep optional dependencies lazy at their integration boundary.
- Add an abstraction only when at least two in-tree callers need it.
- Validate structure at public boundaries; do not scan user tensor values for
  finiteness as general input validation.
- Update documentation and the nearest agent guide when a contract changes.
- Do not weaken Pinocchio parity or architecture tests to land a refactor.
- Preserve unrelated working-tree changes and use focused tests while editing.

The top-level API is intentionally small. Supported specialized APIs remain
qualified under their package.
