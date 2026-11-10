# BetterRobot — Agent Guide

BetterRobot is a PyTorch-native robotics library for batched kinematics,
dynamics, residual modeling, and nonlinear least squares. Fixed- and
floating-base robots use the same state API
(`forward_kinematics(model, q) -> KinematicState`, then Jacobians and
`dynamics` functions on that state); a floating base is a
`JointFreeFlyer` in the model topology.

## Start here

- Reader documentation: `docs/index.md`
- Architecture: `docs/concepts/architecture.md`
- Design contracts (rule IDs `C-*`, ADRs): `docs/contracts/00-index.md`
- Public roadmap: `docs/reference/roadmap.md`
- Package-specific rules: the nearest `CLAUDE.md` under `src/better_robot/`

## Commands

```bash
uv run pytest tests/contract -q -m "not slow and not cuda"   # fast tier, about 16 s (pre-commit hook)
uv run pytest tests -q -m "not cuda"                         # full tier, about 4.5 min (pre-push hook)
bash tools/release_tier.sh --gpu <index>                     # release tier, by hand, a free GPU
uv run ruff check src/ tests/
uv run mkdocs build --strict
uv run python examples/01_ik.py
```

## Architecture

Each package imports only the packages of its row in the edge table
(`ALLOWED` in `tests/contract/test_layer_dependencies.py`, contract 07):

```text
exceptions  -> nothing
_internal_checks -> nothing
lie         -> exceptions
spatial     -> exceptions lie
model       -> exceptions lie spatial
kinematics  -> exceptions lie model
dynamics    -> exceptions lie spatial model kinematics
collision   -> exceptions lie model kinematics
io          -> exceptions _internal_checks lie model
optim       -> exceptions
terms       -> exceptions lie model kinematics collision optim
viewer      -> exceptions lie model kinematics collision
testing     -> exceptions model
```

The root `__init__` re-exports packages and has no row. No row names `viewer`,
`terms` or `testing`. Outside the package, `lie`, `spatial` and `optim` import
torch only (`optim` knows nothing about robots), and every other import of a
third-party package is one `pyproject.toml` declares for that package
(`tests/contract/test_optional_imports.py`).

Public wrappers validate shapes, dtype, device, and model attachment. Raw
tensor passes below them trust validated inputs. `Model` is one flat frozen
record: its static fields hold the topology, its tensor fields the
differentiable parameters; a frozen `KinematicState` holds the results of one
forward-kinematics query. Torch is the only implementation.

## Numerical contracts

- Pose: `[tx, ty, tz, qx, qy, qz, qw]`; quaternion scalar is last.
- Tangent/twist: `[linear(3), angular(3)]`.
- Public tensors use trailing event axes and arbitrary leading batch axes.
- Frame Jacobians default to `LOCAL_WORLD_ALIGNED`; do not apply a full
  adjoint when only a rotation into the local frame is required.
- `nq` and `nv` can differ. Use `kinematics.integrate(model, q, v)` and
  `kinematics.difference(model, q0, q1)`, not configuration-space addition or
  subtraction.
- Preserve dtype/device and PyTorch broadcasting; never silently cast, move,
  squeeze, or change process-wide Torch settings.
- The full list of shapes, layouts and formats other repositories read:
  `docs/contracts/coupling-standard.md`.

## Optimization

`better_robot.optim` optimizes named tensors with plain-function terms: a
`Problem` holds the variables and `TermFn` terms, and LM, Gauss-Newton, L-BFGS
or a wrapped `torch.optim` optimizer solves it (`solve`, `run` with `Phase`s,
`solve_layer` for differentiable use, `solve_constrained` for hard
constraints). It imports torch and `better_robot.exceptions` only. Robot
terms (poses, limits, collision, `ik_problem`, `solve_ik`) live in
`better_robot.terms`. See `src/better_robot/optim/CLAUDE.md` before changing
optimizer behavior.

## Working rules

- Keep Lie and spatial operations as direct Torch formulas.
- Keep optional dependencies lazy at their integration boundary.
- Add an abstraction only when at least two in-tree callers need it.
- Validate structure at public boundaries; do not scan user tensor values for
  finiteness as general input validation.
- Update documentation and the nearest agent guide when a contract changes.
- A new word in a public name enters `docs/reference/glossary.md` in the same
  change (`test_glossary_words`).
- Do not weaken Pinocchio parity or architecture tests to land a refactor.
- Preserve unrelated working-tree changes and use focused tests while editing.

The top-level API is intentionally small. Supported specialized APIs remain
qualified under their package.
