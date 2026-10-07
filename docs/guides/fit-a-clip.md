# Fit a clip in phases

A motion fit rarely works as one solve: the root has to find the subject first, a cheap optimizer gets close,
and Levenberg-Marquardt finishes on the banded system. This guide fits a floating-base humanoid to noisy marker
tracks with `optim.Phase` and `optim.run`. You keep one problem and change only weights, free coordinates and the
solver between phases.

Read [Time and structure](../concepts/time-and-structure.md) first if time variables and `window` are new to you.

## Build one problem over the whole clip

The unknown is the whole clip `q (B, T, nq)`. `time=("q",)` marks it as a time variable, so the solver sees a
block-tridiagonal normal matrix instead of a dense one. The example's Unitree G1 has a free-flyer root
(`nq = nv + 1`), so the variable gets `terms.RobotManifold(model)`, which steps it with `Model.integrate`.

```python
import torch
import better_robot as br
from robot_descriptions import g1_description
from better_robot import optim, terms
from better_robot.kinematics import forward_kinematics_state

KNOTS, FPS = 20, 30.0
model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
markers = [n for n in model.frame_names if any(k in n for k in ("pelvis", "ankle_roll", "wrist_yaw", "head"))][:6]
marker_ids = [model.frame_id(n) for n in markers]

# A synthetic reference clip and noisy markers with 2 % outliers of 0.5 m.
generator = torch.Generator().manual_seed(0)
phase = torch.linspace(0, 6.28, KNOTS, dtype=torch.float64)[:, None]
q_true = model.q_neutral.repeat(KNOTS, 1)
q_true[:, 7:] += 0.4 * torch.sin(phase + torch.arange(model.nq - 7, dtype=torch.float64))
q_true[:, 0] += 0.5 * phase[:, 0] / 6.28
true_positions = forward_kinematics_state(model, q_true[None]).frame_pose[:, :, marker_ids, :3]  # (1, T, M, 3)
observed = true_positions + 0.005 * torch.randn(true_positions.shape, generator=generator, dtype=torch.float64)
observed[torch.rand(true_positions.shape[:-1], generator=generator) < 0.02] += 0.5

problem = optim.Problem(
    {"q": model.q_neutral.repeat(1, KNOTS, 1)},           # q (1, T, nq)
    manifolds={"q": terms.RobotManifold(model)},
    data={"observed": observed},
    time=("q",),
)
problem.cached(terms.kinematic_state(model), name="state")
problem.residual(
    "markers",
    lambda state, observed: state.frame_pose[:, :, marker_ids, :3] - observed,
    kernel=optim.Huber(delta=0.03),
    window=1,
)
problem.residual("smooth", terms.finite_difference(model, order=2, dt=1 / FPS), weight=1e-5)
problem.residual("rest", lambda q: q[..., 7:] - model.q_neutral[7:], weight=1e-3, window=1)
```

The terms do not know about phases. `markers` and `rest` touch one knot each (`window=1`); `smooth` is a
second-order finite difference over neighbouring knots. The Huber kernel keeps the outliers from dragging the fit.

## Describe each phase by what changes

A `Phase` is a solver, an iteration count, and overrides. Everything not overridden comes from the problem.

| Field | Meaning |
|---|---|
| `solver`, `iters` | any `Solver` (`LevenbergMarquardt`, `GaussNewton`, `LBFGS`, a wrapped `torch.optim` optimizer) and its step count |
| `name` | key of this phase in `result.history` |
| `weights` | replaces term weights by name for this phase; `0.0` switches a term off |
| `free` | per-variable boolean mask over the flattened tangent; `False` freezes a coordinate |
| `on_start(problem, x)` | runs first; may return new variables or rewrite the phase's copy of `problem.data` |
| `on_step(problem, state)` | runs after every step, for logging |
| `stop_every` | how often the driver checks whether every batch element has converged; `0` runs exactly `iters` |

```python
root_only = torch.zeros(KNOTS, model.nv, dtype=torch.bool)
root_only[:, :6] = True                                    # the 6 root velocity coordinates of every knot

phases = [
    optim.Phase(optim.LevenbergMarquardt(), 5, name="root", free={"q": root_only.flatten()}, weights={"smooth": 0.0}),
    optim.Phase(optim.LBFGS(), 10, name="lbfgs"),
    optim.Phase(optim.LevenbergMarquardt(), 5, name="lm"),
]
result = optim.run(problem, phases)
```

1. **root**: only the floating base moves, with smoothing off, so the body slides onto the markers before any joint
   bends to compensate for a wrong root.
2. **lbfgs**: every coordinate is free; L-BFGS takes cheap first-order steps.
3. **lm**: Levenberg-Marquardt solves the banded normal equations and converges in a few steps.

`free` masks the tangent, not the configuration: its length is `T * nv` for a time variable, and the root block is
the first 6 tangent coordinates of each knot even though the root uses 7 configuration numbers.

## A weight schedule is a list of phases

There is no separate schedule object. To anneal a term, repeat a phase with a different `weights` value; each phase
starts from the previous result.

```python
schedule = [optim.Phase(optim.LevenbergMarquardt(), 3, name=f"smooth={w:g}", weights={"smooth": w}) for w in (1e-3, 1e-5)]
annealed = optim.run(problem, schedule, x=result.x)
```

## Read the result

`result.x` holds the fitted variables and `result.history` maps each phase name to its loss per step.

```python
for name, losses in result.history.items():
    print(f"{name:6s}: loss {float(losses[0].mean()):.4f} -> {float(losses[-1].mean()):.4f} in {len(losses)} steps")
fitted = forward_kinematics_state(model, result.x["q"]).frame_pose[:, :, marker_ids, :3]
print(f"mean marker error {float((fitted - true_positions).norm(dim=-1).mean()) * 1e3:.1f} mm")
```

The full example uses 120 knots and more iterations:

```bash
uv run python examples/03_clip_fit_phases.py
```

## Related

- [Optimization](../concepts/optimization.md): `Problem`, terms, kernels and solvers.
- [Time and structure](../concepts/time-and-structure.md): time variables, `window`, banded solves.
- [Enforce hard constraints](hard-constraints.md) when a term must hold exactly.
