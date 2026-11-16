# Track moving targets

A target that moves every control tick, such as a hand pose from a teleoperation device, needs one small
correction per tick, not a solve to convergence. `optim.track` takes that correction: one quadratic program per
call that moves the robot toward the problem's targets and keeps its constraint terms and bounds. After this guide
you can run a tracking loop, keep a robot off a table, out of an obstacle, out of itself, inside its joint limits
and away from singular poses, move the obstacle every tick, compile the tick, and choose a step for which the rows
hold.

## One call per tick

`terms.ik_problem` builds the problem once; each tick swaps the new target in with `with_(data=)` and calls
`track`. The Panda's hand follows a vertical circle of 10 cm, one turn in 400 ticks (4 s at 100 Hz):

```python
import math

import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot import collision, forward_kinematics, optim, terms

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
ready = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04, 0.04]
q0 = torch.tensor([ready[: model.nq]], dtype=torch.float64)                   # (B=1, nq)
hand = model.frame_id("panda_hand")
start = forward_kinematics(model, q0).frame_poses[:, hand]                     # (1, 7)


def target(tick: int) -> torch.Tensor:                                         # (1, 7): the start pose, on the circle
    angle = 2 * math.pi * tick / 400
    return start + torch.tensor([0.1 * math.sin(angle), 0, 0.1 * (math.cos(angle) - 1), 0, 0, 0, 0])


problem = terms.ik_problem(model, {"panda_hand": start}, q0=q0)
x, warm, lag = dict(problem.variables), None, []
for tick in range(1, 401):
    x, warm = optim.track(problem.with_(data={"target/panda_hand": target(tick)}), x, warm=warm, step=0.005)
    lag.append(float((forward_kinematics(model, x["q"]).frame_poses[:, hand, :3] - target(tick)[:, :3]).norm()))
print(f"the hand follows {1e3 * sorted(lag)[200]:.1f} mm behind its target (median), at most {1e3 * max(lag):.1f} mm")
```
```output
the hand follows 2.9 mm behind its target (median), at most 26.6 mm
```

`track` returns the new point and the program's solution. Passing that solution back as `warm=` starts the next
program from it; the pair is all the state a loop carries. `step` bounds the move of every joint per call: a
robot's velocity limits times the tick period, or less (below). The lag is that bound at work: parts of the circle
ask some joints for more than 5 mrad per tick.

## The five behaviours are rows

Each behaviour is the term of its cost form registered with `hinge=False` as an inequality, `upper=0.0`:

| Behaviour | Registration |
|---|---|
| joint limits | none: `ik_problem` passes them as `bounds=`, a box on every step |
| floor or table | `terms.sphere_clearance(spheres, to_floor, margin=, hinge=False, closest=k)` |
| obstacle | `terms.sphere_clearance(spheres, to_obstacle, margin=, hinge=False, closest=k)` |
| self-collision | `terms.self_clearance(spheres, pairs=, margin=, hinge=False, closest=k)` |
| singular poses | `terms.manipulability(model, frame=, threshold=, hinge=False)` |

`closest=k` keeps the `k` nearest spheres or pairs as rows: few rows, and the same number at every tick, as the
warm start needs. Below, a table 30 cm up and a sphere on the circle stand in the hand's way:

```python
spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", model)
pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)
margin, up = 0.02, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
obstacle = torch.cat([target(100)[0, :3], torch.tensor([0.04], dtype=torch.float64)])      # centre and radius
shoulder = torch.tensor([model.frame_names[int(f)] in ("panda_link0", "panda_link1", "panda_link2")
                         for f in spheres.frame_ids])                                # these stand through the table


def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:   # (B, S, 3), (S,) -> (B, S)
    return (centers - obstacle[:3]).norm(dim=-1) - radii - obstacle[3]


def to_table(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    return collision.sphere_halfspace(centers, radii, normal=up, offset=0.3) + shoulder


rows = {
    "table": terms.sphere_clearance(spheres, to_table, margin=margin, hinge=False, closest=8),
    "obstacle": terms.sphere_clearance(spheres, to_obstacle, margin=margin, hinge=False, closest=8),
    "self": terms.self_clearance(spheres, pairs=pairs, margin=margin, hinge=False, closest=8),
    "singular": terms.manipulability(model, frame="panda_hand", threshold=0.03, hinge=False),
}
held = terms.ik_problem(model, {"panda_hand": start}, q0=q0)
for name, row in rows.items():
    held.inequality(name, row, upper=0.0)


def deepest(q: torch.Tensor) -> torch.Tensor:                                  # every row's largest violation
    state = forward_kinematics(model, q)
    centers = spheres.centers(state)
    return torch.stack([torch.maximum(model.lower_position_limit - q, q - model.upper_position_limit).max(),
                        (margin - to_table(centers, spheres.radii)).max(),
                        (margin - to_obstacle(centers, spheres.radii)).max(),
                        (margin - collision.self_distances(spheres, state, pairs=pairs)).max(),
                        rows["singular"].fn(state).max()]).clamp_min(0.0)


x, warm, worst, away = dict(held.variables), None, torch.zeros(5, dtype=torch.float64), 0.0
for tick in range(1, 401):
    x, warm = optim.track(held.with_(data={"target/panda_hand": target(tick)}), x, warm=warm, step=0.005)
    worst = torch.maximum(worst, deepest(x["q"]))
    away = max(away, float((forward_kinematics(model, x["q"]).frame_poses[:, hand, :3] - target(tick)[:, :3]).norm()))
print("limits, table, obstacle, self, singular:", [f"{float(v):.0e}" for v in worst])
print(f"the hand stayed up to {1e3 * away:.0f} mm from targets inside the table or the sphere")
```
```output
limits, table, obstacle, self, singular: ['0e+00', '1e-05', '3e-06', '0e+00', '0e+00']
the hand stayed up to 148 mm from targets inside the table or the sphere
```

## How far the rows hold

A row's value can shrink toward its bound by at most `gain` of the distance left per call (`gain=0.5`): a discrete
barrier, exact for rows linear in the configuration, such as the joint limits. A clearance or the manipulability
is curved; its linearization misses a part that grows with the square of the step, so the robot can end a tick a
little past the margin. Largest violation over 300 ticks of 256 targets whose paths cross each region (float64; the
joint-limit box held exactly in every run, to rounding in float32):

| Step per tick | Panda, all five | Panda, worst single behaviour | G1, all five | G1, worst single behaviour |
|---|---|---|---|---|
| 25 mrad | 3.9e-4 | 5.0e-4 (obstacle) | 5.9e-4 | 9.0e-4 (obstacle) |
| 6.3 mrad | 7.5e-5 | 6.9e-5 (singular) | 2.9e-4 | 9.9e-5 (self-collision) |

Within 1e-4 m (or 1e-4 of manipulability) takes steps of about 6.3 mrad per tick: the Panda then holds all five
behaviours, the G1 each one alone; the G1 with all five at once is just over (3 of 256 tasks, the same in float32
with 5). There more `solve_qp` iterations per call close the gap, since its programs are exact on only 44 % of the
ticks at the default 20: `track(..., iterations=60)` leaves 1.0e-4 (1 task just over, 59 % exact) and
`iterations=100` 8.8e-5 (none over, 68 % exact), at a price: a compiled G1 tick with clearance rows takes 3.0 ms
instead of 1.75 ms on one CPU core. A weighted hinge cost in place of the rows leaves 1e-2 to 6e-2, and no
behaviour at all 0.2 to 3. Without `step`, a far target gives a long step and the curvature has its way.

## A moving obstacle

The problem holds its functions by reference, so the obstacle's distance function can read a tensor that the
loop moves in place: no new problem, and a compiled tick compiles nothing new. The rows see the obstacle where it
is when the tick starts. One that closes in by `v` per tick therefore ends a tick up to `v / gain` inside the
margin; pad its radius by that much. Here a sphere sweeps through the hand at 1 mm per tick while the hand holds
its pose:

```python
still = held.with_(data={"target/panda_hand": start})
for pad in (0.0, 0.002):
    x, warm, inside, moved = dict(held.variables), None, 0.0, 0.0
    for tick in range(1, 601):
        obstacle.copy_(torch.tensor([start[0, 0], tick / 1000 - 0.3, start[0, 2], 0.04 + pad]))
        x, warm = optim.track(still, x, warm=warm, step=0.005)
        obstacle.copy_(torch.tensor([start[0, 0], (tick + 1) / 1000 - 0.3, start[0, 2], 0.04]))  # where it went
        inside = max(inside, float(deepest(x["q"])[2]))
        moved = max(moved, float((forward_kinematics(model, x["q"]).frame_poses[:, hand, :3] - start[:, :3]).norm()))
    print(f"radius padded {1e3 * pad:.0f} mm: {1e3 * inside:.2f} mm inside the margin; the hand gave way {1e3 * moved:.0f} mm")
```
```output
radius padded 0 mm: 2.00 mm inside the margin; the hand gave way 125 mm
radius padded 2 mm: 0.00 mm inside the margin; the hand gave way 128 mm
```

Move the tensor in place (`copy_`, as here): the problem keeps its function, and a compiled tick keeps its graph
(measured: no graph after the first ones over 600 ticks with a new target each on CPU, and over 420 ticks of CUDA
graphs).

## A twist target

There is no velocity task: a twist `[v, ω]` in the frame's own axes, held for one tick of `dt` seconds, is the
target pose `compose(pose, exp(twist · dt))` of the frame's current pose.

```python
from better_robot import lie

twist = torch.tensor([[0.0, 0.05, 0.0, 0.0, 0.0, 0.3]], dtype=torch.float64)  # (1, 6): 5 cm/s, 0.3 rad/s, hand axes
pose = forward_kinematics(model, x["q"]).frame_poses[:, hand]
goal = lie.se3.compose(pose, lie.se3.exp(twist * 0.01))
x, warm = optim.track(held.with_(data={"target/panda_hand": goal}), x, warm=warm, step=0.005)
```

A twist in world-aligned axes (`reference="local_world_aligned"`) first turns into the frame's axes: both halves
by `lie.so3.act(lie.so3.inverse(pose[..., 3:]), ·)`.

## Compile the tick

```python { .no-run data-reason="a compile takes about 20 s; tests/terms/test_constrained_ik.py runs this loop compiled against the eager one" }
tick = br.kinematics.compiled(lambda problem, x, warm: optim.track(problem, x, warm=warm, step=0.005))
x, warm = tick(held.with_(data={"target/panda_hand": pose}), x, warm)       # 0 reads back to the host
```

The first calls run eagerly (one with `warm=None`, one with a solution), the next compiles; a new target, an
obstacle moved in place or a new problem from `with_` reuses the graph. `mode="reduce-overhead"` replays CUDA
graphs on a GPU.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `warm` | `None` | the solution of the last call: its program starts from there |
| `gain` | `0.5` | share of a row's distance to its bound (and of the box's) one call may use, and of a violation it repairs; 0.2 tracks the same within the measurement |
| `damping` | `1e-3` | added to the diagonal of the Gauss-Newton matrix: keeps a step finite where the targets' Jacobian loses rank; 1e-2 lags the targets by 0.3 mm (median), 1e-4 by 0.02 mm |
| `step` | `None` | bound on every entry of the step, a number or `(nv,)`: velocity limits times the tick period |

`track` takes the step the program gives, without the acceptance test of `SequentialQP`. Where the program's step
is not its exact solution (`solution.exact` unset), the step taken is scaled back until it leaves no row by more
than 1e-6. A target's Jacobian that loses rank does not stall the loop: the damping keeps the program positive
definite there.

## Where next

- [Solve inverse kinematics under constraints](constrained-ik.md): the same rows, solved to convergence.
- [Optimization](../concepts/optimization.md#constraint-terms): the rows of a constraint term.
- [Collision](../concepts/collision.md): the sphere distances behind the clearance terms.
