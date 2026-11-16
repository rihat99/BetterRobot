# Try the avoidance demos

Five scripts in `examples/viewer/` put the behaviours of [Track moving targets](track-targets.md) in a browser:
joint limits, singular poses, the floor, an obstacle and the robot's own body. You drag a target, the robot follows
it with one `optim.track` call per tick, and a GUI switch turns the behaviour into a cost term or off, so you see
what the rows prevent. After this guide you can run each demo, read what it shows, and know the settings and the
limits behind it.

## Run a demo

```bash
uv run python examples/viewer/05_avoid_joint_limits.py            # Panda
uv run python examples/viewer/06_avoid_singularity.py             # Panda
uv run python examples/viewer/07_avoid_floor.py --robot g1        # Panda by default, or the G1
uv run python examples/viewer/08_avoid_obstacle.py --robot g1
uv run python examples/viewer/09_avoid_self_collision.py --robot g1
```

Each starts a viewer at `http://127.0.0.1:8080` (forward the port to view it from another machine) and tracks the
transform handle named "target": drag it, or turn it by its rings. The GUI holds a switch with three choices and a
readout:

| Choice | What `track` gets |
|---|---|
| rows (the default) | the behaviour as inequality rows: the term with `hinge=False`, registered with `problem.inequality(..., upper=0.0)`; for the joint limits, the box of the problem's bounds |
| cost | the same term as a weighted hinge residual (`problem.residual(..., weight=1e3)`): it pushes back, and gives way a little |
| off | nothing: the robot goes where the target pulls it |

The readout shows the behaviour's number (the smallest clearance, the distance to the nearest limit or the
manipulability) and the time of the last `track` call. Spheres or markers more than 1 mm inside the margin turn
red, and each layer has its own switch in the group's folder, such as the sphere set. Switching the form starts
the next program cold (`warm=None`), since its row count changes.

Every script also runs a fixed path without a browser: `--check` runs it on the stand-in server and prints the
numbers below, `--record clip.webm` records it through a headless Chromium (see
[Record a video of a scene](record-a-video.md)). The recording shows the robot without the behaviour on the left
and with its rows on the right; the blue ball is the target.

## Joint limits

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_joint_limits.png" width="960"><source src="../../clips/avoid_joint_limits.webm" type="video/webm"></video><figcaption>The target turns 4.5 rad about the hand's axis and back. Without limits (left) the last joint turns 1.58 rad past its range (red marker); with the box (right) it stops at its limit (amber) and the other joints turn the hand.</figcaption></figure>

`terms.ik_problem` passes the position limits as the problem's bounds, and `track` makes them a box on every step.
The box is linear in the configuration, so it holds exactly at any step size: this demo takes 20 mrad per tick. The
cost form is `terms.joint_limits(model, margin=0.05)` on a problem without bounds.

## Singular poses

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_singularity.png" width="960"><source src="../../clips/avoid_singularity.webm" type="video/webm"></video><figcaption>The target reaches 80 cm out and down and tilts outward. Without the row (left) the hand lines up with the shoulder and the elbow and its manipulability falls to 0.005 (red); with the row (right) it stays at the threshold, 0.03.</figcaption></figure>

The row is `terms.manipulability(model, frame="panda_hand", threshold=0.03, hinge=False)`: `threshold - w <= 0`,
with `w = sqrt(det(J Jᵀ))` of the hand's position Jacobian. The Panda's elbow stops 4° short of straight, so a
plain stretch keeps `w` near 0.05; the low values come where the wrist folds the hand onto the line of the shoulder
and the elbow, which the scripted path reaches on its way out.

## Floor

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_floor_panda.png" width="960"><source src="../../clips/avoid_floor_panda.webm" type="video/webm"></video><figcaption>The target goes 65 cm down, 25 cm out and back. Without the floor (left) the hand goes 160 mm into it; with the rows (right) every sphere stays 20 mm above.</figcaption></figure>

Every sphere of the robot's set is a row, `terms.sphere_clearance(spheres, to_floor, margin=0.02, hinge=False,
closest=8)` (the G1: `closest=None`, every sphere), with `to_floor` from `collision.sphere_halfspace`. The spheres
that stand on the floor by design, the Panda's base links and the G1's feet, get 1 m added to their distance, so
their rows never press. On the G1 the left hand follows the target and the other hand and both feet hold their
start poses, so it bends at the hips and knees to reach down.

## Obstacle

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_obstacle_panda.png" width="960"><source src="../../clips/avoid_obstacle_panda.webm" type="video/webm"></video><figcaption>The hand holds its pose while a 6 cm sphere sweeps through it at 1 mm per tick. Without the rows (left) it passes through the hand; with them (right) the arm gives way up to 114 mm and keeps 20 mm away.</figcaption></figure>

The obstacle `[x, y, z, radius]` lives in a tensor that the distance function reads, and the demo moves it in place
(`obstacle.copy_(...)`) before each tick, from a second transform handle named "obstacle". The rows see the
obstacle where the tick starts, so one that closes in by `v` per tick would end it `v / gain` inside the margin: the
demo pads the radius by the handle's move over the gain each tick, and the readout shows the padding.

## Self-collision

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_self_collision_panda.png" width="960"><source src="../../clips/avoid_self_collision_panda.webm" type="video/webm"></video><figcaption>The target goes down and back toward the robot's base. Without the rows (left) the hand's spheres go 25 mm into the base's; with them (right) the closest pair stays 20 mm apart.</figcaption></figure>

The pairs are `collision.self_collision_pairs(model, spheres, ignore=ignore)` with the ignore list of the sphere
file (`collision.load_spheres`), and the rows are the 8 closest of them each tick (the G1: 64),
`terms.self_clearance(spheres, pairs=pairs, margin=0.02, hinge=False, closest=8)`.

## Settings that matter

| Setting | Demos | Why |
|---|---|---|
| `step` | 5 mrad per tick (Panda), 6.3 mrad (G1), 20 mrad for the joint limits | curved rows hold within 1e-4 m at these steps; at 25 mrad they leave up to about 1e-3 m |
| `gain` | 0.5 | the share of a row's distance one tick may use |
| `margin` | 20 mm | the clearance the rows keep |
| `closest` | 8 (Panda), 64 (G1 self-collision) | rows per tick; the same count every tick, so the warm start fits |
| `iterations` | 20 (the default) | one behaviour at a time; the G1 with several behaviours as rows at once needs about 100 |
| cost `weight` | 1e3 | the cost form's hinge; it leaves a few millimetres inside the margin |

The scripted runs print, per form (CPU, float64, one thread; the tick is one `track` call):

| Demo | Without | With the rows | Tick with rows |
|---|---|---|---|
| joint limits (Panda) | 1.58 rad past joint 7 | 0 | 2.4 ms |
| singular poses (Panda) | `w` = 0.005 | `w` = 0.030 | 2.8 ms |
| floor (Panda) | sphere 160 mm into the floor | 20.0 mm above | 3.2 ms |
| floor (G1) | 25 mm into the floor | 20.0 mm above | 6.9 ms |
| obstacle (Panda) | 90 mm inside | 20.9 mm away | 3.2 ms |
| self-collision (Panda) | pair 25 mm into each other | 20.0 mm apart | 3.1 ms |

## Limits

- Curved rows (clearances, manipulability) hold for small steps only; [How far the rows
  hold](track-targets.md#how-far-the-rows-hold) has the measurements. The joint-limit box holds at any step.
- The robot moves at most `step` per tick, so a fast drag leaves the target behind; the loop runs at most at 100
  Hz.
- An obstacle dragged faster than the arm can give way pushes into it: the rows are elastic and give way when no
  step holds them all.
- The spheres contain the meshes and overshoot them: 7-37 mm on average per body, and 65 mm on the G1's head, which
  is two spheres, so a hand near the head stops early.
- Self-collision skips the frame pairs of the file's ignore list; the "neutral" ones touch at the neutral pose (the
  Panda's hand and fifth link, fifth and seventh link, the fingers; the G1's knees and feet) and are never checked,
  though they can collide elsewhere.
- Start from a pose where every row holds, such as the Panda's ready pose (the scripts do).

## Where next

- [Track moving targets](track-targets.md): the loop, the rows and their measurements, compiling the tick.
- [Collision](../concepts/collision.md): the sphere sets and the distances behind the rows.
- [Viewer](../concepts/viewer.md): the layers, the GUI and the gizmo these scripts use.
