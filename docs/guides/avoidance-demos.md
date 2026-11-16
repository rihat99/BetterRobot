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

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_joint_limits.png" width="960"><source src="../../clips/avoid_joint_limits.webm" type="video/webm"></video><figcaption>The target turns 4.5 rad about the hand's axis and back. Without limits (left) the last joint turns 1.58 rad past its range (red marker); with the box (right) it stops at its limit (amber) and the other joints turn the hand, which leaves the target by up to 45 mm.</figcaption></figure>

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

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_floor_g1.png" width="960"><source src="../../clips/avoid_floor_g1.webm" type="video/webm"></video><figcaption>The G1's left-hand target goes 1.05 m down and 30 cm forward, below the floor, and back, while the other hand and both feet hold their poses. Without the rows (left) the spheres at its ankles go 25 mm into the floor as it crouches (red); with them (right) every sphere stays 20 mm above.</figcaption></figure>

Every sphere of the robot's set is a row, `terms.sphere_clearance(spheres, to_floor, margin=0.02, hinge=False,
closest=8)` (the G1: `closest=None`, every sphere), with `to_floor` from `collision.sphere_halfspace`. The spheres
that stand on the floor by design, the Panda's base links and the G1's feet, get 1 m added to their distance, so
their rows never press. On the G1 the left hand follows the target and the other hand and both feet hold their
start poses, so it bends at the hips and knees to reach down.

## Obstacle

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_obstacle_panda.png" width="960"><source src="../../clips/avoid_obstacle_panda.webm" type="video/webm"></video><figcaption>The hand holds its pose while a ball of 6 cm radius sweeps through it at 1 mm per tick. Without the rows (left) it passes through the hand; with them (right) the arm gives way up to 115 mm and keeps 20 mm away.</figcaption></figure>

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_obstacle_g1.png" width="960"><source src="../../clips/avoid_obstacle_g1.webm" type="video/webm"></video><figcaption>The G1's hands hold their poses while the same ball sweeps sideways through them. Without the rows (left) it passes through the hands; with them (right) the left hand gives way up to 136 mm and every sphere keeps 21 mm away.</figcaption></figure>

The obstacle `[x, y, z, radius]` lives in a tensor that the distance function reads, and the demo moves it in place
(`obstacle.copy_(...)`) before each tick, from a second transform handle named "obstacle". The rows see the
obstacle where the tick starts, so one that closes in by `v` per tick would end it `v / gain` inside the margin: the
demo pads the radius by the handle's move over the gain each tick, and the readout shows the padding.

## Self-collision

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_self_collision_panda.png" width="960"><source src="../../clips/avoid_self_collision_panda.webm" type="video/webm"></video><figcaption>The target goes down and back toward the robot's base. Without the rows (left) the arm folds into itself and two of its spheres overlap by 25 mm (red); with them (right) every pair stays 20 mm apart.</figcaption></figure>

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/avoid_self_collision_g1.png" width="960"><source src="../../clips/avoid_self_collision_g1.webm" type="video/webm"></video><figcaption>The G1's left-hand target goes through its torso toward the right side and back. Without the rows (left) the arm passes through the torso (red: pairs inside their margins, the deepest by 140 mm); with them (right) it stops at the torso, every pair within 0.01 mm of its margin.</figcaption></figure>

The pairs are `collision.self_collision_pairs(model, spheres, ignore=ignore)` with the ignore list of the sphere
file (`collision.load_spheres`): every one of them. Each keeps a margin of 20 mm, or its distance at the start less
1 mm where that is smaller, so that the start holds every row (the Panda's ready pose has no pair closer than
21 mm). On the standing G1:

```python
import torch
from robot_descriptions import g1_description

import better_robot as br
from better_robot import collision, forward_kinematics, terms

model = br.load(g1_description.URDF_PATH, root_joint=br.JointFreeFlyer(), dtype=torch.float64)
q0 = model.q_neutral.clone()[None]
q0[0, 2] = 0.793                                                                   # the soles on z = 0
spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)
pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)              # (P, 2)
apart = collision.self_distances(spheres, forward_kinematics(model, q0), pairs=pairs)[0]  # (P,) at the start
margin = (apart - 1e-3).clamp(max=0.02)                                            # (P,) one per pair
row = terms.self_clearance(spheres, pairs=pairs, margin=margin, hinge=False, closest=64)
print(f"{len(pairs)} pairs, {int((apart < 0.02).sum())} closer than 20 mm, the closest {1e3 * apart.min():.2f} mm")
```
```output
3890 pairs, 58 closer than 20 mm, the closest 2.45 mm
```

The rows are the 8 pairs closest to their margins each tick (the G1: 64, with `iterations=100`, since many pairs
press at once). The readout shows the deepest pair's
distance outside its own margin.

## Settings that matter

| Setting | Demos | Why |
|---|---|---|
| `step` | 5 mrad per tick (Panda), 6.3 mrad (G1), 20 mrad for the joint limits | the Panda's curved rows hold within 1e-4 m at these steps; the G1's, with `iterations=100`, leave up to about 4e-4 m on some targets; at 25 mrad rows leave up to about 1e-3 m |
| `gain` | 0.5 | the share of a row's distance one tick may use |
| `margin` | 20 mm; per pair for self-collision, at most the pair's start distance less 1 mm | the clearance the rows keep; a start inside a margin would begin with violated rows |
| `closest` | 8 (Panda), 64 (G1 self-collision) | rows per tick; the same count every tick, so the warm start fits |
| `iterations` | 20 (the default); 100 for the G1's self-collision | QP iterations per tick: the G1 needs 100 whenever its self-collision rows are present, and with several behaviours as rows at once |
| cost `weight` | 1e3 | the cost form's hinge; it leaves a few millimetres inside the margin |

The scripted runs print, per form (CPU, float64, one thread; the tick is one `track` call):

| Demo | Without | With the rows | Tick with rows |
|---|---|---|---|
| joint limits (Panda) | 1.58 rad past joint 7 | 0 | 2.4 ms |
| singular poses (Panda) | `w` = 0.005 | `w` = 0.030 | 3.0 ms |
| floor (Panda) | sphere 160 mm into the floor | 20.0 mm above | 3.6 ms |
| floor (G1) | 25 mm into the floor | 20.1 mm above | 7.2 ms |
| obstacle (Panda) | 90 mm inside | 21.2 mm away | 3.4 ms |
| obstacle (G1) | 94 mm inside | 21.3 mm away | 7.4 ms |
| self-collision (Panda) | a pair 45 mm inside its margin (25 mm into each other) | every pair outside its margin | 3.3 ms |
| self-collision (G1) | a pair 140 mm inside its margin | every pair within 0.003 mm of its margin | 9.0 ms |

## Limits

- Curved rows (clearances, manipulability) hold for small steps only; [How far the rows
  hold](track-targets.md#how-far-the-rows-hold) has the measurements. The joint-limit box holds at any step.
- A robot that moves along a row rides slightly short of it, by the reserve `track` keeps (about 0.4 mm at 5 mrad
  per tick), and settles onto it when the target stops: the scripted runs end 20.0 to 21.3 mm away.
- The robot moves at most `step` per tick, so a fast drag leaves the target behind; the loop runs at most at 100
  Hz.
- An obstacle dragged faster than the arm can move pushes into the margin; the rows repair the violation once it
  slows. One pressed between two links, or onto a hand that another row holds, can leave the arm no direction: it
  waits until the obstacle moves.
- With the joint-limit rows the hand leaves the target by up to 45 mm while the other joints turn it: the pose
  residual weighs 1 rad of rotation like 1 m.
- The spheres contain the meshes and overshoot them: 7-37 mm on average per body, and 65 mm on the G1's head, which
  is two spheres, so a hand near the head stops early.
- Self-collision skips the frame pairs of the file's ignore list; the "neutral" ones touch at the neutral pose (the
  Panda's hand and fifth link, fifth and seventh link, the fingers; the G1's knees and feet) and are never checked,
  though they can collide elsewhere. A pair that starts closer than 20 mm keeps only its start distance less 1 mm.
- Start from a pose where every row holds, such as the Panda's ready pose (the scripts do).

## Where next

- [Track moving targets](track-targets.md): the loop, the rows and their measurements, compiling the tick.
- [Collision](../concepts/collision.md): the sphere sets and the distances behind the rows.
- [Viewer](../concepts/viewer.md): the layers, the GUI and the gizmo these scripts use.
