# P5b-2 brief — SVG figures for the BetterRobot docs

Read `context.md`, then the figure skill `~/.claude/skills/figure/SKILL.md` (style rules; `scripts/render.sh`
to render and inspect; `templates/example.svg`), then the existing `docs/figures/*.svg` (8 files) and the pages
that embed them (`grep -rn "figures/" docs`). Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`. Scope:
`docs/figures/**` only (writers embed them; keep existing filenames so links stay valid; new files listed in the
final message with the page they are meant for). Accent colour: navy `#1f3a68` (the docs theme), not the
skill's default blue; dark-mode: the site shows SVGs on a dark background too, so use `fill="#f3f5f8"` boxes
with `#1f2328` strokes/text — they stay readable on both (do not use transparent backgrounds with dark text only;
give each figure a white rounded background rect).

Owner's words: "More visuals, better visuals. Better figures with SVG. After reading I want to fully understand
the overall architecture and high-level ideas." Each figure: one message, ≤ 9 shapes, labels inside shapes,
one accent path. Render each with `render.sh`, LOOK at the PNG (Read tool), fix, re-render. Verify every claim
in a figure against the code (package names, function names, shapes) — figures are checked by a reviewer later.

Figures (rework the existing 8 where they fall short of the rules; add the new ones):
1. `architecture.svg` — the package DAG as the code has it now (`tests/contract/test_layer_dependencies.py`
   ranks; terms→optim edge; viewer on top). Message: "optim knows nothing about robots; everything else stacks
   on data_model."
2. `model_values_state.svg` — Model = ModelStructure (static) + ModelValues (tensors) → `forward_kinematics_state`
   → frozen KinematicState (joint/frame transforms) → Jacobians / dynamics read it. Include `model.fk(q)`.
3. `joints_bodies_frames.svg` — a 3-joint chain: joints (kinds, nq/nv), bodies (inertia), frames (named points);
   placement vs joint transform.
4. `lie_spatial.svg` — SE3 pose `[t, q_xyzw]`, tangent `[v, ω]`, exp/log, adjoint; spatial Motion/Force/Inertia.
5. `jacobians.svg` — frame Jacobian reference frames (local / world / local-world-aligned) as three small
   sketches of the same arm; which is the default.
6. `optim_problem.svg` — Problem(variables, manifolds, bounds, data) + residual/scalar/cached terms → linearize →
   solver (LM / GN / L-BFGS / torch.optim) → Result; `solve`, `run(phases)`, `solve_layer`, `solve_constrained`
   as four entry doors. Include the CacheRef handle idea.
7. `banded_normal.svg` — why time-structured problems give a banded normal matrix (windows of width w).
8. `viewer_layers.svg` — Viewer = viser server + groups of Layers + Timeline (only for nsteps>1) + Layers panel;
   `record` → headless Chromium → webm/png.
9. NEW `pipeline_overview.svg` — the one-screen story for the landing page: URDF/MJCF/builder → Model → fk/
   Jacobians/dynamics → terms → optim → viewer/record. Message: "one tensor path from file to solved motion."
10. NEW `ik_loop.svg` — what `solve_ik` does per iteration (residual → J → damped normal eqs → retract on manifold
    → bounds). Small, for the IK tutorial.
11. NEW `contracts_map.svg` — the 11 contract families as a grid with their enforcer kind (ruff / test / review),
    for `docs/contracts/00-index.md`.
12. NEW `time_structure.svg` — knots × variables, windows, the `time=("q",)` axis, finite-difference terms.

Final message ≤ 25 lines: per figure, path + one-line message + "checked PNG: ok/fixed what". Do not spawn
agents. Do not commit. Budget: render-and-look once per figure; if budget runs low, finish the current figure
and list the rest.
