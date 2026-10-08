# P1 — adversarial architecture review (one theme per agent)

Read `context.md` in this folder first. You are a critical reviewer preparing BetterRobot (BR) and BetterHuman (BH)
for growth. The owner's words: "Our project is still small and has huge potential for improvement and growth, so we
should prepare for future development now. We don't implement them now; show what we miss, have a roadmap. We should
be like LEGO bricks: modularity, coherence, convenience for expansion."

## Method (all themes)
1. Read the current code of your theme in full (not just docs), the matching `CLAUDE.md`, contracts and concept pages.
2. Study how the reference projects in `references/` handled the same area: their design digests
   (`references/design/*.md`), the actual source, and the web (their changelogs, issues, design docs, papers) —
   what they started with, what they had to redesign later, and why. Cite file paths or URLs for every claim.
3. Attack our design: where would a user hit a wall? Where does adding a feature force a rewrite instead of a new
   brick? What is coupled that should not be? What abstraction is missing or premature? What is slow by
   construction? Check claims by running small snippets (`uv run python -c ...`) where cheap.
4. Produce a roadmap for your theme: concrete future bricks, each with: what, why (which user), what must already be
   true in the architecture to add it cleanly, and what we should change NOW (small, non-breaking) to keep the door
   open. Mark each item: `now` (small prep change worth doing before merge), `next` (first thing after merge),
   `later`, `maybe`.

## Deliverable
Write `agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/subagents/p1_<theme>.md` (≤ 400 lines) with sections:
`## Current state (facts, with paths)`, `## How references evolved (with sources)`, `## Findings (ranked, F1..Fn:
severity, evidence, consequence)`, `## Roadmap (table: brick | why | prerequisite | when)`, `## Prep changes
recommended now (each ≤ 1 day, non-breaking)`, `## Open questions for the owner`. Final message: ≤ 20 lines summary
+ the path. Do not edit any source file.

## Themes

### optim — `p1_optim.md`
`src/better_robot/optim`, `src/better_robot/terms`, BH `terms.py`. Beyond least squares: general NLP (constraints,
barriers, augmented Lagrangian, SQP), native sparsity (block-sparse Jacobians, sparse linear solves, Schur
complement for trajectory problems), trajectory optimization / MPC (DDP/iLQR as in crocoddyl), differentiable
solve (theseus), warm starts, batching semantics, convergence diagnostics, custom term authoring ergonomics,
torch.optim interoperability. References: theseus, jaxopt, optax, crocoddyl, pyroki, curobo, mink, drake (MathematicalProgram).

### kindyn — `p1_kindyn.md`
`src/better_robot/{lie,spatial,data_model,kinematics,dynamics}`. Speed and completeness of kinematics and
dynamics: RNEA/ABA/CRBA coverage, derivatives (analytical vs autograd), centroidal dynamics, contact dynamics,
constrained dynamics, joint types, closed loops, model reduction/composition, batching/vmap/compile friendliness,
GPU kernels (warp/triton/CUDA) as a future brick without the "duality" the owner disliked, dtype/device policy,
benchmark and profiling infrastructure (`benchmarks/`). References: pinocchio (incl. its derivative/codegen
history), mujoco_warp, newton, warp, brax, drake, pypose, jaxlie, frax, IsaacLab/mjlab.

### collision — `p1_collision.md`
`src/better_robot/collision` and its terms. Primitive set, mesh collision (convex hulls, SDFs), broadphase,
self-collision pairs, differentiable distance, contact geometry for dynamics, environment/scene representation,
how geometry is attached to the model (`data_model`), how parsers (`io`) bring geometry in. References: curobo,
pinocchio/hpp-fcl (coal), mujoco, drake (SceneGraph), newton, warp.

### viewer — `p1_viewer.md`
`src/better_robot/viewer`, BH `viz/`, BVR `tools/viewer`. Layer/tool architecture, timeline, recording, tools
(gizmo, picking, marquee, measurement), forces/torques/contacts/trajectories/plots overlays, cameras/video
overlay (BVR), multi-model scenes, scene graph vs layers, viser's limits and alternatives (rerun, meshcat, own
three.js client), headless rendering, plugin points for a project-specific studio. Find concrete BVR viewer
features we lack. References: BVR `tools/viewer`, viser (web), rerun (web), drake meshcat.

### human — `p1_human.md`
BH `src/better_human` whole package + `docs/`. Architecture of body families (smpl/smplx/mhr/…), shape/pose
spaces, segmentation → bodies → dynamics, contact frames, reduction, retargeting (mink, GMR, soma-retargeter),
fitting pipelines (BVR needs), measurements (GarmentMeasurements), clothing/anny, model asset handling, how BH
reuses BR bricks vs duplicates them, what BH-specific bricks should move to BR. References: BH `references/`.

### toolbox — `p1_toolbox.md`
Whole BR public API against the general robotics toolbox: what a robotics researcher expects from a library
(Pinocchio/Drake/MuJoCo/OpenSim users): model editing & composition, frames/placements, joint limits & calibration,
kinematic chains/closed loops, trajectories (splines, retiming), time-series/state types, planning hooks, control
(PD, impedance, QP-WBC), simulation hook (integrators), units/conventions, serialization/formats (URDF, MJCF, USD,
SDF), logging, config, testing utilities, error types, naming consistency across modules, `io` layer. Spot API
inconsistencies between modules (same concept, different name/signature) — these matter most before merge.
