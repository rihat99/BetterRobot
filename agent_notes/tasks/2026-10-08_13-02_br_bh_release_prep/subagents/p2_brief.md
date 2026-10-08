# P2 — release sweep (one slice per agent) and studio changes

Read `context.md` in this folder first. The owner wants a final check of every file before `dev` merges to main:
minor bugs, computational correctness, bottlenecks. Owner's policy: **fix small local bugs** (wrong index, missing
guard, dead path, off-by-one, wrong dtype/device handling, stale docstring/comment, broken example) directly, each
with a pinning test when it is a behaviour bug; **report, do not change** anything that touches API shape,
numerics/conventions, or needs a design decision. Keep diffs minimal and in-style. Other agents edit other
slices of the same worktree concurrently: touch only files in your slice (plus their tests); run only the tests of
your slice while iterating, the full suite once at the end.

## Method (sweep slices)
1. Read every `.py` file of your slice end to end, and the tests that cover it. Keep a checklist of files read.
2. For each public function: check shapes/broadcasting with batch dims, dtype/device preservation, nq≠nv handling,
   edge cases (empty batch, single joint, fixed vs floating base), gradient flow (no accidental `.detach()`,
   in-place ops on leaves), and that docstrings match behaviour. Run tiny probes with `uv run python -c`.
3. Correctness against references where cheap: Pinocchio is installed in the BR env (`import pinocchio`) — use it
   for spot checks of kinematics/dynamics/Jacobians on a random configuration if your slice is numerical.
4. Bottlenecks: look for Python loops over batch or joints that could be vectorised, repeated allocations, `.item()`
   / `.cpu()` syncs on the hot path, unnecessary `clone()`s, `torch.linalg` calls that could be batched. Measure with
   `torch.utils.benchmark` before claiming a bottleneck (numbers in the report). Do not optimise unless the fix is
   local and clearly faster; otherwise report.
5. At the end: full test suite of the repo, `ruff check src tests` (no new findings), and for BR `mkdocs build --strict`
   if you touched docstrings referenced by docs.

## Deliverable
`agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/subagents/p2_<slice>.md` (≤ 250 lines): `## Files read`
(every file), `## Fixed (what, where, test)`, `## Reported, not changed (ranked; severity; evidence; suggested fix)`,
`## Performance notes (measured)`, `## Gate results`. Final message ≤ 20 lines + path. Do not commit.

## Slices

### br_core — `p2_br_core.md`
BR `src/better_robot/{lie,spatial,data_model,kinematics,dynamics}` (~6200 lines). Pinocchio spot checks for FK,
Jacobians (all three reference frames), RNEA, ABA/CRBA if present, `integrate`/`difference` on floating base.

### br_optim — `p2_br_optim.md`
BR `src/better_robot/{optim,terms,collision}` (~2300 lines) + `examples/*.py` (run each with `--check` where it
exists). LM/GN/L-BFGS/torch.optim paths, `solve_layer` gradients (check with `torch.autograd.gradcheck` on a tiny
problem), `solve_constrained`, phases, convergence/termination flags, warm start, batching semantics.

### br_io_viewer — `p2_br_io_viewer.md`
BR `src/better_robot/{io,viewer}` + `examples/viewer/*.py` (run each with `--check`), `src/better_robot/{exceptions,_typing,_validation,__init__}.py`.
Parsers (URDF/MJCF/builder): joint types, limits, inertia frames, mesh scaling, mimic joints, root handling. Viewer:
layers with empty/edge inputs (zero frames, one step), timeline, record (headless Chromium env in context.md),
threads/locks in tools (gizmo worker), theme.

### bh — `p2_bh.md`
BH `src/better_human` whole package (~6200 lines) + `examples/*.py --check`, `benchmarks/run.py --device cpu`.
Blend shapes, skinning, joint regression, `drop_joints`/reduction (the known failing test: diagnose the cause and
report it; do not change the test), mass/segmentation, contact frames, MHR rig, studio code (another agent changes
studio GUI files `viz/studio/{app,gizmo}.py` concurrently — do not edit those two; report findings about them).

### perf — `p2_perf.md`
Both repos' benchmarks: BR `benchmarks/run.py` and `benchmarks/m1_eager_cpu.py`, `tests -m bench` (read how they
are marked), results in `benchmarks/results/*.json`; BH `benchmarks/run.py`. Run them on CPU (8 threads) and, if
`gpu_free.py` says FREE, on one GPU. Compare against the stored results (same machine? check the json metadata) and
against the numbers claimed in `docs/` (search docs for "ms", "µs", "x faster"). Profile the three hottest public
calls (`forward_kinematics_state`, frame Jacobian, `solve_ik`/LM step, BH `identity.forward`) with
`torch.profiler` for one batch size each; list the top-5 ops. Report regressions and bottlenecks with numbers; fix
only trivial local ones. Save raw outputs under the task folder `artifacts/perf/` (gitignored) and reference them.

### studio — `p2_studio.md` (BH, implements owner-chosen changes; not a sweep)
Files: BH `src/better_human/viz/studio/{app.py,gizmo.py}` (+ tests under `tests/viz`), BR gizmo tool
`src/better_robot/viewer/tools/gizmo.py` only if an option must be passed through (keep it generic: forward
`**options` already reaches viser's `add_transform_controls`).
1. Gizmo usable inside the body: pass `depth_test=False` to viser's transform controls (draw through the mesh) and
   raise the default scale 0.15 → 0.20. Keep it slight; the owner does not want a big gizmo.
2. Above the tab group, add a compact row: buttons `Reset pose`, `Deselect joint`, `Deselect frame`, and a one-line
   status label showing `model · selected joint · selected frame` (e.g. `smplx neutral · joint: left_elbow · frame: —`)
   that updates on every selection/model change. Remove the two buttons from the Body tab (keep the rest). Use
   viser's `gui.add_button_group` or a folder with buttons, whichever looks compact in the sidebar; check how
   `frames_editor.py` deselects a frame and reuse that.
3. Verify on a real server (127.0.0.1, port 8082): extend/run `agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/subagents/drive_studio.py`
   (previous task; read it) so it clicks the three new buttons and selects a joint, screenshot with the headless
   Chromium helper from context.md before/after selecting a joint (gizmo visible through the body). Save screenshots
   under the task folder `vis/studio/`. Run BH tests + ruff. Confirm `git ls-files studio_data` lists the frames
   files (they are tracked already; just state it with the output).
Report in `p2_studio.md`: what changed, screenshot paths, test/ruff results, anything the owner should look at.
