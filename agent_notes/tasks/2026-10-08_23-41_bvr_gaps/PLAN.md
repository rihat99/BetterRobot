# PLAN — generic gaps BVR needs from BR dev (optim diagnostics + viewer)

Origin: BVR task `dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite` (step G);
investigations `inv_A_br_optim.md` §3 and `inv_D_viewer.md` §1–3 there. BVR moves its optimization engine and
viewer onto BR dev; these are the generic pieces it is missing. Smallest generic form, one commit per item,
each with a test and a line in the matching `docs/concepts` page.

Baseline: `uv run pytest tests/ -q -m "not bench and not cuda"` → 1652 passed, 11 skipped, 20 deselected (3m26s).

## Items
1. optim — per-term diagnostics from the same evaluation. `Problem.evaluate(x) -> (loss, terms, info)`; a term
   may return `(value, info: dict)` (info detached, never in the loss). Every solver writes
   `State.extra["terms"]` `{name: (B,)}` (the evaluation `State.loss` comes from); `TorchOptim` also writes
   `State.extra["info"]`. Tensors stay on device (C-TENS-4: no host sync in solvers.py). `Phase.on_step` reads them,
   no second forward. Existing API unchanged (tensor-returning terms, `losses`, `loss`).
2a. viewer — `Viewer.add(..., valid=)`: `(T,)` hides the whole layer on invalid samples (any layer);
   `(T, N)` hides single spheres/arrows (`SpheresLayer`, `ArrowsLayer`).
2b. `remove_group` shrinks the timeline to the remaining layers (`Timeline.resize`).
2c. `ImagePanel(name, images, *, nsteps=None)`: sidebar image per step, `(T,H,W,3)` or lazy `step -> image`.
2d. per-step colours `(T, N, 3)` for `SpheresLayer` and `ArrowsLayer`.
2e. `GroundLayer(origin=, normal=, fill=, fill_color=)`: tilted plane + viser's native plane fill; default unchanged.
2f. fps: check `times=` first; add `Viewer.set_fps` only if needed.

## Verify
Full suite before/after; new tests; `examples/viewer/02_play_trajectory.py` headless with the fake server.
