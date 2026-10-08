# REPORT — generic gaps for BVR: per-term optim diagnostics + six viewer additions

Task `2026-10-08_23-41_bvr_gaps` (step G of BVR `2026-10-08_22-50_dev_optim_rewrite`), branch `dev`, nothing pushed.
One commit per item, each with a test, a `docs/concepts` line and a changelog line.

| Item | Commit | API |
|---|---|---|
| 1 optim diagnostics | `72a7d02` | term may return `(value, info)`; `Problem.evaluate(x) -> (loss (B,), terms {name: (B,)}, info {name: dict})`; every solver writes `State.extra["terms"]` (point of `State.loss`; LM/GN/L-BFGS select per batch element); `TorchOptim` also `extra["info"]` |
| 2a validity | `0c3d167` | `Viewer.add(layer, ..., valid=)`: `(T,)` hides any layer per sample (switch untouched); `(T, N)` for `SpheresLayer`/`ArrowsLayer` |
| 2b timeline shrink | `b6602aa` | `remove_group` fits the timeline to the remaining layers; `Timeline.resize(nsteps)` |
| 2c image panel | `64d3fb5` | `ImagePanel(name, images, *, nsteps=None, format="jpeg")`, `(T,H,W,3)` / `(H,W,3)` / `step -> image` |
| 2d per-step colours | `2dd0a9d` | `SpheresLayer(..., colors=(T,S,3))`, `ArrowsLayer(..., colors=(T,N,3))` |
| 2e ground | `1f3fbe3` | `GroundLayer(..., origin=(3,), normal=(3,), fill=opacity, fill_color=RGB)`; default unchanged |
| 2f fps | `4b7b1da` | `Viewer.set_fps(fps)`; `times=` alone was not enough (a step stops being a frame; hand-set `timeline.fps` mis-sized timed layers and jumped during playback) |

## Decisions (small, reversible)
- Per-term values are detached `(B,)` device tensors, not floats: `optim/solvers.py` is under the no-host-sync lint
  (C-TENS-4). The logger converts. They are the weighted contributions (`w · value`), the same as `Problem.losses`.
- `info` only from `TorchOptim`: second-order states mix accepted and rejected candidates per batch element and
  `info` has no batch contract. LM/GN/L-BFGS still accept tuple-returning terms and drop the info.
- `TorchOptim.init` writes no `terms` (it evaluates nothing before the first step); `LBFGS.init` does one extra
  no-grad forward per phase start for its initial `terms`.
- `valid` enters through `Viewer.add` like `times`, so BetterHuman layers get it with no change. Points take only
  `(T,)`: viser cannot hide single points (pad as documented).
- No `Viewer.add_image` helper: `viewer.add(ImagePanel(...), group=...)` suffices (`add_plot` exists for its fps).

## Verification
- Full suite (`uv run pytest tests/ -q -m "not bench and not cuda"`): before 1652 passed / 11 skipped / 20 deselected;
  after 1673 passed / 11 skipped / 20 deselected (+21: 11 new test cases, 10 contract cases auto-parametrized over
  the new public names — keyword-only and optional-imports).
- Downstream: BetterHuman dev (its venv imports this BR dev tree) `pytest tests`: 317 passed, 6 skipped, 17 errors/failures
  all `FileNotFoundError` for SMPL weights (no `$BETTERHUMAN_MODELS_DIR`); `tests/viz` rerun with it set: 72 passed.
- Ruff on `src/better_robot/viewer` + `tests/viewer`: 11 findings before and after (pre-existing FBT001/FBT002/BLE001;
  the new `ImagePanel.set_visible` carries a reasoned `noqa`). `ruff check src/better_robot/optim tests/optim` clean.
- `examples/viewer/02_play_trajectory.py --check`: runs with the fake server (60 atomic pushes, 19 nodes) and with a
  real loopback viser server. `examples/03_clip_fit_phases.py` unchanged output.
- Real viser 1.0.26 smoke (every new prop accepted) and a headless-Chromium `record` of a tilted filled ground,
  per-step sphere/arrow colours, a per-element and a whole-layer mask: rendered as intended (4 PNGs, scratchpad).

## Not done
- `solve_constrained` constraint terms must still return a plain tensor (its shifted wrapper does not unpack info).
- A `PlotPanel` keeps the time axis it was built with after `set_fps` (documented).
- No adversarial review was run (subagent brief; the parent owns it).
