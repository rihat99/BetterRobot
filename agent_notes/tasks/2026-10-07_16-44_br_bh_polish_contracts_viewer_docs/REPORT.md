# REPORT — BetterRobot + BetterHuman: contracts, polish, viewer, studio, docs

Task `2026-10-07_16-44_br_bh_polish_contracts_viewer_docs`, 2026-10-07 16:44 → 2026-10-08 ~02:30 Dubai.
Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot` (branch `dev`), BH
`/home/rikhat.akizhanov/better/dev/BetterHuman` (branch `dev`). Nothing pushed or merged.

## What was asked

Five workstreams (owner, verbatim intent in `PLAN.md`): (1) polish the code and trim the tests, (2) design
contracts that the whole codebase follows, (3) guidebook-quality docs with a navy theme, output panes, SVG
figures, clips and math, also for BetterHuman, (4) a light, navy, user-friendly viewer with joint shapes and no
own rasterizer, (5) a HumanStudio that actually works, plus the gizmo-IK lag and API sugar. Delegate heavily.

## Commits (all on `dev`)

| Repo | Commit | Phase | Content |
|---|---|---|---|
| BR | `6ec3134` | P1–P3a | `docs/contracts/` (10 pages, 47 `C-*` rules, 6 ADRs), `tests/contract/`, test trim (586 → 469 functions), viewer batch 1 |
| BR | `2f7c6e2` | P2b/P3a2 | keyword-only API, typed errors (`IRError`), joint shapes in `SkeletonLayer`, `RobotView` slots |
| BR | `31a37ed` | P2d/P2e/P3b | `Viewer.record` via headless Chromium (rasterizer `viewer/render/` trashed), `int \| str` ids, API sugar (`model.fk`, `result["q"]`, `CacheRef`, `br.load(dtype=, device=)`), latest-wins gizmo worker |
| BR | `171509d` | P5a/b | guidebook docs: Material navy theme, 78 output panes, 12 SVG figures, 5 clips + README GIF, rewritten index/architecture/README |
| BR | `2892301` | P5d | 25 docs-vs-code review findings fixed (see below) |
| BH | `e898fd5` | P1/P3 | contracts addendum (`H-*`), viewer API rename, test merges |
| BH | `83cc96e`, `81ca789` | P4 | studio rebuilt on the BR viewer: model chooser (smpl/smplx/mhr, gender, LOD), one mesh with skin/segmentation/parts modes, frames layer, light theme, every control driven on a real server (`subagents/drive_studio.py`, 53 steps) |
| BH | `bad0de3` | P5c | docs site in the same style (index, getting started, architecture, segmentation, studio, contracts, roadmap) |

## Gates at the end

| Repo | Tests | Ruff | Docs |
|---|---|---|---|
| BR | 1580 passed, 11 skipped (`-m "not bench and not cuda"`) | 82 pre-existing findings in untouched files (PLC0415/FBT/PLR in parsers, viewer, bench); touched files clean | `mkdocs build --strict` clean; `tests/docs` runs every snippet |
| BH | 332 passed, 1 skipped, 1 failed — `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`, pre-existing, untouched | clean | strict clean; `tests/docs` 22 passed |

## Owner decisions applied

Headless Chromium instead of the rasterizer (no Playwright dependency — `$BETTER_ROBOT_CHROME`); test trim
−22 % per `subagents/p0_tests.md` (owner ran `drop_tests.py --apply`, 76 drops; ledger `p2_tests_ledger.md`);
order contracts → polish → viewer → studio → docs; studio chooser smpl/smplx/MHR; time axis `step`/`nsteps`;
Drake monogram names only inside math locals; all four API sugar items.

My own small choices: `mkdocs>=1.6,<2` pin (both repos), Inter / JetBrains Mono from Google Fonts, clips rendered
on CPU (no GPU was free), ffmpeg optional (VP8 webm only on this machine).

## Docs-vs-code review (P5d)

Two read-only reviewers found 25 items (`subagents/p5d_review_{a,b}.md`), all fixed in `2892301`: the viewer
"renders offline to MP4" wording (it records through headless Chromium to webm/gif/png), reference frames shown as
enum names (the API takes `"local" | "world" | "local_world_aligned"`), hidden `*` in optim signatures,
`joint_jacobian` takes an index only, C-DOC-2/C-DOC-3 now cite their tests, planned ruff codes tagged in the
matrix, three agent-guide signature errors, stale `frame` wording, a duplicated enforcer, a literal `--`.

## Where to look

- Docs: `uv run mkdocs serve` in either repo (BR was live on http://127.0.0.1:8765 during the task).
- Figures `docs/figures/*.svg` (generator `subagents/make_figs.py` for the 5 new ones), clips `docs/clips/`
  (generator `subagents/make_clips.py --out docs`), screenshots `vis/p5final/{light,dark}/` (gitignored).
- Studio: `BETTERHUMAN_MODELS_DIR=$PWD/models uv run python -m better_human.viz.studio --family smplx`.
- Ledgers: `subagents/p1_violations.md` (L-1..L-9 + P2 status), `subagents/p2_tests_ledger.md`.

## Left open (for the owner)

1. Ruff rules deferred by decision: `N`, `D`, `ANN001`, `SLF001` (ledger L-7); the 82 pre-existing findings.
2. Ledger items L-1 (consumers still `isinstance` on joint classes), L-4 (hot-path lint scope to `spatial`/`terms`),
   L-6 (`state_manifold` detach docstring), L-8 (vmap test per public entry point).
3. Figures 2–8 were restyled, not redrawn; `viewer/_math.py::as_frames` and the timeline's "Frame counter" label
   still use the old word (outside the docs).
4. Studio cosmetics: finger-joint spheres too large, a stray marker top-right in `docs/images/studio_segmentation.png`.
5. Nav-tab labels look faint in the full-page screenshots; Material animates them in and the capture catches the
   fade — check once in a real browser.
6. Guides `differentiate`, `write-a-term`, `own-the-loop`, `benchmark-and-profile` kept their instruction style
   (visuals and "Where next" added, prose not rewritten); `panda_trajectory` clip is 3.3 s, under the 4 s target.
7. BH: `pyproject.toml` dev group now lists mkdocs/material/mkdocstrings (same pins as BR) — a manifest change
   made for the docs site; `uv sync --group dev` alone drops viser, use `--all-extras`.
8. Trash to empty after review: `/home/rikhat.akizhanov/better/BetterRobot/trash/` (replaced docs PNGs, retired
   collision/geometry modules), `/home/rikhat.akizhanov/better/dev/BetterHuman/trash/` (old segmentation guide) and
   `/home/rikhat.akizhanov/better/BetterHuman/trash/` (a studio scratch file).
