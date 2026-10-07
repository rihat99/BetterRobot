# PLAN — BetterRobot + BetterHuman: contracts, polish, viewer, studio, docs (2026-10-07_16-44)

Owner request (verbatim intent, 2026-10-07 16:30 Dubai): the docs rewrite went in the right direction; now
(1) polish the code and trim the tests, (2) write design contracts and make the whole codebase follow them,
(3) redo the docs to guidebook quality (navy blue, highlighted code, separate output panes, SVG figures, GIFs,
math where useful; BetterHuman gets the same), (4) polish the viewer (light navy theme, hide play/speed
without a trajectory, joint-specific shapes, cleaner sidebar, viser default lights, no own rasterizer),
(5) make HumanStudio actually work (light theme, in-app model chooser, every tool working), fix the laggy
IK gizmo example, propose API sugar. Delegate heavily.

## Decisions taken at intake
- Both repos on their `dev` worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`,
  BH `/home/rikhat.akizhanov/better/dev/BetterHuman`. Commit per phase on `dev`; never push/merge.
- Offline rendering: drop `better_robot/viewer/render` (own rasterizer); `Viewer.record` screenshots the real
  viser scene in headless Chromium (Playwright, already installed) → PNG / GIF / MP4. Optional dev extra.
- Test trim: moderate, about −40 % of BR's ~570 tests. Pinocchio parity and contract tests stay. Every dropped
  test is listed with its reason. Never skip/xfail/weaken.
- Order: P1 contracts → P2 polish + tests → P3 viewer → P4 studio → P5 docs (BR then BH).
- Studio model chooser: smpl, smplx, MHR (gender / LOD sub-options).
- Rules in force: trash not rm; 127.0.0.1 only; no new deps without asking (Playwright is approved as a dev
  extra); CPU only unless `gpu_free.py` says FREE; mars_* agents only, ≤6 parallel, one `mars_top`.

## Phases → verification
P0 discovery (mars_aux ×5, read-only): architecture + legacy audit (BR, BH), test inventory with trim
   candidates, viewer/studio audit (what is broken, how), docs tooling (Material theme customisation, GIF
   pipeline), external contract/design references. Output: `subagents/p0_*.md`.
P1 contracts: `docs/contracts/` (one page per contract family) + root `CONTRACTS.md` pointer in both repos;
   `tests/contract/` enforces what is mechanically checkable. Verify: contract tests pass; adversarial review.
P2 polish + trim: legacy leftovers out; tests trimmed to target with a dropped-test ledger; API sugar list
   proposed (owner picks). Verify: full suites green, ruff clean on touched files, coverage of public API
   unchanged (every public symbol still exercised by ≥1 test or docs snippet).
P3 viewer: theme (light, navy), conditional playback UI, joint shapes, sidebar, default lights, headless
   record, gizmo example lag fixed. Verify: `tests/viewer`, screenshots reviewed by me, examples run.
P4 studio: rebuild around the contracts, model chooser, light theme, every tool verified by a scripted
   headless session + screenshots. Verify: `tests/viz`, screenshot review, owner demo.
P5 docs: theme (navy, highlighting, output panes), SVG figures, GIFs via headless record, BR then BH site.
   Verify: `mkdocs build --strict`, snippet tests, rendered pages reviewed as PNG, adversarial docs-vs-code pass.

Each phase: REPORT section + commit on `dev` + STATE.md checkpoint. Compact only at phase boundaries.
