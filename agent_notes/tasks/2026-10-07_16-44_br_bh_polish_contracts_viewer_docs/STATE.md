# STATE — BR+BH / 2026-10-07_16-44_br_bh_polish_contracts_viewer_docs (updated 2026-10-08 00:40 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/PLAN.md
- Subagent notes (`subagents/`): context.md; p0_*.md + p0_synthesis.md (discovery); p1_* (contracts: review,
  violations ledger, state); p2_tests_ledger.md, drop_tests.py; p3a_state.md, p3b_state.md (viewer);
  p4_state.md, drive_studio.py (studio); p5a/p5b briefs, make_clips.py, make_figs.py (docs).
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman`, both `dev`.
- Rules: commit per phase on dev only, never push/merge; trash not rm; 127.0.0.1 only; mars_* only (≤6, one mars_top);
  no new deps; never weaken tests. Owner present (working hours 10–22 Dubai; it is past 22 now → off-hours rules).

## Commits so far
BR dev: 6ec3134 (contracts, trim, viewer batch 1), 2f7c6e2 (polish, joint shapes), 31a37ed (headless record,
keyword-only, ids, sugar). BH dev: e898fd5, 83cc96e, 81ca789 (studio rebuilt + live-verified, docs/studio.md).
Owner saw the viewer demo ("looks fine"); owner answered intake + 3 decision rounds (see PLAN/p0_synthesis).

## Status: P5 docs (BR) in tree, UNCOMMITTED; clip job may still be running
- Uncommitted in BR: P5a theme (mkdocs.yml, docs/stylesheets/guidebook.css, 78 output fences, `{ .no-run }` fences,
  tests/docs/test_snippets.py extensions, scripts/docs_screenshots.mjs, pyproject mkdocs<2 pin); figures (5 drawn +
  7 restyled, docs/figures/); stills docs/images/{panda_rest,panda_ik,g1_floating,trajectory_strip}.png; writers
  A/B/C edits (index, getting-started, concepts/architecture rewritten, Where-next on all, clips embedded, README).
- Background: clip job (scratchpad/clips.log; expected outputs docs/clips/{panda_trajectory,batched_ik,g1_clip_fit,
  viewer_layers,obstacle_trajectory}.{webm,png} + docs/images/readme_ik.gif). A bash waiter (task b1gx39rr6) tails it.
  If it died: `uv run python agent_notes/.../subagents/make_clips.py --out docs --only <section>` (env vars in its
  docstring; port 8094). Check each clip ≤ 1.5 MB, gif ≤ 3 MB; look at posters.
- mkdocs serve of the dev docs still on http://127.0.0.1:8765 (task bh6kicmzx).

## Next steps
1. Confirm clips exist; `uv run pytest tests/docs tests/viewer -q` green; `uv run mkdocs build --strict`; full BR suite;
   commit BR ("Guidebook docs: theme, output panes, figures, clips").
2. Docs-vs-code adversarial pass (mars_aux ×2: getting-started+concepts / guides+contracts+README), fix findings.
   Known: architecture.svg says KinematicState lives in data_model (it is kinematics/state.py) — fix the SVG label.
   Second guides pass if budget allows (differentiate, write-a-term, own-the-loop, benchmark still instruction-style).
3. P5c: BetterHuman docs site — mkdocs.yml + copy BR theme assets (guidebook.css, output fence, snippet test),
   pages from docs/*.md (architecture, studio, contracts, segmentation guide), same style; strict build; screenshots.
4. REPORT.md (phases, commits, gates, ledgers, what is left: ruff deferred rules N/D/ANN001/SLF001, L-1/L-4/L-6/L-8
   ledger items, figure redraws 2–8, finger-joint sphere size in studio, stray marker in studio screenshot, nav-tab
   label faintness to check in a real browser), final commit, drop the index line in
   `$(git rev-parse --git-common-dir)/autoresearch/STATE.md`, TG ping (env not set → skip).

## Compact message
Task 2026-10-07_16-44 (contracts/trim/viewer/studio/docs for BR+BH): phases P1–P4 committed (BR 31a37ed, BH 81ca789);
P5 docs for BR is uncommitted in tree with a background clip job; next = finish clips, gates, commit, docs review,
BH docs site, REPORT. Read this STATE.md and resume from "Next steps".
