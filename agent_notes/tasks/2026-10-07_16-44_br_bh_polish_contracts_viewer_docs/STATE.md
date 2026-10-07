# STATE — BR+BH / 2026-10-07_16-44_br_bh_polish_contracts_viewer_docs (updated 2026-10-07 16:50 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/PLAN.md
- Subagent context: `subagents/context.md`; discovery outputs `subagents/p0_*.md`.
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman` (both `dev`).
- Rules: commit per phase on dev only, never push/merge; trash not rm; 127.0.0.1 only; mars_* only (≤6, one mars_top);
  no new deps (Playwright dev extra approved); never weaken tests. Owner present (working hours).

## Status: P0 done (p0_*.md + p0_synthesis.md). P1 / P2a / P3a running in parallel (17:45 Dubai)
Owner decisions (17:35): test trim −22 % as in p0_tests.md; viewer `frame`→`step`; monograms in math code only;
studio UI shell rebuilt on BR viewer (domain core kept). Mine: pin mkdocs<2; Material theme.font; record on GPU
only if gpu_free FREE; ffmpeg optional; chrome wrapper as dev helper.

## Done so far (uncommitted, in tree)
- P1: docs/contracts (11 pages, 42 rules, 5 ADRs), tests/contract/{test_keyword_only,test_frozen,test_contract_matrix}.py,
  conventions.md + design-decisions.md folded in (trashed), ruff rules enabled/deferred, ledger p1_violations.md (L-1..L-8).
  Adversarial review p1_review.md: items 1–13 applied; 14–16 open (p1_state.md "P1c").
- P2a: 76 drops applied by owner via `subagents/drop_tests.py --apply` (19:05); parity docstrings fixed; `slow` marker on examples.
- P3a: gizmo latest-wins worker (5.8 s→0.3 s), viser default lights, navy LIGHT theme (`Theme.brand_color`), rename
  `frame`→`step`/`nsteps`, lazy Timeline, one Layers panel + `Viewer.rebuild` + `Layer.listed`. Items 7–8 (joint
  shapes, RobotView chain) open. Stale docs: concepts/viewer.md, getting-started/visualize.md (2 snippet failures).
  BH call sites to update listed in p3a_state.md (num_frames/set_frame in viz/).

## Running
- mars_top_opus P1d: tests/contract/{test_dtype_device,test_no_inplace}.py + BH docs/contracts addendum + BH contract tests.
- mars_top_opus P2a-2: test merges + orphan cleanup + BH merges → p2_tests_ledger.md.
- mkdocs serve of BR docs on http://127.0.0.1:8765 (task bh6kicmzx) — informational only.

## Next steps
1. P1e (short): review items 14–16. Then run all gates, show owner the diff summary, commit P1+P2a+P3a on dev (BR, BH).
2. P3a-3: joint shapes by kind + RobotView chain; P3b: drop rasterizer, headless `record` (chrome wrapper), 04_record.
3. P2b polish through p1_violations.md (+ API sugar list for owner). P4 studio shell. P5 docs BR then BH.
