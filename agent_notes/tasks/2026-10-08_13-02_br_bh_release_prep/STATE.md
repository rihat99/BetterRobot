# STATE — BR+BH / 2026-10-08_13-02_br_bh_release_prep (updated 2026-10-08 15:30 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/PLAN.md
- Report: REPORT.md (same folder). Site: site/index.html. Reviews subagents/p1_*.md, sweeps subagents/p2_*.md,
  site notes subagents/p3_site.md, adversarial check subagents/p3c_aux_findings.md.
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman`, both `dev`.
- Rules: commit on dev only, never push/merge; trash not rm; 127.0.0.1 only; mars_* only; no new deps; never weaken tests.

## Status: DONE (2026-10-08 15:30 Dubai)
Code committed: BR 05c46d2, BH 2b8da78. Gates: BR 1589 passed / 0 failed, ruff 82 (baseline), strict docs clean;
BH 334 passed / 1 known failure. Nothing pushed or merged. No servers or agents running.
Owner next: read site/index.html and the Decisions page; decide the "reported, not changed" items in REPORT.md §4b
(MJCF parser, to_euler gimbal lock, test_reduction tolerance, prep changes).
