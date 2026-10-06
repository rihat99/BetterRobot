# STATE — BetterRobot / 2026-10-06_21-40_p0_measure_and_fix (updated 2026-10-06 21:45)

## Pointers
- Task status: PLAN.md written, nothing executed yet; REPORT.md not started.
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/PLAN.md
- Approved redesign + owner decisions (section 6): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md
- Previous task report: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/REPORT.md ; evidence notes: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/
- Repos: /home/rikhat.akizhanov/better/dev/BetterRobot and /home/rikhat.akizhanov/better/dev/BetterHuman (branch dev, commit only, never push/merge). BVR is read-only.
- Chrome for rendering HTML needs a wrapper: see memory note headless-chrome-libasound.

## Running jobs
- Background HTTP server for the plan pages: python3 -m http.server 8765 --bind 127.0.0.1 in /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/site (user views it through an SSH port forward). Binding to 0.0.0.0 was denied by the permission system; do not retry.

## Next steps
- Step 1 of PLAN.md: uv sync dev group + warp + demos extras in the dev venv; confirm tests/test_pinocchio run.
- Step 2: write failing tests for the three bugs (lie/tangents.py:187 first), then fix.
- Step 3: benchmark runner with operation counts; CPU baselines; GPU baselines only on a FREE GPU.
- Subagents: mars_* only, each has a ~50k token budget, so give narrow briefs; at most one mars_top.

## Assumptions pending user confirmation
- Name `KinematicState` (variable `state`) for the frozen FK result; owner rejected `kin`, has not seen this name yet.
- Telegram pings do not work (TG_BOT_TOKEN / TG_CHAT_ID unset).
