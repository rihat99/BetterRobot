# P3 — the "Where we stand, what we miss, roadmap" site

Read `context.md` first. Build a small multi-page HTML site that the owner reads on a phone or laptop, days later,
with no context: what BetterRobot and BetterHuman are today, what the six reviews found we miss, and the roadmap
of bricks we will add. The owner's words: "easy read, easy understanding, visuals where necessary"; "we don't
implement everything now, show what we miss, have a roadmap"; "we should be like LEGO bricks".

## Inputs (read all)
- Reviews: `subagents/p1_{optim,kindyn,collision,viewer,human,toolbox}.md` in this task folder.
- Sweep reports (for the "release state" page): `subagents/p2_{br_core,br_optim,br_io_viewer,bh,perf,studio}.md`
  — read whichever exist when you start; re-check before finishing and fold in late ones.
- The previous site for style and structure: `agent_notes/tasks/2026-10-06_20-13_architecture_plan/site/`
  (`build.py` assembles `src/*.html` + `style.css` into pages with a shared nav; `diagrams/` holds archify
  outputs). Reuse that mechanism and CSS (copy `build.py` and `style.css`; adapt). Follow the `explain-html`
  skill rules at `~/.claude/skills/explain-html/SKILL.md` (one column, claims as headings, figures where they
  explain, `<details>` for depth, light/dark).
- Current code for any fact you state (do not trust a review blindly: spot-check each code claim you repeat,
  at least by opening the file and line it cites; drop or soften claims you cannot confirm).

## Pages (in `agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/site/`)
1. `index.html` — Overview: one-screen map of both libraries as bricks today (inline SVG or an archify
   architecture diagram built with `~/.claude/skills/archify` — read its SKILL.md; put outputs under `site/diagrams/`),
   the release state (gates, what the sweep fixed, what is left), the top 10 gaps across all themes, and the
   roadmap at a glance: a table brick | theme | when (now/next/later/maybe) | why.
2. `optim.html`, `kindyn.html`, `collision.html`, `viewer.html`, `human.html`, `toolbox.html` — one per theme:
   current state (short), what references taught us (2–4 lessons with links), findings ranked, roadmap table,
   "prep changes recommended now" list, open questions for the owner. Each ≤ ~1 screen of main text plus
   `<details>`; a figure where it explains (e.g. the sink seam in optim, the time-base problem in viewer, the
   Rig/Body seam in human). Figures: inline SVG following the `figure` skill (`~/.claude/skills/figure/SKILL.md`).
3. `decisions.html` — every "open question for the owner" from all themes, grouped, each with the options and the
   reviewer's recommendation, so the owner can answer in one sitting.

## Rules
- Facts with file paths; numbers only from measurements in the reports (say which report). No invented numbers.
- Consistent names: BetterRobot/BetterHuman, brick, theme names as above, `now/next/later/maybe`.
- Render-check with `bash ~/.claude/skills/explain-html/scripts/render.sh <page>` for every page (desktop and
  narrow; light and dark); the Chromium env is in context.md (`CHROME=` / `BETTER_ROBOT_CHROME`). Open the PNGs
  and fix clipping. Keep the PNGs under the task `vis/site/`.
- Deliverable: the site, plus `subagents/p3_site.md` (what each page says in 3 lines, checks run, claims you
  dropped as unverified). Final message ≤ 20 lines + absolute path to `site/index.html`.
