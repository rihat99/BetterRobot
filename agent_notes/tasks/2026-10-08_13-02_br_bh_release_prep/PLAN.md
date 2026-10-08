# PLAN — BR + BH release preparation (2026-10-08 13:02 Dubai)

Owner's request (verbatim intent): finishing touches before `dev` merges to main and dependent repos update.
1. Adversarial review of the architecture and the main choices that affect users and computation; reviewers look at
   how reference projects evolved and what they struggled with. Find what we miss and what we may add later
   (optimization beyond least squares, native sparsity, advanced collision, more 3D tools like in BVR, general
   robotics-theory tools). Not implemented now: the output is a gap list + roadmap, ensuring modularity ("LEGO bricks").
   Delivered as an HTML page (a few pages allowed), like `2026-10-06_20-13_architecture_plan/site/`. Both BR and BH.
2. Docs: videos/GIFs show as white space → fix.
3. HumanStudio: frames files from main tracked in git (checked: already tracked, identical); gizmo usable inside the
   body (owner chose: draw on top via `depth_test=False`, scale 0.15 → 0.20); `Reset pose`, `Deselect joint`,
   `Deselect frame` + a status line move above the tabs.
4. Final code sweep of every file in both repos: minor bugs, computational correctness, bottlenecks. Policy (owner):
   fix small local bugs with a pinning test; report anything touching API shape or numerics.
5. Delegate heavily.

## Intake answers (2026-10-08 13:10)
Gizmo: on top + slightly bigger. Toolbar: three buttons + status line. Bugs: fix small, report the rest.
Review depth: 6 reviewers across themes (optim/sparsity, kin-dyn speed, collision, viewer/3D tools incl. BVR,
BH architecture, general robotics toolbox).

## Phases
- P0 (orchestrator): docs video paths (`../clips/` → `../../clips/` in raw `<video>` HTML; mkdocs does not rewrite
  raw HTML) → verify: every `<video>`/`poster` URL resolves in `site/`.
- P1 (6 × mars_top_opus, parallel): architecture reviews → `subagents/p1_<theme>.md` each.
- P2 (parallel with P1 as agent budget allows; 6 agents max): studio changes (1), code sweep BR core (1), BR
  optim/terms/collision (1), BR io/viewer (1), BH (1), perf/bench check (1) → `subagents/p2_*.md` + fixes.
- P3: HTML report site (`site/`) from P1 + P2 findings (1 mars_top_opus), then one mars_aux adversarial pass on the
  site's claims against the code → fixes.
- P4: gates (tests, ruff, mkdocs strict, both repos), REPORT.md, commit `dev` in both repos, drop index line.

## Metrics / success
- Docs: all 7 video embeds play (URLs resolve, checked in the built site + screenshot).
- Studio: gizmo visible inside body on a real server screenshot; three buttons + status line above tabs; frames files
  tracked (`git ls-files`).
- Sweep: all tests pass in both repos (BH known failure unchanged), ruff not worse, bugs fixed listed with tests.
- Review: site at `agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/site/index.html`, every claim about current
  code checked by the adversarial pass.
