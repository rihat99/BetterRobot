"""Assemble the plan pages: shared head + navigation around each body in src/."""

from pathlib import Path

SITE = Path(__file__).parent
PAGES = {
    "index": ("Overview", "One torch implementation, one small optim, one viewer",
              "Where BetterRobot and BetterHuman stand today, what is wrong, and the plan to fix it."),
    "optim": ("Optim", "Costs become plain functions; solvers become small",
              "The proposed optimization layer: five ideas to learn, the same terms under Adam or Levenberg-Marquardt."),
    "speed": ("Speed", "Speed comes from fewer operations, not from a second backend",
              "Why the library is slow today, why Warp does not fix it, and how benchmarking keeps it fast."),
    "viewer": ("Viewer", "One viewer: a timeline and layers of preloaded arrays",
               "Three viewers become one core in BetterRobot that serves robots, humans and BVR clips."),
    "human": ("BetterHuman", "BetterHuman keeps its core idea and loses its special cases",
              "Body families as data, per-frame inputs without re-baking, and only public BetterRobot API."),
}
CSS = (SITE / "src" / "style.css").read_text()

for slug, (nav_label, title, lede) in PAGES.items():
    nav = " ".join(
        f'<a href="{other}.html"{" aria-current=\"page\"" if other == slug else ""}>{label}</a>'
        for other, (label, _, _) in PAGES.items()
    )
    body = (SITE / "src" / f"{slug}.html").read_text()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{lede}">
<style>
{CSS}</style>
</head>
<body>
<nav class="pages" aria-label="Pages">{nav}</nav>
<header>
  <h1>{title}</h1>
  <p class="lede">{lede}</p>
</header>
<main>
{body}
</main>
<footer>BetterRobot dev 7e2008f, BetterHuman dev 793d716. Plan drafted 2026-10-06. Evidence files are in <code>agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/</code>.</footer>
</body>
</html>
"""
    (SITE / f"{slug}.html").write_text(html)
    print("wrote", slug)
