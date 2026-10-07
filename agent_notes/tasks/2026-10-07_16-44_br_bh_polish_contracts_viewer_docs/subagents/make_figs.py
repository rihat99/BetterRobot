"""Write the reworked/new BetterRobot doc figures (hand-laid SVG, grid of 8)."""
import sys
from html import escape

OUT = sys.argv[1]
INK, SUB, FILL, NAVY = "#1f2328", "#424a53", "#f3f5f8", "#1f3a68"
FONT = "Helvetica, Arial, sans-serif"


def head(w, h):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">',
        "<defs>",
        f'<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>',
        f'<marker id="an" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/></marker>',
        "</defs>",
        "<!-- background: white card so the figure reads on light and dark site themes -->",
        f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="12" fill="#ffffff" stroke="#d0d7de" stroke-width="1"/>',
    ]


def text(x, y, s, size=12, color=INK, anchor="middle", weight="normal"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>'


def box(x, y, w, h, lines, accent=False, dash=False):
    stroke = NAVY if accent else INK
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{FILL}" stroke="{stroke}" stroke-width="{2 if accent else 1.5}"' + (' stroke-dasharray="5 4"' if dash else "") + "/>"]
    n = len(lines)
    top = y + h / 2 - (n - 1) * 8 + 5
    for i, s in enumerate(lines):
        first = i == 0
        out.append(text(x + w / 2, top + i * 16 + (1 if not first else 0), s, 15 if first else 12, (NAVY if accent else INK) if first else SUB, weight="bold" if first else "normal"))
    return out


def arrow(points, accent=False):
    d = " ".join(f"{a},{b}" for a, b in points)
    color, mk = (NAVY, "an") if accent else (INK, "ag")
    return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{2 if accent else 1.5}" marker-end="url(#{mk})"/>'


def save(name, parts):
    with open(f"{OUT}/{name}", "w") as f:
        f.write("\n".join(parts + ["</svg>", ""]))


# 1. architecture: rows follow LAYER_RANK in tests/contract/test_layer_dependencies.py
p = head(640, 456)
rows = [("viewer", "rank 7 · nothing imports it"), ("terms", "rank 6 · robot residuals, solve_ik"), None,
        ("data_model", "rank 4 · Model, KinematicState"), ("spatial", "rank 3 · Motion, Force, Inertia"), ("lie", "rank 2 · SE3, SO3 formulas")]
for i, r in enumerate(rows):
    y = 24 + i * 72
    if r:
        p += box(48, y, 360, 48, list(r))
    else:  # rank 5: four packages of equal rank in one row
        p.append(f'<rect x="48" y="{y}" width="360" height="48" rx="6" fill="{FILL}" stroke="{INK}" stroke-width="1.5"/>')
        for k, name in enumerate(["io", "kinematics", "dynamics", "collision"]):
            p.append(text(93 + 90 * k, y + 29, name, 13, INK, weight="bold"))
            if k:
                p.append(f'<line x1="{48 + 90 * k}" y1="{y + 10}" x2="{48 + 90 * k}" y2="{y + 38}" stroke="{SUB}" stroke-width="1"/>')
    if i < 5:
        p.append(arrow([(228, y + 48), (228, y + 72)]))
p += box(448, 312, 176, 48, ["optim", "rank 3 · imports only torch"], accent=True)
p.append(arrow([(408, 120), (536, 120), (536, 312)], accent=True))
p.append(text(526, 220, "Problem, TermFn,", 12, NAVY, "end"))
p.append(text(526, 236, "solvers", 12, NAVY, "end"))
save("architecture.svg", p)

# 9. pipeline overview (landing page)
p = head(1060, 152)
steps = [(["Robot file", "URDF · MJCF", "or builder"], "io"),
         (["Model", "structure (static)", "values (tensors)"], "data_model"),
         (["State", "model.fk(q)", "Jacobians · dynamics"], "kinematics · dynamics"),
         (["Terms", "residual functions", "pose · limits · rest"], "terms"),
         (["Solve", "Problem → LM, GN,", "L-BFGS, torch.optim"], "optim"),
         (["Viewer", "viser layers", "record → webm / png"], "viewer")]
for i, (lines, pkg) in enumerate(steps):
    x = 26 + i * 172
    p += box(x, 24, 148, 80, lines)
    p.append(text(x + 74, 126, pkg, 12, SUB))
    if i < 5:
        p.append(arrow([(x + 148, 64), (x + 172, 64)], accent=True))
save("pipeline_overview.svg", p)

# 10. IK loop: one LevenbergMarquardt iteration inside solve_ik
p = head(640, 312)
cells = [(24, 24, ["residual r(q)", "pose error per target", "(+ optional limit, rest)"]),
         (232, 24, ["Jacobian J", "dr/dδ from frame", "Jacobians"]),
         (440, 24, ["normal equations", "JᵀJ and Jᵀr"]),
         (440, 216, ["damped step", "(JᵀJ + λD) δ = −Jᵀr", "δ kept inside q limits"]),
         (232, 216, ["retract", "q ← Model.integrate(q, δ)", "(RobotManifold)"]),
         (24, 216, ["accept or reject", "gain ratio updates λ", "(Nielsen rule)"])]
for x, y, lines in cells:
    p += box(x, y, 176, 72, lines)
p += [arrow([(200, 60), (232, 60)], True), arrow([(408, 60), (440, 60)], True),
      arrow([(528, 96), (528, 216)], True), arrow([(440, 252), (408, 252)], True),
      arrow([(232, 252), (200, 252)], True), arrow([(112, 216), (112, 96)], True)]
p.append(text(320, 150, "solve_ik(model, targets, q0=...)", 13, INK, weight="bold"))
p.append(text(320, 170, "LevenbergMarquardt, up to iters=100", 12, SUB))
save("ik_loop.svg", p)

# 11. contracts map: families and enforcer kinds from docs/contracts/00-index.md
fams = [("C-NAME", "naming", "ruff · test · review"), ("C-TENS", "tensors", "test · review"),
        ("C-STATE", "state, mutability", "test · review"), ("C-DIFF", "differentiability", "test · review"),
        ("C-API", "API shape", "ruff · test · review"), ("C-ERR", "validation, errors", "ruff · test · review"),
        ("C-DEP", "dependencies", "ruff · test · review"), ("C-EXT", "extension points", "ruff · test · review"),
        ("C-PERF", "performance", "review only"), ("C-DOC", "docs", "ruff · test · review"),
        ("C-TEST", "tests", "ruff · review")]
p = head(720, 296)
for i, (code, name, enf) in enumerate(fams):
    x, y = 32 + (i % 4) * 168, 24 + (i // 4) * 88
    p += box(x, y, 152, 72, [code, name, enf], accent=(code == "C-PERF"))
lx = 32 + 3 * 168
for k, s in enumerate(["ruff: lint rule in CI", "test: pytest test", "review: human review"]):
    p.append(text(lx, 228 + k * 18, s, 12, SUB, "start"))
save("contracts_map.svg", p)

# 12. time structure: knots x variables, a window, a time-free variable
p = head(680, 320)
p.append(text(168, 36, "knot t →", 12, SUB, "start"))
for t in range(6):
    p.append(f'<rect x="{168 + 64 * t}" y="56" width="56" height="40" rx="4" fill="{FILL}" stroke="{INK}" stroke-width="1.5"/>')
    p.append(text(196 + 64 * t, 81, f"q[{t}]", 13))
p.append(text(152, 74, "q  (B, T, nq)", 13, INK, "end", "bold"))
p.append(text(152, 90, 'time=("q",)', 12, NAVY, "end"))
p.append(f'<rect x="290" y="48" width="132" height="56" rx="6" fill="none" stroke="{NAVY}" stroke-width="2"/>')
p.append(arrow([(356, 104), (356, 136)], True))
p += box(196, 136, 320, 48, ["window=2: row t reads q[t], q[t+1]", "e.g. a finite-difference velocity term"], accent=True)
p += box(168, 232, 376, 48, ["time-free variable (B, k), e.g. body shape", "read by every knot → arrow column in JᵀJ"], dash=True)
p.append(text(152, 260, "no time axis", 12, SUB, "end"))
save("time_structure.svg", p)
