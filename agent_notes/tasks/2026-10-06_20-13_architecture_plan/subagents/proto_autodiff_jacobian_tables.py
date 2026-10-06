"""Turn RESULTS_JSON lines of the benchmark logs into markdown tables (ms median; GPU peak MiB; x = / analytic)."""
import json
import sys

for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        if line.startswith("  {"):
            rows.append(json.loads(line))
        if line.startswith("RESULTS_JSON "):
            rows = [r for r in json.loads(line[len("RESULTS_JSON "):]) if "B" in r]
    if not rows:
        print(f"{path}: no RESULTS_JSON (incomplete run)")
        continue
    print(f"\n### {path.split('/')[-1]}")
    for model in dict.fromkeys(r["model"] for r in rows):
        mr = [r for r in rows if r["model"] == model]
        keys = [k for k in mr[0] if k not in ("model", "fk", "B")]
        print(f"\n{model} (fk={mr[0]['fk']})\n")
        print("| variant | " + " | ".join(f"B={r['B']}" for r in mr) + " |")
        print("|---|" + "---|" * len(mr))
        for k in keys:
            cells = []
            for r in mr:
                v = r.get(k)
                if isinstance(v, list):
                    ms, peak = v
                    ana = r["analytic"][0] if not k.startswith("gn_") else r["gn_analytic"][0]
                    ratio = ms / ana
                    cells.append(f"{ms:.2f} ms ({ratio:.1f}x)" + (f", {peak:.0f} MiB" if peak is not None else ""))
                else:
                    cells.append(str(v)[:60] if v is not None else "-")
            print(f"| {k} | " + " | ".join(cells) + " |")
