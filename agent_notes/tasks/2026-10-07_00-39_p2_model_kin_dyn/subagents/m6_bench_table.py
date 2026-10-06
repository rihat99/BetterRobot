"""Before/after table for dynamics: benchmarks/results/0953f28_cpu.json vs p2_world_dynamics_cpu.json."""
import json, sys
old = {(r["algo"], r["model"], r["batch"], r["mode"]): r for r in json.load(open(sys.argv[1]))["rows"] if r["impl"] == "better_robot"}
new = {(r["algo"], r["model"], r["batch"], r["mode"]): r for r in json.load(open(sys.argv[2]))["rows"] if r["impl"] == "better_robot"}
print(f"{'algo':5} {'model':8} {'batch':>5} {'mode':16} {'before ms':>10} {'ops':>6} {'after ms':>10} {'ops':>5} {'speed-up':>8}")
for key in new:
    if key not in old or new[key].get("median_ms") is None or old[key].get("median_ms") is None:
        continue
    a, b = old[key], new[key]
    print(f"{key[0]:5} {key[1]:8} {key[2]:>5} {key[3]:16} {a['median_ms']:10.2f} {a['ops']:>6} {b['median_ms']:10.2f} {b['ops']:>5} {a['median_ms']/b['median_ms']:8.1f}")
