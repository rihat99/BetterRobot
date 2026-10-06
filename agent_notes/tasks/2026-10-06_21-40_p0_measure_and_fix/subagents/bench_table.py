"""Print a markdown table (median ms / ops) from a benchmarks/run.py JSON; optionally compare ops with a second JSON."""

from __future__ import annotations

import json
import sys


def cell(row: dict | None) -> str:
    if row is None:
        return "-"
    if "error" in row:
        return "skipped" if row["error"].startswith("skipped") else "ERROR"
    return f"{row['median_ms']:.3g} / {row['ops']}"


def main() -> None:
    first = json.load(open(sys.argv[1]))
    rows = {(r["impl"], r["algo"], r["model"], r["batch"], r["mode"]): r for r in first["rows"] if "batch" in r}
    print("meta:", json.dumps(first["meta"]))
    print("\n| algo | model | fwd B=1 | fwd+bwd B=1 | fwd B=4096 | fwd+bwd B=4096 | pinocchio fwd B=1 (ms) |")
    print("|---|---|---|---|---|---|---|")
    for algo in ("fk", "jacobian", "rnea", "crba", "aba"):
        for model in ("panda", "g1", "tree25", "tree200"):
            cells = [cell(rows.get(("better_robot", algo, model, b, m)))
                     for b in (1, 4096) for m in ("forward", "forward_backward")]
            pin = rows.get(("pinocchio", algo, model, 1, "forward"))
            pin_cell = f"{pin['median_ms']:.2g}" if pin and "median_ms" in pin else "-"
            print(f"| {algo} | {model} | " + " | ".join(cells) + f" | {pin_cell} |")
    if len(sys.argv) > 2:
        second = {(r["impl"], r["algo"], r["model"], r["batch"], r["mode"]): r
                  for r in json.load(open(sys.argv[2]))["rows"] if "batch" in r}
        both = [k for k in rows if k in second and rows[k].get("ops") is not None and second[k].get("ops") is not None]
        different = [(k, rows[k]["ops"], second[k]["ops"]) for k in both if rows[k]["ops"] != second[k]["ops"]]
        print(f"\nops compared on {len(both)} rows; different: {len(different)}")
        for item in different[:20]:
            print(item)


if __name__ == "__main__":
    main()
