"""Print markdown tables from status_cpu_*.json and baseline_cpu_eager.json."""

import json
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])
for path in sorted(artifacts.glob("status_cpu_*.json")):
    payload = json.loads(path.read_text())
    print(f"\n### {path.name} env={payload['env']}\n")
    print("| lib | model | algo | B | median ms | evals/s | n | extra |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        if "error" in row and "median_ms" not in row:
            print(f"| {row.get('lib')} | {row.get('model')} | {row.get('algo')} | {row.get('batch')} | ERROR | | | {row['error'][:120]} |")
            continue
        rate = row.get("evals_per_s", row.get("solves_per_s", 0.0))
        extra = ""
        if "iters_median" in row:
            extra = f"iters med {row['iters_median']:.0f} max {row['iters_max']:.0f}, conv {row['converged_frac']:.3f}"
        print(f"| {row['lib']} | {row['model']} | {row['algo']} | {row['batch']} | {row['median_ms']:.4g} | {rate:.3g} | {row.get('n', '')} | {extra} |")

baseline = artifacts / "baseline_cpu_eager.json"
if baseline.exists():
    payload = json.loads(baseline.read_text())
    rows = payload.get("results", payload.get("rows", payload)) if isinstance(payload, dict) else payload
    print(f"\n### baseline.py keys={list(payload)[:12] if isinstance(payload, dict) else 'list'}\n")
    for row in rows if isinstance(rows, list) else []:
        keep = {k: row[k] for k in row if k in ("case_id", "case", "status", "median_ms", "warm_median_ms", "p50_ms", "error", "reason", "throughput_per_s")}
        if not keep:
            keep = {k: row[k] for k in list(row)[:8]}
        print(json.dumps(keep)[:300])
