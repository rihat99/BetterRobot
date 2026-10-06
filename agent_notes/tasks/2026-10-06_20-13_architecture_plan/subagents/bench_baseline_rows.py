"""Print status + warm median per case from baseline_cpu_eager.json (structure-agnostic scan)."""

import json
import sys

payload = json.load(open(sys.argv[1]))


def walk(node):
    if isinstance(node, dict):
        if "case_id" in node:
            yield node
        else:
            for value in node.values():
                yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


for row in walk(payload):
    numbers = {k: v for k, v in row.items() if isinstance(v, (int, float)) and "median" in k}
    nested = {k: v for k, v in row.items() if isinstance(v, dict)}
    for key, sub in nested.items():
        for sub_key, value in sub.items():
            if isinstance(value, (int, float)) and "median" in sub_key:
                numbers[f"{key}.{sub_key}"] = value
    status = row.get("status", "?")
    reason = str(row.get("reason", row.get("error", "")))[:80]
    print(row["case_id"], status, json.dumps({k: round(v, 4) for k, v in list(numbers.items())[:3]}), reason)
