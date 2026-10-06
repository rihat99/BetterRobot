"""E2: does torch.compile work, and help, for library FK and the lean FK prototype? CPU, 1 thread, float32.

Also re-measures eager library FK vs lean FK independently of proto_lean_fk.py's own timing code.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
from robot_descriptions import g1_description, panda_description

import better_robot as br

sys.path.insert(0, str(Path(__file__).parent))
import proto_lean_fk as lean  # noqa: E402


def median_ms(call, repeats: int = 15, warmup: int = 3) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def runner(fn, q: torch.Tensor, backward: bool):
    if backward:
        def call():
            x = q.detach().requires_grad_(True)
            fn(x).sum().backward()
    else:
        def call():
            with torch.no_grad():
                fn(x := q)
    return call


def main() -> None:
    torch.set_num_threads(1)
    models = {
        "panda": br.load(panda_description.URDF_PATH),
        "g1": br.load(g1_description.URDF_PATH, free_flyer=True),
    }
    rows = []
    for name, model in models.items():
        plan = lean.LeanPlan(model)
        lanes = {
            "library_eager": lambda q: br.forward_kinematics(model, q).joint_pose_world,
            "library_use_compile": lambda q: br.forward_kinematics(model, q, use_compile=True).joint_pose_world,
            "lean_eager": lambda q: lean.lean_world(plan, q),
            "lean_torch_compile": torch.compile(lambda q: lean.lean_world(plan, q), dynamic=True),
        }
        for lane, fn in lanes.items():
            for batch in (1, 4096):
                q = lean.sample_q(model, 16)[torch.arange(batch) % 16].float()
                for backward in (False, True):
                    row = {"model": name, "lane": lane, "batch": batch, "mode": "fwd+bwd" if backward else "fwd"}
                    call = runner(fn, q, backward)
                    try:
                        start = time.perf_counter()
                        call()
                        row["first_call_s"] = round(time.perf_counter() - start, 2)
                        row["median_ms"] = round(median_ms(call), 4)
                    except Exception as exc:  # noqa: BLE001 - the failure is the result
                        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    rows.append(row)
                    print(row, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
