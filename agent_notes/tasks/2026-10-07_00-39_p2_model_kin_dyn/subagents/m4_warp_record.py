"""M4: last record of the Warp FK lane before removal. Device from argv[2] (cpu or cuda), float32, 1 CPU thread.

Compares `forward_kinematics(use_warp=True)` with the torch `Data` lane and with `forward_kinematics_state`,
forward and forward+backward. To repeat on a GPU later: check out commit a0c3ab1 and run this script with `cuda`.
"""

import json
import sys
import time
import warnings

import torch
from robot_descriptions import g1_description, panda_description

import better_robot as br
from better_robot.kinematics import forward_kinematics_state


def bench(fn, q, backward: bool, sync) -> float:
    def call():
        if backward:
            x = q.detach().requires_grad_(True)
            torch.autograd.grad(fn(x).sum(), x)
        else:
            fn(q)
        sync()

    for _ in range(5):
        call()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1e3)
    return sorted(times)[len(times) // 2]


def main() -> None:
    device = sys.argv[2]
    torch.set_num_threads(1)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    out = {"device": device, "rows": []}
    models = {"panda": br.load(panda_description.URDF_PATH), "g1": br.load(g1_description.URDF_PATH, free_flyer=True)}
    for name, model in models.items():
        model = model.to(device)
        lanes = {
            "warp": lambda x, m=model: br.forward_kinematics(m, x, use_warp=True).joint_pose_world,
            "torch_data": lambda x, m=model: br.forward_kinematics(m, x).joint_pose_world,
            "torch_state": lambda x, m=model: forward_kinematics_state(m, x).joint_transform,
        }
        for batch in (1, 256, 4096):
            q = model.q_neutral.expand(batch, -1).contiguous()
            for lane, fn in lanes.items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    row = {"model": name, "batch": batch, "lane": lane,
                           "forward_ms": round(bench(fn, q, False, sync), 4),
                           "forward_backward_ms": round(bench(fn, q, True, sync), 4)}
                fallback = [str(w.message)[:120] for w in caught if "warp" in str(w.message).lower()]
                if fallback:
                    row["warning"] = fallback[0]
                out["rows"].append(row)
                print(row, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
