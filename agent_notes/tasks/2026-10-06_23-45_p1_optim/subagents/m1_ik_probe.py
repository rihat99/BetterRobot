"""Probe: why does new LM reach fewer targets? Variants on the stored problems (float64)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim2, terms

sys.path.insert(0, str(Path(__file__).parent))
from m1_ik_check import FRAME, GOLDEN, errors  # noqa: E402

golden = np.load(GOLDEN)
model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
target = torch.as_tensor(golden["target"])
q0 = model.q_neutral.expand(256, -1).clone()
q0 = torch.minimum(torch.maximum(q0, model.lower_pos_limit), model.upper_pos_limit)


def report(label, q):
    position, rotation = errors(model, q, target)
    ok = (position < 1e-4) & (rotation < 1e-3)
    print(f"{label:40s} reached {float(ok[:128].float().mean()):.3f} / {float(ok[128:].float().mean()):.3f}", flush=True)
    return ok


variant = sys.argv[1] if len(sys.argv) > 1 else "all"
problem = terms.ik_problem(model, {FRAME: target}, q0)
ok_bounds = report("lm bounds (default)", optim2.solve(problem, optim2.LevenbergMarquardt(), iters=100).x["q"])
free = terms.ik_problem(model, {FRAME: target}, q0)
free.bounds = {}
q_free = optim2.solve(free, optim2.LevenbergMarquardt(), iters=100).x["q"]
ok_free = report("lm no bounds", q_free)
inside = ((q_free >= model.lower_pos_limit) & (q_free <= model.upper_pos_limit)).all(-1)
print("no-bounds solutions inside limits:", float(inside[:128].float().mean()), float(inside[128:].float().mean()))
old_ok = torch.as_tensor(golden["position_error_no_regularizer"]) < 1e-4
print("old reached:", float(old_ok[:128].float().mean()), float(old_ok[128:].float().mean()))
q_old = torch.as_tensor(golden["q_no_regularizer"])
inside_old = ((q_old >= model.lower_pos_limit - 1e-9) & (q_old <= model.upper_pos_limit + 1e-9)).all(-1)
print("old solutions inside limits:", float(inside_old.float().mean()))
soft = terms.ik_problem(model, {FRAME: target}, q0, limit_weight=100.0)
soft.bounds = {}
report("lm soft limits w=100", optim2.solve(soft, optim2.LevenbergMarquardt(), iters=100).x["q"])
