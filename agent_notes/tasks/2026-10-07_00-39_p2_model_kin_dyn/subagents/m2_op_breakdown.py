"""M2: where do the torch operators of forward_kinematics_state go? Top-level aten calls per stage (Panda, G1)."""

import torch
from robot_descriptions import g1_description, panda_description
from torch.profiler import ProfilerActivity, profile

import better_robot as br
from better_robot.data_model.execution_batch import broadcast_to_execution_batch
from better_robot.data_model.reduced_coordinates import expand_configuration
from better_robot.kinematics._fk_lean import lean_core
from better_robot.kinematics.forward import _validate_q
from better_robot.kinematics.state import forward_kinematics_state, frame_jacobian
from better_robot.lie import se3


def count(fn) -> tuple[int, dict]:
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    events = [e for e in prof.events() if e.name.startswith("aten::")]
    top = [e for e in events if e.cpu_parent is None or not e.cpu_parent.name.startswith("aten::")]
    names: dict[str, int] = {}
    for e in top:
        names[e.name[6:]] = names.get(e.name[6:], 0) + 1
    return len(top), dict(sorted(names.items(), key=lambda kv: -kv[1])[:8])


for name, model in (("panda", br.load(panda_description.URDF_PATH)), ("g1", br.load(g1_description.URDF_PATH, free_flyer=True))):
    s, v = model.structure, model.values
    q = model.q_neutral.expand(256, -1).contiguous()
    q_full = expand_configuration(s, q)
    state = forward_kinematics_state(model, q)
    stages = {
        "validate": lambda: _validate_q(s, v, q),
        "batch_shape+broadcast": lambda: broadcast_to_execution_batch(q, v._execution_batch_shape(q), (s.nq,), name="q"),
        "expand_configuration": lambda: expand_configuration(s, q),
        "placements to_matrix": lambda: se3.to_matrix(v.joint_placements),
        "frames to_matrix": lambda: se3.to_matrix(v.frame_placements),
        "lean_core": lambda: lean_core(s, q_full, se3.to_matrix(v.joint_placements)[0:]),
        "total fk_state": lambda: forward_kinematics_state(model, q),
        "frame_jacobian on a state": lambda: frame_jacobian(model, forward_kinematics_state(model, q), model.nframes - 1),
    }
    for stage, fn in stages.items():
        total, names = count(fn)
        print(f"{name:6s} {stage:30s} {total:4d}  {names}")
