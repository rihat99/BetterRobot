"""Decision prototype: level-batched / pointer-jumping world-transform composition vs library FK."""

import json
import statistics
import sys
import time

import torch
PANDA_URDF = "/home/rikhat.akizhanov/.cache/robot_descriptions/example-robot-data/robots/panda_description/urdf/panda.urdf"

import better_robot as br
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics
from better_robot.kinematics._fk_matrix import _local_matrices, get_plan, matrix_forward_compiled, prepare_inputs
from better_robot.kinematics.forward import forward_kinematics_raw
from better_robot.lie import se3, so3

# ---------------------------------------------------------------- models
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def humanoid(chain_len: int, fingers: bool) -> br.Model:
    """SMPL-24 spherical tree on a free flyer; optional 5x3 spherical fingers per hand and revolute leaf chains."""
    builder = ModelBuilder(f"humanoid_{chain_len}_{fingers}")
    gen = torch.Generator().manual_seed(0)

    def org() -> torch.Tensor:
        return torch.cat([0.1 * torch.randn(3, generator=gen), torch.tensor([0.0, 0.0, 0.0, 1.0])])

    builder.add_body("b0", mass=1.0)
    builder.add_free_flyer_root("root", child="b0", origin=org())
    for i in range(1, 24):
        builder.add_body(f"b{i}", mass=1.0)
        builder.add_spherical(f"j{i}", parent=f"b{SMPL_PARENTS[i]}", child=f"b{i}", origin=org())
    if fingers:
        for wrist in (22, 23):
            for f in range(5):
                parent = f"b{wrist}"
                for k in range(3):
                    child = f"f{wrist}_{f}_{k}"
                    builder.add_body(child, mass=0.1)
                    builder.add_spherical(f"jf{wrist}_{f}_{k}", parent=parent, child=child, origin=org())
                    parent = child
    for i in range(24 if chain_len else 0):
        parent = f"b{i}"
        for k in range(chain_len):
            child = f"c{i}_{k}"
            axis = torch.nn.functional.normalize(torch.randn(3, generator=gen), dim=0)
            builder.add_body(child, mass=0.1)
            builder.add_revolute(f"jc{i}_{k}", parent=parent, child=child, axis=axis, origin=org())
            parent = child
    return build_model(builder.finalize())


# ---------------------------------------------------------------- prototype
class LevelFK:
    """Level-batched and pointer-jumping world composition reusing the library's batched local step."""

    def __init__(self, structure, device):
        self.structure = structure
        self.plan = get_plan(structure)
        parents = list(self.plan.parents)
        nj = len(parents)
        depth = [0] * nj
        for j in self.plan.topo_order:
            depth[j] = 0 if parents[j] < 0 else depth[parents[j]] + 1
        self.depth = max(depth)
        levels = [[j for j in range(nj) if depth[j] == d] for d in range(self.depth + 1)]
        perm = [j for lvl in levels for j in lvl]
        self.perm = torch.tensor(perm, device=device)
        self.inv_perm = torch.argsort(self.perm)
        self.level_slices, self.parent_pos = [], []
        start = 0
        for d, lvl in enumerate(levels):
            self.level_slices.append((start, start + len(lvl)))
            if d > 0:
                prev = {j: i for i, j in enumerate(levels[d - 1])}
                self.parent_pos.append(torch.tensor([prev[parents[j]] for j in lvl], device=device))
            start += len(lvl)
        # pointer jumping: sentinel index nj holds identity; roots point at it
        pointer = [p if p >= 0 else nj for p in parents] + [nj]
        self.jumps = []
        while any(p != nj for p in pointer[:nj]):
            self.jumps.append(torch.tensor(pointer, device=device))
            pointer = [pointer[p] for p in pointer]

    def local(self, q_full, placements):
        rotations, translations = _local_matrices(self.plan, q_full, placements)
        batch_shape, nj = tuple(placements.shape[:-2]), self.plan.njoints
        top = torch.cat([rotations, translations.unsqueeze(-1)], dim=-1)
        bottom = torch.cat([placements.new_zeros(*batch_shape, nj, 1, 3), placements.new_ones(*batch_shape, nj, 1, 1)], dim=-1)
        return se3.to_matrix(placements) @ torch.cat([top, bottom], dim=-2)

    def chain_level(self, local_mat):
        local_perm = local_mat.index_select(-3, self.perm)
        start, end = self.level_slices[0]
        out = [local_perm[..., start:end, :, :]]
        for (start, end), ppos in zip(self.level_slices[1:], self.parent_pos):
            out.append(out[-1].index_select(-3, ppos) @ local_perm[..., start:end, :, :])
        return torch.cat(out, dim=-3).index_select(-3, self.inv_perm)

    def chain_jump(self, local_mat):
        eye = torch.eye(4, dtype=local_mat.dtype, device=local_mat.device).expand(*local_mat.shape[:-3], 1, 4, 4)
        acc = torch.cat([local_mat, eye], dim=-3)
        for pointer in self.jumps:
            acc = acc.index_select(-3, pointer) @ acc
        return acc[..., :-1, :, :]

    def world_pose(self, world_mat):
        return torch.cat([world_mat[..., :3, 3], so3.from_matrix(world_mat[..., :3, :3])], dim=-1)

    def run(self, values, q, mode):
        q_full, placements = prepare_inputs(self.structure, values, q)
        local_mat = self.local(q_full, placements)
        world_mat = self.chain_level(local_mat) if mode == "level" else self.chain_jump(local_mat)
        return self.world_pose(world_mat)


# ---------------------------------------------------------------- timing
def bench(fn, q, backward, device, reps=30, warmup=5, cg=False):
    def call():
        if cg:
            torch.compiler.cudagraph_mark_step_begin()
        if backward:
            qq = q.detach().requires_grad_(True)
            return torch.autograd.grad(fn(qq).sum(), qq)[0]
        return fn(q)

    for _ in range(warmup):
        call()
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def count_kernels(fn, q, backward):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        if backward:
            qq = q.detach().requires_grad_(True)
            torch.autograd.grad(fn(qq).sum(), qq)
        else:
            fn(q)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def main(device: str, lanes: str) -> None:
    out = open(sys.argv[3], "a")
    models = {
        "panda": br.load(PANDA_URDF),
        "humanoid24": humanoid(0, False),
        "humanoid222": humanoid(7, True),
    }
    for name, model_cpu in models.items():
        # accuracy check on CPU, float32 and float64
        for dtype, tol in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
            model = model_cpu.to(dtype=dtype)
            proto = LevelFK(model.structure, "cpu")
            gen = torch.Generator().manual_seed(1)
            q = torch.stack([model.random_configuration(generator=gen) for _ in range(64)])
            q_dtype_in = str(q.dtype)
            q = q.to(dtype)
            q = torch.where(torch.isfinite(q), q, 0.5 * torch.randn(q.shape, generator=gen, dtype=dtype))  # unlimited revolutes sample inf
            ref = forward_kinematics_raw(model.structure, model.values, q).joint_pose_world
            for mode in ("level", "jump"):
                got = proto.run(model.values, q, mode)
                err = (got - ref).abs().max().item()
                rec = {"check": name, "dtype": str(dtype), "mode": mode, "max_abs_err": err, "pass": err < tol, "random_cfg_dtype": q_dtype_in}
                print(rec, flush=True)
                out.write(json.dumps(rec) + "\n")
        model = model_cpu.to(device=device)
        proto = LevelFK(model.structure, device)
        meta = {"model": name, "njoints": model.structure.njoints, "nq": model.structure.nq, "depth": proto.depth,
                "jump_steps": len(proto.jumps)}
        print(meta, flush=True)
        out.write(json.dumps(meta) + "\n")
        values = model.values
        fns = {}
        if "eager" in lanes:
            fns["lib_eager"] = lambda q: forward_kinematics_raw(model.structure, values, q).joint_pose_world
            fns["proto_level_eager"] = lambda q: proto.run(values, q, "level")
            fns["proto_jump_eager"] = lambda q: proto.run(values, q, "jump")
        if "compile" in lanes:
            fns["lib_compile"] = lambda q: matrix_forward_compiled(model.structure, values, q)[0]
            fns["proto_level_compile"] = torch.compile(lambda q: proto.run(values, q, "level"), dynamic=True)
            if device == "cuda":
                fns["proto_level_compile_ro"] = torch.compile(lambda q: proto.run(values, q, "level"), mode="reduce-overhead")
        if "warp" in lanes:
            fns["lib_warp"] = lambda q: forward_kinematics(model, q, use_warp=True).joint_pose_world
        for batch in (1, 256, 4096):
            q = (model.q_neutral + 0.0).expand(batch, -1).contiguous()
            for lane, fn in fns.items():
                for backward in (False, True):
                    t0 = time.perf_counter()
                    ms = bench(fn, q, backward, device, cg=lane.endswith("_ro"))
                    rec = {"model": name, "device": device, "batch": batch, "lane": lane,
                           "pass": "fwd+bwd" if backward else "fwd", "median_ms": round(ms, 4),
                           "wall_s": round(time.perf_counter() - t0, 1)}
                    if device == "cuda" and lane.endswith("_eager") and batch == 256:
                        rec["kernels"] = count_kernels(fn, q, backward)
                    print(rec, flush=True)
                    out.write(json.dumps(rec) + "\n")
    out.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
