"""E3 prototype: lean batched FK (rotation+translation, one affine feature map, level composition) vs library FK."""

import importlib.util
import statistics
import time

import torch

import better_robot as br
from better_robot.data_model.model_structure import JOINT_KIND_CODES
from better_robot.kinematics._fk_matrix import get_plan
from better_robot.lie import so3

_spec = importlib.util.spec_from_file_location(
    "plf", "agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/proto_level_fk.py")
plf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plf)

K = JOINT_KIND_CODES
REV = {K["revolute_rx"], K["revolute_ry"], K["revolute_rz"], K["revolute_unaligned"]}
PRI = {K["prismatic_px"], K["prismatic_py"], K["prismatic_pz"], K["prismatic_unaligned"]}
FIX = {K["universe"], K["fixed"]}
QUAT = {K["spherical"]: 0, K["free_flyer"]: 3}  # code -> offset of [qx qy qz qw] inside the joint's q block

# vec(R(q)) = vec(q q^T) @ C / |q|^2 for q = [x y z w]; row index a*4+b of q_a q_b.
_C_TERMS = {0: [(0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, 1)], 1: [(0, 1, 2), (2, 3, -2)], 2: [(0, 2, 2), (1, 3, 2)],
            3: [(0, 1, 2), (2, 3, 2)], 4: [(0, 0, -1), (1, 1, 1), (2, 2, -1), (3, 3, 1)], 5: [(1, 2, 2), (0, 3, -2)],
            6: [(0, 2, 2), (1, 3, -2)], 7: [(1, 2, 2), (0, 3, 2)], 8: [(0, 0, -1), (1, 1, -1), (2, 2, 1), (3, 3, 1)]}


def hat(v: torch.Tensor) -> torch.Tensor:
    """Skew matrix of v: (3,) -> (3, 3)."""
    x, y, z = v.tolist()
    return v.new_tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class LeanPlan:
    """Precomputed constants: q gather, affine feature->local-4x4 map, tree levels."""

    def __init__(self, model: br.Model) -> None:
        st, dtype = model.structure, model.values.joint_placements.dtype
        fk_plan = get_plan(st)
        parents, nj = list(fk_plan.parents), st.njoints
        depth = [0] * nj
        for j in fk_plan.topo_order:
            depth[j] = 0 if parents[j] < 0 else depth[parents[j]] + 1
        self.levels = [[j for j in range(nj) if depth[j] == d] for d in range(max(depth) + 1)]
        order = [j for lvl in self.levels for j in lvl]
        slot = {j: i for i, j in enumerate(order)}  # level-order slot of each joint
        pl = model.values.joint_placements.detach()
        R0, p0 = so3.to_matrix(pl[:, 3:7]), pl[:, :3]  # unnormalised, as the library's 4x4 composition uses it
        axes = st.joint_axes.to(dtype)
        C = torch.zeros(16, 9, dtype=dtype)
        for r, terms in _C_TERMS.items():
            for a, b, k in terms:
                C[a * 4 + b, r] += k
        rev, quat, lin = [], [], []  # (joint, q column) / (joint, first quat column) / (joint, q column, 3-vector)
        for j in range(nj):
            code, i0 = st.joint_kind_codes[j], st.idx_qs_full[j]
            if code in REV:
                rev.append((j, i0))
            elif code in PRI:
                lin.append((j, i0, R0[j] @ axes[j]))
            elif code in QUAT:
                quat.append((j, i0 + QUAT[code]))
                if QUAT[code]:
                    lin += [(j, i0 + b, R0[j][:, b]) for b in range(3)]
            elif code not in FIX:
                raise NotImplementedError(f"joint {j} kind code {code}")
        self.cols = torch.tensor([c for _, c in rev] + [c + k for _, c in quat for k in range(4)] + [c for _, c, _ in lin])
        self.nrev, self.nquat = len(rev), len(quat)
        # mimic: q_full = q @ E^T + o, folded into the gather as one addmm over the needed columns
        self.mimic = (st.q_expansion.to(dtype)[self.cols].mT.contiguous(), st.q_offset.to(dtype)[self.cols]) \
            if st.has_mimic else None
        nfeat = 2 * len(rev) + 16 * len(quat) + len(lin)
        W = torch.zeros(nfeat, nj, 4, 4, dtype=dtype)
        bias = torch.zeros(nj, 4, 4, dtype=dtype)
        bias[:, 3, 3] = 1.0
        bias[:, :3, 3] = p0
        bias[:, :3, :3] = R0
        for f, (j, _) in enumerate(rev):  # R0 R = (R0 + R0K^2) + s R0K - c R0K^2
            Kj = hat(axes[j])
            bias[j, :3, :3] += R0[j] @ Kj @ Kj
            W[f, j, :3, :3] = R0[j] @ Kj
            W[len(rev) + f, j, :3, :3] = -R0[j] @ Kj @ Kj
        base = 2 * len(rev)
        for f, (j, _) in enumerate(quat):  # R0 R(q): column block of vec(qq^T)/|q|^2
            bias[j, :3, :3] = 0.0
            Cj = C.view(16, 3, 3)
            W[base + 16 * f: base + 16 * (f + 1), j, :3, :3] = torch.einsum("im,fmk->fik", R0[j], Cj)
        base += 16 * len(quat)
        for f, (j, _, vec) in enumerate(lin):
            W[base + f, j, :3, 3] = vec
        perm = torch.tensor(order)
        self.W = W.index_select(1, perm).reshape(nfeat, nj * 16)
        self.bias = bias.index_select(0, perm).reshape(nj * 16)
        self.nj, self.inv_perm = nj, torch.argsort(perm)
        self.slices, self.parent_pos = [], []
        start = 0
        for d, lvl in enumerate(self.levels):
            self.slices.append((start, start + len(lvl)))
            if d > 0:
                prev = {j: i for i, j in enumerate(self.levels[d - 1])}
                ppos = [prev[parents[j]] for j in lvl]
                # broadcast instead of gather when the previous level has a single joint
                self.parent_pos.append(None if len(prev) == 1 else torch.tensor(ppos))
            start += len(lvl)
        self.identity_out = order == list(range(nj))


def lean_fk(plan: LeanPlan, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World rotations (B, nj, 3, 3) and translations (B, nj, 3) for q (B, nq); both are views of one tensor."""
    world = lean_world(plan, q)
    return world[..., :3], world[..., 3]


def lean_world(plan: LeanPlan, q: torch.Tensor) -> torch.Tensor:
    """World [R | p] (B, nj, 3, 4) for q (B, nq)."""
    qg = q.index_select(-1, plan.cols) if plan.mimic is None else torch.addmm(plan.mimic[1], q, plan.mimic[0])
    nr, nq4 = plan.nrev, plan.nquat
    feats = []
    if nr:
        qr = qg[..., :nr]
        feats += [torch.sin(qr), torch.cos(qr)]
    if nq4:
        qv = qg[..., nr: nr + 4 * nq4].view(*q.shape[:-1], nq4, 4)
        outer = qv.unsqueeze(-1) * qv.unsqueeze(-2)
        outer = outer / outer.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        feats.append(outer.flatten(-3))
    feats.append(qg[..., nr + 4 * nq4:])
    local = torch.addmm(plan.bias, torch.cat(feats, dim=-1), plan.W).view(-1, plan.nj, 4, 4)
    s0, e0 = plan.slices[0]
    out = [local[:, s0:e0, :3, :]]
    for (s, e), ppos in zip(plan.slices[1:], plan.parent_pos):
        parent = out[-1] if ppos is None else out[-1].index_select(1, ppos)
        out.append(parent @ local[:, s:e])
    world = torch.cat(out, dim=1)
    if not plan.identity_out:
        world = world.index_select(1, plan.inv_perm)
    return world


def lean_pose(plan: LeanPlan, q: torch.Tensor) -> torch.Tensor:
    """Lean FK converted to the library (B, nj, 7) pose layout."""
    R, p = lean_fk(plan, q)
    return torch.cat([p, so3.from_matrix(R)], dim=-1)


def count_ops(fn, q: torch.Tensor, backward: bool) -> dict:
    """CPU profiler counts: top-level aten calls, and leaf aten calls that are not views/allocs (kernel proxy)."""
    from torch.profiler import ProfilerActivity, profile

    views = ("view", "as_strided", "expand", "unsqueeze", "squeeze", "select", "slice", "reshape", "permute",
             "transpose", "t", "alias", "diagonal", "empty", "empty_like", "empty_strided", "resolve", "detach",
             "unbind", "split", "narrow", "_unsafe_view", "lift_fresh", "result_type")
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        if backward:
            qq = q.detach().requires_grad_(True)
            torch.autograd.grad(fn(qq).sum(), qq)
        else:
            fn(q)
    ev = [e for e in prof.events() if e.name.startswith("aten::")]
    top = sum(1 for e in ev if e.cpu_parent is None or not e.cpu_parent.name.startswith("aten::"))
    leaf = sum(1 for e in ev if not any(c.name.startswith("aten::") for c in e.cpu_children)
               and e.name[6:].split(".")[0] not in views)
    return {"top": top, "leaf": leaf}


def bench(fn, q, backward, reps=25, warmup=5) -> float:
    """Median ms over reps after warm-up."""
    def call():
        if backward:
            qq = q.detach().requires_grad_(True)
            return torch.autograd.grad(fn(qq).sum(), qq)[0]
        return fn(q)

    for _ in range(warmup):
        call()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def sample_q(model: br.Model, n: int) -> torch.Tensor:
    """Random configurations with infinite-limit entries replaced by Gaussians."""
    gen = torch.Generator().manual_seed(1)
    q = torch.stack([model.random_configuration(generator=gen) for _ in range(n)]).to(model.values.joint_placements.dtype)
    return torch.where(torch.isfinite(q), q, 0.5 * torch.randn(q.shape, generator=gen, dtype=q.dtype))


def main() -> None:
    """Parity, op counts and CPU timing."""
    torch.set_num_threads(1)
    from robot_descriptions import g1_description, panda_description

    models = {"panda": br.load(panda_description.URDF_PATH), "g1": br.load(g1_description.URDF_PATH, free_flyer=True),
              "tree25": plf.humanoid(0, False)}
    for name, m32 in models.items():
        m64 = m32.to(dtype=torch.float64)
        plan = LeanPlan(m64)
        q = sample_q(m64, 64)
        ref = br.forward_kinematics(m64, q).joint_pose_world
        R, p = lean_fk(plan, q)
        e_p = (p - ref[..., :3]).abs().max().item()
        e_R = (R - so3.to_matrix(ref[..., 3:])).abs().max().item()
        q1 = q.clone().requires_grad_(True)
        g_ref = torch.autograd.grad(br.forward_kinematics(m64, q1).joint_pose_world[..., :3].sum(), q1)[0]
        q2 = q.clone().requires_grad_(True)
        g_lean = torch.autograd.grad(lean_fk(plan, q2)[1].sum(), q2)[0]
        e_g = (g_ref - g_lean).abs().max().item()
        e_pose = (lean_pose(plan, q) - ref).abs()
        e_pose = min(e_pose.max().item(), 99.0)  # sign gauge may differ; reported for information
        print(f"PARITY {name} nj={plan.nj} nq={m64.structure.nq} levels={len(plan.levels)} feats={plan.W.shape[0]} "
              f"pos={e_p:.2e} rot={e_R:.2e} grad={e_g:.2e} pose7={e_pose:.2e}", flush=True)
        plan32 = LeanPlan(m32)
        fns = {"lib": lambda x: br.forward_kinematics(m32, x).joint_pose_world,
               "lean": lambda x: lean_world(plan32, x),
               "lean+pose7": lambda x: lean_pose(plan32, x)}
        q256 = sample_q(m32, 256)
        for lane, fn in fns.items():
            cf, cb = count_ops(fn, q256, False), count_ops(fn, q256, True)
            print(f"OPS {name} {lane} fwd top={cf['top']} leaf={cf['leaf']} fwdbwd top={cb['top']} leaf={cb['leaf']}",
                  flush=True)
        for batch in (1, 256, 4096):
            qb = sample_q(m32, 1).expand(batch, -1).contiguous()
            for lane in ("lib", "lean"):
                ms = [bench(fns[lane], qb, bw) for bw in (False, True)]
                print(f"TIME {name} B={batch} {lane} fwd={ms[0]:.3f} fwdbwd={ms[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
