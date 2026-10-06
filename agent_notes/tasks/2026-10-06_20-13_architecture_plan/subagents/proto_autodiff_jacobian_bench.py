"""Autodiff vs analytic Jacobian of a 6-D frame pose residual through BetterRobot FK.

Usage: uv run python proto_autodiff_jacobian_bench.py [--device cpu|cuda] [--quick]
"""
import argparse
import json
import statistics
import sys
import time
import traceback

import torch
from torch.func import jacfwd, jacrev, vmap
import torch.autograd.forward_ad as fwAD

import better_robot as br
from better_robot.kinematics.forward import forward_kinematics_raw, frame_placements_raw
from better_robot.kinematics.jacobian import compute_joint_jacobians, get_frame_jacobian
from better_robot.lie import se3, so3
from better_robot.lie.tangents import right_jacobian_inv_se3


def load_models():
    # robot_descriptions is not installed in the uv env; use its on-disk cache directly.
    root = "/home/rikhat.akizhanov/.cache/robot_descriptions"
    panda = br.load(f"{root}/example-robot-data/robots/panda_description/urdf/panda.urdf")
    g1 = br.load(f"{root}/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf", free_flyer=True)
    g1_frame = "body_left_rubber_hand" if "body_left_rubber_hand" in g1.frame_name_to_id else list(g1.frame_name_to_id)[-1]
    p_frame = "body_panda_hand" if "body_panda_hand" in panda.frame_name_to_id else list(panda.frame_name_to_id)[-1]
    return [("panda", panda, p_frame), ("g1_freeflyer", g1, g1_frame)]


def make_inputs(model, batch, dtype, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    qs = torch.stack([model.random_configuration(generator=gen) for _ in range(batch)])
    targets_q = torch.stack([model.random_configuration(generator=gen) for _ in range(batch)])
    m = model.to(device=device, dtype=dtype) if hasattr(model, "to") else model
    return m, qs.to(device=device, dtype=dtype), targets_q.to(device=device, dtype=dtype)


def frame_pose_public(model, q, fid):
    data = br.forward_kinematics(model, q, compute_frames=True)
    return data.frame_pose_world[..., fid, :]


def frame_pose_raw(model, q, fid):
    fk = forward_kinematics_raw(model.structure, model.values, q)
    return frame_placements_raw(model.structure, model.values, fk.joint_pose_world).frame_pose_world[..., fid, :]


def pose_error(ee, target):
    return se3.log(se3.compose(se3.inverse(target), ee))


def analytic(model, q, target, fid):
    """Library path used by PoseResidual: FK + joint Jacobians + frame Jacobian + Jr^-1 chain. -> (B,6,nv)."""
    data = br.forward_kinematics(model, q, compute_frames=True)
    data = compute_joint_jacobians(model, data)
    world = get_frame_jacobian(model, data, fid)
    ee = data.frame_pose_world[..., fid, :]
    err = pose_error(ee, target)
    rot = so3.to_matrix(ee[..., 3:])
    local = torch.cat((rot.mT @ world[..., :3, :], rot.mT @ world[..., 3:, :]), dim=-2)
    return right_jacobian_inv_se3(err) @ local


def make_variants(model, fid, fk):
    nv = model.nv

    def r_batched(q, delta, target):  # q (B,nq), delta (B,nv) -> (B,6)
        return pose_error(fk(model, model.integrate(q, delta), fid), target)

    def r_single(delta, q, target):  # unbatched for vmap
        return pose_error(fk(model, model.integrate(q, delta), fid), target)

    def vmap_jacfwd(q, target):
        z = q.new_zeros(q.shape[0], nv)
        return vmap(jacfwd(r_single))(z, q, target)

    def vmap_jacrev(q, target):
        z = q.new_zeros(q.shape[0], nv)
        return vmap(jacrev(r_single))(z, q, target)

    def jacfwd_shared(q, target):  # one shared tangent d (nv,) broadcast over batch -> (B,6,nv)
        B = q.shape[0]
        return jacfwd(lambda d: r_batched(q, d.expand(B, nv), target))(q.new_zeros(nv))

    def jacrev_sum(q, target):  # outputs independent over batch -> sum then 6 VJPs -> (6,B,nv)
        z = q.new_zeros(q.shape[0], nv)
        return jacrev(lambda d: r_batched(q, d, target).sum(0))(z).permute(1, 0, 2)

    def autograd_batched(q, target):  # one backward with is_grads_batched (vmap over 6 cotangents)
        d = q.new_zeros(q.shape[0], nv, requires_grad=True)
        y = r_batched(q, d, target)
        eye = torch.eye(6, dtype=q.dtype, device=q.device)[:, None, :].expand(6, *y.shape)
        (g,) = torch.autograd.grad(y, d, eye, is_grads_batched=True)
        return g.permute(1, 0, 2)

    def autograd_loop(q, target):  # 6 plain backward passes, no vmap
        d = q.new_zeros(q.shape[0], nv, requires_grad=True)
        y = r_batched(q, d, target)
        rows = [torch.autograd.grad(y[:, k].sum(), d, retain_graph=k < 5)[0] for k in range(6)]
        return torch.stack(rows, dim=1)

    def fwd_tangent_batch(q, target):  # forward-mode duals, nv directions as extra batch axis, no vmap
        B = q.shape[0]
        qq = q.unsqueeze(0).expand(nv, B, q.shape[-1])
        tt = target.unsqueeze(0).expand(nv, B, 7)
        tangent = torch.eye(nv, dtype=q.dtype, device=q.device)[:, None, :].expand(nv, B, nv).contiguous()
        with fwAD.dual_level():
            d = fwAD.make_dual(q.new_zeros(nv, B, nv), tangent)
            y = r_batched(qq.reshape(nv * B, -1), d.reshape(nv * B, nv), tt.reshape(nv * B, 7))
            jt = fwAD.unpack_dual(y).tangent.reshape(nv, B, 6)
        return jt.permute(1, 2, 0)

    variants = {
        "vmap_jacfwd": vmap_jacfwd,
        "vmap_jacrev": vmap_jacrev,
        "jacfwd_shared": jacfwd_shared,
        "jacrev_sum": jacrev_sum,
        "autograd_batched": autograd_batched,
        "autograd_loop": autograd_loop,
        "fwd_tangent_batch": fwd_tangent_batch,
    }
    return r_batched, variants


def timeit(fn, device, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device == "cuda" else None
    return statistics.median(times) * 1e3, peak


def first_line_error(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    lib = [f for f in tb if "better_robot" in f.filename]
    where = f"{lib[-1].filename.split('src/')[-1]}:{lib[-1].lineno}" if lib else f"{tb[-1].filename}:{tb[-1].lineno}"
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]} @ {where}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batches", default="1,256,4096")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fks", default="public")
    ap.add_argument("--models", default="panda,g1_freeflyer")
    args = ap.parse_args()
    dev = args.device
    results = []
    for name, model64, frame in [x for x in load_models() if x[0] in args.models.split(",")]:
        fid = model64.frame_id(frame)
        print(f"== {name}: nq={model64.nq} nv={model64.nv} njoints={getattr(model64, 'njoints', '?')} frame={frame}", flush=True)
        for fk_name, fk in [(k, {"public": frame_pose_public, "raw": frame_pose_raw}[k]) for k in args.fks.split(",")]:
            # correctness, float64, batch 8
            m, q, tq = make_inputs(model64, 8, torch.float64, dev)
            target = frame_pose_raw(m, tq, fid)
            ref = analytic(m, q, target, fid)
            _, variants = make_variants(m, fid, fk)
            status = {}
            for vname, vfn in variants.items():
                try:
                    J = vfn(q, target)
                    status[vname] = f"ok maxabs={float((J - ref).abs().max()):.2e}"
                except Exception as exc:  # noqa: BLE001 - recording failure mode is the point
                    status[vname] = "FAIL " + first_line_error(exc)
                print(f"  [{fk_name}] check {vname}: {status[vname]}", flush=True)
            results.append({"model": name, "fk": fk_name, "check": status})
            if fk_name == "public" and all(s.startswith("ok") for s in status.values()):
                pass
            # timing, float32
            for B in [int(b) for b in args.batches.split(",")]:
                m, q, tq = make_inputs(model64, B, torch.float32, dev)
                target = frame_pose_raw(m, tq, fid)
                r_batched, variants = make_variants(m, fid, fk)
                z = q.new_zeros(B, m.nv)
                row = {"model": name, "fk": fk_name, "B": B}
                with torch.no_grad():
                    row["residual"] = timeit(lambda: r_batched(q, z, target), dev, args.repeats)
                row["analytic"] = timeit(lambda: analytic(m, q, target, fid), dev, args.repeats)
                for vname, vfn in variants.items():
                    if not status[vname].startswith("ok"):
                        continue
                    if vname == "autograd_loop" and B > 256 and name != "panda":
                        pass
                    try:
                        row[vname] = timeit(lambda: vfn(q, target), dev, args.repeats)
                    except Exception as exc:  # noqa: BLE001
                        row[vname] = "FAIL " + first_line_error(exc)
                # GN step with analytic J vs vmap_jacfwd J (if it works)
                def gn(jfn):
                    J = jfn()
                    r = r_batched(q, z, target)
                    H = J.mT @ J + 1e-2 * torch.eye(m.nv, device=q.device, dtype=q.dtype)
                    return torch.cholesky_solve(-(J.mT @ r.unsqueeze(-1)), torch.linalg.cholesky_ex(H)[0])
                row["gn_analytic"] = timeit(lambda: gn(lambda: analytic(m, q, target, fid)), dev, args.repeats)
                for best in ("jacfwd_shared", "vmap_jacfwd"):
                    if status.get(best, "").startswith("ok"):
                        row[f"gn_{best}"] = timeit(lambda: gn(lambda: variants[best](q, target)), dev, args.repeats)
                if args.compile and status.get("jacfwd_shared", "").startswith("ok"):
                    try:
                        cf = torch.compile(variants["jacfwd_shared"])
                        row["compiled_jacfwd_shared"] = timeit(lambda: cf(q, target), dev, args.repeats, warmup=3)
                        ca = torch.compile(lambda: analytic(m, q, target, fid))
                        row["compiled_analytic"] = timeit(ca, dev, args.repeats, warmup=3)
                    except Exception as exc:  # noqa: BLE001
                        row["compiled"] = "FAIL " + first_line_error(exc)
                print("  " + json.dumps(row), flush=True)
                results.append(row)
    print("RESULTS_JSON " + json.dumps(results))


if __name__ == "__main__":
    sys.exit(main())
