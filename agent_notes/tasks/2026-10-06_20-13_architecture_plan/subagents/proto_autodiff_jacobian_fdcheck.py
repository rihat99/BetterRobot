"""Which side is wrong: analytic chain or autodiff? Compare both to central finite differences (float64)."""
import torch

import proto_autodiff_jacobian_bench as b
from better_robot.optim import RobotVariable
from better_robot.residuals.pose import PoseResidual

for name, model, frame in b.load_models():
    fid = model.frame_id(frame)
    for label, scale in (("random target", None), ("near target (dq~0.05)", 0.05)):
        m, q, tq = b.make_inputs(model, 8, torch.float64, "cpu")
        if scale is not None:
            tq = m.integrate(q, scale * torch.randn(8, m.nv, dtype=torch.float64))
        target = b.frame_pose_raw(m, tq, fid)
        r, variants = b.make_variants(m, fid, b.frame_pose_raw)
        ja = b.analytic(m, q, target, fid)
        jr = variants["vmap_jacrev"](q, target)
        eps = 1e-6
        cols = []
        for k in range(m.nv):
            e = torch.zeros(8, m.nv, dtype=torch.float64)
            e[:, k] = eps
            cols.append((r(q, e, target) - r(q, -e, target)) / (2 * eps))
        jfd = torch.stack(cols, dim=-1)
        err = b.pose_error(b.frame_pose_raw(m, q, fid), target)
        print(f"{name} {label}: |err|max={float(err.norm(dim=-1).max()):.2f} "
              f"analytic-FD={float((ja - jfd).abs().max()):.2e} jacrev-FD={float((jr - jfd).abs().max()):.2e}")
        # Same check against the library's own PoseResidual.jacobian() (the real code path).
        try:
            var = RobotVariable(m, q.clone(), name="q")
            res = PoseResidual(var, frame=frame, target=target)
            jl = res.jacobian()[0]
            print(f"   PoseResidual.jacobian()-FD={float((jl - jfd).abs().max()):.2e}")
        except Exception as exc:  # noqa: BLE001
            print("   PoseResidual path failed:", type(exc).__name__, str(exc).splitlines()[0][:160])
