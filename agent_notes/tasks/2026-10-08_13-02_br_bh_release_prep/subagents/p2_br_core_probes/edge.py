"""Edge-case probes: batch shapes, dtype, gradients at Lie singular points, slerp, euler."""
import math, torch, better_robot as br
from better_robot.lie import so3, se3, tangents as tg
from better_robot.kinematics import frame_jacobian
from better_robot.dynamics import inverse_dynamics, mass_matrix, forward_dynamics
from robot_descriptions import g1_description, panda_description
def run(name, fn):
    try: print(f"{name:40s}", fn())
    except Exception as e: print(f"{name:40s} EXC {type(e).__name__}: {str(e)[:110]}")
g1 = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
pa32 = br.load(panda_description.URDF_PATH)
run("panda default dtype", lambda: pa32.joint_placements.dtype)
run("fk empty batch", lambda: tuple(g1.fk(g1.q_neutral.expand(0, -1)).frame_pose.shape))
run("fk nested batch (2,3)", lambda: tuple(g1.fk(g1.q_neutral.expand(2, 3, -1)).frame_pose.shape))
run("fk unbatched", lambda: tuple(g1.fk(g1.q_neutral).frame_pose.shape))
def dyn_shapes():
    q = g1.q_neutral.expand(2, 3, -1); st = g1.fk(q); v = torch.zeros(2, 3, g1.nv, dtype=torch.float64)
    return tuple(inverse_dynamics(g1, st, v=v, a=v).shape), tuple(mass_matrix(g1, st).shape), tuple(forward_dynamics(g1, st, v=v, tau=v).shape)
run("dyn nested batch", dyn_shapes)
def dyn_empty():
    st = g1.fk(g1.q_neutral.expand(0, -1)); v = torch.zeros(0, g1.nv, dtype=torch.float64)
    return tuple(forward_dynamics(g1, st, v=v, tau=v).shape)
run("aba empty batch", dyn_empty)
run("dyn v broadcast (nv,) with B=2", lambda: tuple(inverse_dynamics(g1, g1.fk(g1.q_neutral.expand(2, -1)), v=torch.zeros(g1.nv, dtype=torch.float64), a=torch.zeros(g1.nv, dtype=torch.float64)).shape))
run("integrate q(B,nq) v(nv,)", lambda: tuple(g1.integrate(g1.q_neutral.expand(3, -1), torch.zeros(g1.nv, dtype=torch.float64)).shape))
run("difference q0(nq,) q1(B,nq)", lambda: tuple(g1.difference(g1.q_neutral, g1.q_neutral.expand(3, -1)).shape))
def grads():
    q = torch.stack([g1.random_configuration() for _ in range(2)]).requires_grad_()
    st = g1.fk(q); J = frame_jacobian(g1, st, frame="body_torso_link", reference="local")
    tau = inverse_dynamics(g1, st, v=torch.ones(2, g1.nv, dtype=torch.float64), a=torch.ones(2, g1.nv, dtype=torch.float64))
    (J.sum() + tau.sum() + mass_matrix(g1, st).sum()).backward()
    return bool(torch.isfinite(q.grad).all()), float(q.grad.abs().max())
run("grad J+rnea+crba wrt q finite", grads)
def gfin(f, x):
    x = x.clone().requires_grad_(); f(x).sum().backward(); return bool(torch.isfinite(x.grad).all())
z3 = torch.zeros(3, dtype=torch.float64); z6 = torch.zeros(6, dtype=torch.float64)
I7 = se3.identity(dtype=torch.float64)
run("so3.exp grad at 0", lambda: gfin(so3.exp, z3))
run("so3.log grad at identity", lambda: gfin(so3.log, so3.identity(dtype=torch.float64)))
run("se3.exp grad at 0", lambda: gfin(se3.exp, z6))
run("se3.log grad at identity", lambda: gfin(se3.log, I7))
run("right_jac_so3 grad at 0", lambda: gfin(tg.right_jacobian_so3, z3))
run("right_jac_inv_se3 grad at 0", lambda: gfin(tg.right_jacobian_inv_se3, z6))
qpi = torch.tensor([1.0, 0, 0, 1e-9], dtype=torch.float64)
run("so3.log near pi", lambda: so3.log(qpi).tolist())
run("so3.log -q == log q (double cover)", lambda: float((so3.log(-so3.exp(torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64))) - torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)).abs().max()))
w = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
run("exp/log roundtrip |w|=3", lambda: float((so3.log(so3.exp(w)) - w).abs().max()))
e = torch.tensor([0.3, math.pi / 2, -0.4], dtype=torch.float64)
run("euler gimbal roundtrip R err", lambda: float((so3.to_matrix(so3.from_euler(so3.to_euler(so3.from_euler(e)))) - so3.to_matrix(so3.from_euler(e))).abs().max()))
a = so3.exp(torch.tensor([0.1, 0, 0], dtype=torch.float64)); b = -so3.exp(torch.tensor([0.3, 0, 0], dtype=torch.float64))
run("slerp(a,-b,0.5) angle (want 0.2)", lambda: float(so3.log(so3.slerp(a, b, torch.tensor(0.5, dtype=torch.float64))).norm()))
run("slerp(a,a,0.5) finite", lambda: bool(torch.isfinite(so3.slerp(a, a, torch.tensor(0.5, dtype=torch.float64))).all()))
run("float32 fk dtype", lambda: g1.to(dtype=torch.float32).fk(g1.q_neutral.float()).frame_pose.dtype)
run("rand cfg float64", lambda: g1.random_configuration().dtype)
