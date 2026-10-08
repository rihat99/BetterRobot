"""Independent Pinocchio spot check on G1 + free flyer (nq != nv), random q, v, a."""
import numpy as np, torch, pinocchio as pin, better_robot as br
from better_robot.kinematics import frame_jacobian, joint_jacobian
from better_robot.dynamics import inverse_dynamics, mass_matrix, forward_dynamics
from robot_descriptions import g1_description
torch.manual_seed(3); np.random.seed(3)
m = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
pm = pin.buildModelFromUrdf(g1_description.URDF_PATH, pin.JointModelFreeFlyer()); pd = pm.createData()
assert m.nq == pm.nq and m.nv == pm.nv, (m.nq, pm.nq, m.nv, pm.nv)
B = 4
q = torch.stack([m.random_configuration() for _ in range(B)])
v = torch.randn(B, m.nv, dtype=torch.float64); a = torch.randn(B, m.nv, dtype=torch.float64)
tau = torch.randn(B, m.nv, dtype=torch.float64)
st = m.fk(q)
err = {k: 0.0 for k in ["fk_frame", "J_world", "J_local", "J_lwa", "Jjoint_lwa", "rnea", "crba", "aba", "integrate", "difference"]}
REF = {"world": pin.WORLD, "local": pin.LOCAL, "local_world_aligned": pin.LOCAL_WORLD_ALIGNED}
nchecked = 0
for b in range(B):
    qn = q[b].numpy(); vn = v[b].numpy()
    pin.forwardKinematics(pm, pd, qn); pin.updateFramePlacements(pm, pd); pin.computeJointJacobians(pm, pd, qn)
    for name in m.frame_names[1:]:
        pname = name.removeprefix("body_")
        if not pm.existFrame(pname): continue
        fid = pm.getFrameId(pname); bid = m.frame_id(name); nchecked += 1
        M = pd.oMf[fid]; P = st.frame_pose[b, bid]
        Rb = pin.Quaternion(float(P[6]), float(P[3]), float(P[4]), float(P[5])).matrix()
        err["fk_frame"] = max(err["fk_frame"], np.abs(P[:3].numpy() - M.translation).max(), np.abs(Rb - M.rotation).max())
    for fname in ["left_ankle_roll_link", "right_wrist_yaw_link", "torso_link"]:
        fid = pm.getFrameId(fname); fname = "body_" + fname
        for key, ref in REF.items():
            Jb = frame_jacobian(m, st, frame=fname, reference=key)[b].numpy()
            Jp = pin.getFrameJacobian(pm, pd, fid, ref)
            err[{"world": "J_world", "local": "J_local", "local_world_aligned": "J_lwa"}[key]] = max(err[{"world": "J_world", "local": "J_local", "local_world_aligned": "J_lwa"}[key]], np.abs(Jb - Jp).max())
    jid = m.joint_id("left_knee_joint")
    Jj = joint_jacobian(m, st, joint=jid)[b].numpy()
    err["Jjoint_lwa"] = max(err["Jjoint_lwa"], np.abs(Jj - pin.getJointJacobian(pm, pd, pm.getJointId("left_knee_joint"), pin.LOCAL_WORLD_ALIGNED)).max())
    err["rnea"] = max(err["rnea"], np.abs(inverse_dynamics(m, st, v=v, a=a)[b].numpy() - pin.rnea(pm, pd, qn, vn, a[b].numpy())).max())
    Mp = pin.crba(pm, pd, qn); Mp = np.triu(Mp) + np.triu(Mp, 1).T
    err["crba"] = max(err["crba"], np.abs(mass_matrix(m, st)[b].numpy() - Mp).max())
    err["aba"] = max(err["aba"], np.abs(forward_dynamics(m, st, v=v, tau=tau)[b].numpy() - pin.aba(pm, pd, qn, vn, tau[b].numpy())).max())
    err["integrate"] = max(err["integrate"], np.abs(m.integrate(q, v)[b].numpy() - pin.integrate(pm, qn, vn)).max())
    q1 = m.random_configuration()
    err["difference"] = max(err["difference"], np.abs(m.difference(q[b], q1).numpy() - pin.difference(pm, qn, q1.numpy())).max())
print("frames checked", nchecked)
for k, e in err.items(): print(f"{k:12s} max|err| = {e:.2e}")
