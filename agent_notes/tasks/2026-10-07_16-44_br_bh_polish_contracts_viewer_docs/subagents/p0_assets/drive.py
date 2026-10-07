import sys, time, traceback
from better_human.viz.studio.app import HumanStudio
sd = sys.argv[1]
t = time.perf_counter()
s = HumanStudio(family="smplx", port=8091, data_dir=sd)
print("init s", round(time.perf_counter() - t, 2), "status:", s.status)
L = s.layers
def vis(tag):
    out = {}
    for n, lay in L.layers.items():
        hs = lay._handles
        out[n] = (sum(bool(h.visible) for h in hs), len(hs))
    print(tag, "visible/handles:", out, "| studio.visible:", L.visible)
def step(name, fn):
    t = time.perf_counter()
    try:
        r = fn(); print(f"[OK] {name} {time.perf_counter()-t:.2f}s status={s.status!r} ret={str(r)[:80]!r}")
    except Exception:
        print(f"[FAIL] {name}\n" + "".join(traceback.format_exc().splitlines(True)[-4:]))
vis("init")
groups = [k for k in vars(s) if 'slider' in k.lower()]
print("slider attrs:", groups)
step("random_shape betas", lambda: s.random_shape("betas"))
vis("after shape")
step("seg coloring on", lambda: s.set_segmentation_coloring(True))
step("segment.add_part", lambda: s.segment.add_part("newpart"))
step("segment.assign", lambda: s.segment.assign(__import__('numpy').arange(50)))
vis("after seg edit")
step("seg coloring off", lambda: s.set_segmentation_coloring(False))
step("set_opacity .5", lambda: s.set_opacity(0.5))
vis("after opacity")
step("select_joint 5", lambda: s.select_joint(5))
step("select_joint None", lambda: s.select_joint(None))
step("set_explode .5", lambda: L.set_explode(0.5))
step("segment.load_builtin", lambda: s.segment.load_builtin())
step("segment.save x", lambda: s.segment.save("x"))
step("segment.load x", lambda: s.segment.load("x"))
step("frames.load frames(default name)", lambda: s.frames.load("frames"))
step("frames.load contacts", lambda: s.frames.load("contacts"))
step("frames.load better_contacts", lambda: s.frames.load("better_contacts"))
step("frames.export", lambda: s.frames.export())
step("save_pose", lambda: s.save_pose("p1"))
step("load_pose", lambda: s.load_pose("p1"))
step("reset_pose", lambda: s.session.reset_pose())
step("reload density 1000", lambda: s.reload(1000.0))
vis("after reload")
step("chosen_density", lambda: s.chosen_density())
step("segment.build_bodies", lambda: s.segment.build_bodies())
step("set_mode segment", lambda: s.set_mode("segment"))
step("start_marquee", lambda: s.start_marquee())
print("groups in viewer:", list(s.viewer._groups), "entries:", list(s.viewer._groups['human'].entries))
s.viewer.server.stop()
