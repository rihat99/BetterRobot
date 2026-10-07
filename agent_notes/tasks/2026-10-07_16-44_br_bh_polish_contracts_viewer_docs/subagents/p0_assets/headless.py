import os, sys, time, subprocess, torch, numpy as np
from robot_descriptions import panda_description
import better_robot as br
from better_robot.viewer import GroundLayer, RobotView, Viewer
from PIL import Image
OUT=sys.argv[1]; os.makedirs(OUT, exist_ok=True)
model=br.load(panda_description.URDF_PATH, dtype=torch.float32)
q0=model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
T=60; traj=q0[None].repeat(T,1); traj[:,0]=torch.linspace(-1,1,T); traj[:,3]=q0[3]+0.5*torch.sin(torch.linspace(0,6.28,T))
v=Viewer(port=8092); v.add(GroundLayer())
robot=RobotView(model,traj)
for l in robot: v.add(l, group="panda")
B=os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell")
env=dict(os.environ, LD_LIBRARY_PATH=sys.argv[2])
t0=time.perf_counter()
p=subprocess.Popen([B,"--no-sandbox","--use-angle=swiftshader","--enable-unsafe-swiftshader","--window-size=960,720","--remote-debugging-port=0",v.url],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    while not v.server.get_clients(): time.sleep(0.05)
    client=next(iter(v.server.get_clients().values()))
    print(f"client connected after {time.perf_counter()-t0:.2f} s")
    time.sleep(1.0)  # let meshes arrive
    img=client.get_render(720,960,transport_format="png"); print("first render", img.shape, img.dtype, f"{time.perf_counter()-t0:.2f} s")
    t1=time.perf_counter()
    for k in range(T):
        v.show(k)
        img=client.get_render(480,640,transport_format="jpeg")
        Image.fromarray(img).save(f"{OUT}/f{k:03d}.png")
    print(f"{T} frames 640x480 jpeg->png: {time.perf_counter()-t1:.2f} s ({1e3*(time.perf_counter()-t1)/T:.0f} ms/frame)")
    t1=time.perf_counter()
    for k in range(10):
        v.show(k); img=client.get_render(720,960,transport_format="png")
    print(f"10 frames 960x720 png: {1e3*(time.perf_counter()-t1)/10:.0f} ms/frame")
finally:
    p.terminate(); p.wait(); v.server.stop()
