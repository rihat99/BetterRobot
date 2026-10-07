import os, sys, time, subprocess, torch
from robot_descriptions import panda_description
import better_robot as br
from better_robot.viewer import GroundLayer, RobotView, Viewer
from PIL import Image
D=os.path.dirname(sys.argv[1].split("/libasound")[0]+"/x"); D=sys.argv[1].split("/libasound")[0]
model=br.load(panda_description.URDF_PATH, dtype=torch.float32)
q0=model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
T=20; traj=q0[None].repeat(T,1); traj[:,0]=torch.linspace(-1,1,T)
B=os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell")
env=dict(os.environ, LD_LIBRARY_PATH=sys.argv[1])
for tag,flags in [("gpu-egl",["--use-angle=gl-egl","--enable-gpu","--ignore-gpu-blocklist"]),("cpu-swiftshader-defaultlights",["--use-angle=swiftshader","--enable-unsafe-swiftshader"])]:
    v=Viewer(port=8094); v.add(GroundLayer()); robot=RobotView(model,traj)
    for l in robot: v.add(l, group="panda")
    if tag.startswith("cpu"):
        v.server.scene.add_light_directional("/lights/key",intensity=0.0,cast_shadow=False); v.server.scene.add_light_hemisphere("/lights/fill",intensity=0.0); v.server.scene.configure_default_lights(True, cast_shadow=False)
    p=subprocess.Popen([B,"--no-sandbox",*flags,"--window-size=960,720",v.url],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        t0=time.perf_counter()
        while not v.server.get_clients():
            time.sleep(0.05)
            if time.perf_counter()-t0>20: raise RuntimeError("no client")
        c=next(iter(v.server.get_clients().values()))
        c.camera.position=(1.6,1.2,1.0); c.camera.look_at=(0,0,0.4)
        time.sleep(1.0); c.get_render(480,640)
        t1=time.perf_counter()
        for k in range(T):
            v.show(k); img=c.get_render(480,640)
        print(f"{tag}: {1e3*(time.perf_counter()-t1)/T:.0f} ms/frame 640x480 jpeg")
        Image.fromarray(img).save(f"{D}/{tag}.png")
    except Exception as e: print(tag, "failed:", e)
    finally:
        p.terminate(); p.wait(); v.server.stop(); time.sleep(0.5)
