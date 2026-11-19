"""``Viewer.record``: one headless Chromium runs viser's own client, steps it frame by frame and encodes in the page.

The page opens at a fixed pixel ratio with its canvas pinned to the output size and the client's render loop stopped.
Per frame, Python pushes the step (``Viewer.show``) and a marker through viser's websocket; one call into the page
waits for the marker, applies the messages, renders once and hands the canvas to the browser's video encoder
(WebCodecs), so no image is read back per frame. The encoded chunks stream to Python, which writes the WebM itself
and hands it to ``ffmpeg`` for an MP4. A ``.png`` is one still of the same canvas. The page is reached through viser
client internals (``window.__viserMutable``, the react-three-fiber store of its scene, the camera controls); ``PROBE``
checks their shape, and a mismatch names the viser version.

Chromium is ``$BETTER_ROBOT_CHROME``, else ``chromium``, ``google-chrome`` or ``chrome-headless-shell`` on ``PATH``,
driven over the DevTools protocol on a pipe. A machine without ``libasound.so.2`` (common on servers) makes Chromium
exit at start: put a copy of the library on ``LD_LIBRARY_PATH``. ``gpu=None`` renders on the CPU (SwiftShader, which
ships with Chromium); ``gpu=<index>`` renders on that NVIDIA GPU (the ``nvidia-smi`` index) through ANGLE on Vulkan,
moved to the front of Vulkan's device list by Mesa's device-select layer (``DRI_PRIME``).
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from collections.abc import Sequence
from glob import glob
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import numpy as np

from ..exceptions import ArgumentError, OptionalDependencyError

if TYPE_CHECKING:
    from .viewer import Viewer

_CHROME_NAMES = ("chromium", "google-chrome", "chrome-headless-shell")
_MARKER = "/__record_marker"  # a hidden node whose position tells the page that a frame's messages have arrived
_CODECS = {"vp09.00.51.08": "V_VP9", "vp8": "V_VP8"}  # WebCodecs codec -> WebM codec id, in order of preference
_H264 = "avc1.640034"  # High profile, level 5.2: up to 3840 x 2160 at 60 fps; an .mp4 takes its stream as it is

# The client internals the recorder drives; the guarded test runs this probe against the pinned viser.
PROBE = r"""
(() => {
  const m = window.__viserMutable;
  if (m === undefined) return ['window.__viserMutable'];
  const missing = [];
  if (!(m.canvas instanceof HTMLCanvasElement)) missing.push('__viserMutable.canvas');
  if (!Array.isArray(m.messageQueue)) missing.push('__viserMutable.messageQueue');
  if (!m.cameraControl || typeof m.cameraControl.setTarget !== 'function') missing.push('__viserMutable.cameraControl');
  if (!m.nodeRefFromName || !m.nodeRefFromName['']) missing.push("__viserMutable.nodeRefFromName['']");
  const root = m.scene && m.scene.__r3f && m.scene.__r3f.root;
  const state = root && typeof root.getState === 'function' ? root.getState() : null;
  if (state === null) return [...missing, '__viserMutable.scene.__r3f.root.getState'];
  for (const key of ['advance', 'setFrameloop', 'setSize']) if (typeof state[key] !== 'function') missing.push('r3f state.' + key);
  if (!state.gl || typeof state.gl.setClearColor !== 'function') missing.push('r3f state.gl');
  return missing;
})()
"""

# The recorder installed in the page as window.__brRecorder; Python awaits its methods with Runtime.evaluate.
_RECORDER = r"""
(() => {
  const m = window.__viserMutable, root = m.scene.__r3f.root, state = root.getState(), canvas = m.canvas;  // state's functions are stable
  const tick = () => new Promise((done) => { const c = new MessageChannel(); c.port1.onmessage = () => done(); c.port2.postMessage(0); });
  const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
  const base64 = (bytes) => {
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const rec = { chunks: [], error: null, encoder: null, fps: 30, gop: 60 };
  rec.pin = async (width, height) => {
    const gl = state.gl.getContext();
    const limit = Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), ...gl.getParameter(gl.MAX_VIEWPORT_DIMS));
    const renderer = gl.getParameter(gl.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL);
    if (width > limit || height > limit) return [0, 0, limit, renderer];
    // A video frame has no alpha: clear with the page colour a browser shows behind the transparent canvas.
    let element = canvas, background = '';
    while (element && (background === '' || background === 'transparent' || background === 'rgba(0, 0, 0, 0)')) {
      background = getComputedStyle(element).backgroundColor; element = element.parentElement;
    }
    state.setFrameloop('never');
    const box = canvas.parentElement.parentElement;  // react-three-fiber's container; the GUI panel may dock beside it
    Object.assign(box.style, { position: 'fixed', left: '0px', top: '0px', width: width + 'px', height: height + 'px', zIndex: 2147483647 });
    state.setSize(width, height);  // viser 1.1 sizes the renderer to its dock layout itself: the container alone no longer does
    for (let i = 0; i < 300 && canvas.getBoundingClientRect().width !== width; i++) await sleep(10);
    state.advance(performance.now());
    state.gl.setClearColor(background || '#ffffff', 1);
    return [canvas.width, canvas.height, limit, renderer];
  };
  rec.render = async (id, camera) => {
    // A re-render of viser's canvas component (an intersection or resize event) restarts the loop, which then
    // applies the queue by itself: stop it again, and count a marker already applied as arrived. A loop stopped
    // between taking the marker from the queue and drawing it leaves the node where it was: draw now and then.
    if (root.getState().frameloop !== 'never') state.setFrameloop('never');
    const t0 = performance.now(), marker = () => m.nodeRefFromName['__MARKER__'];
    const arrived = () => m.messageQueue.some((message) => message.type === 'SetPositionMessage' && message.name === '__MARKER__' && message.position[0] === id) || (marker() !== undefined && marker().position.x === id);
    let drawn = t0;
    while (!arrived()) {
      if (performance.now() - t0 > 30000) throw new Error('the frame marker ' + id + ' (a SetPositionMessage) never reached the page');
      if (performance.now() - drawn > 100) { state.advance(performance.now()); drawn = performance.now(); }
      await tick();
    }
    // Positions apply inside one advance, but changed props (points, vertices, batched poses, colours) re-render
    // React after it: apply the messages without drawing, let React commit, then draw.
    state.scene.visible = false; state.advance(performance.now()); state.scene.visible = true;
    await tick(); await tick();
    if (camera !== null) {  // [position, look_at] in viser's world, moved into three.js's by the root node
      const world = m.nodeRefFromName[''].matrixWorld, Vector = m.camera.position.constructor;
      const [p, t] = camera.map((v) => new Vector(v[0], v[1], v[2]).applyMatrix4(world));
      m.cameraControl.setPosition(p.x, p.y, p.z, false); m.cameraControl.setTarget(t.x, t.y, t.z, false);
    }
    state.advance(performance.now());
  };
  rec.settle = async (id, camera) => {
    // Meshes and textures load asynchronously: draw until what is drawn stops changing for half a second.
    await rec.render(id, camera);
    const info = state.gl.info, t0 = performance.now();
    let last = '', same = 0;
    while (same < 10 && performance.now() - t0 < 30000) {
      await sleep(50);
      state.advance(performance.now());
      const key = [info.render.triangles, info.render.points, info.render.lines, info.memory.geometries, info.memory.textures, (info.programs || []).length].join();
      same = key === last ? same + 1 : 0; last = key;
    }
  };
  rec.start = async (codecs, width, height, bitrate, fps) => {
    for (const codec of codecs) {
      const config = { codec, width, height, bitrate, framerate: fps, latencyMode: 'quality', avc: { format: 'annexb' } };
      if (!(await VideoEncoder.isConfigSupported(config)).supported) continue;
      rec.fps = fps; rec.gop = Math.max(1, Math.round(2 * fps));
      rec.encoder = new VideoEncoder({
        output: (chunk) => { const bytes = new Uint8Array(chunk.byteLength); chunk.copyTo(bytes); rec.chunks.push([chunk.timestamp, chunk.type === 'key', bytes]); },
        error: (error) => { rec.error = String(error); },
      });
      rec.encoder.configure(config);
      return codec;
    }
    return null;
  };
  rec.drain = () => {
    if (rec.error !== null) throw new Error('video encoder: ' + rec.error);
    const out = rec.chunks.map(([time, key, bytes]) => [time, key, base64(bytes)]);
    rec.chunks = [];
    return out;
  };
  rec.frame = async (index, camera) => {
    await rec.render(index + 1, camera);
    const frame = new VideoFrame(canvas, { timestamp: Math.round(index * 1e6 / rec.fps), duration: Math.round(1e6 / rec.fps) });
    rec.encoder.encode(frame, { keyFrame: index % rec.gop === 0 });
    frame.close();
    // A bounded queue: a 4K frame is 33 MB, and on a GPU the renderer outruns the encoder.
    while (rec.encoder.encodeQueueSize > 4) await new Promise((done) => rec.encoder.addEventListener('dequeue', done, { once: true }));
    return rec.drain();
  };
  rec.finish = async () => { await rec.encoder.flush(); rec.encoder.close(); return rec.drain(); };
  rec.png = async () => base64(new Uint8Array(await (await new Promise((done) => canvas.toBlob(done, 'image/png'))).arrayBuffer()));
  window.__brRecorder = rec;
  return true;
})()
""".replace("__MARKER__", _MARKER)


def _record(
    viewer: Viewer,
    path: Path,
    *,
    steps: list[int],
    size: tuple[int, int],
    camera: tuple[np.ndarray, np.ndarray],
    fov: float,
    gpu: int | None,
    fps: float,
    bitrate: float,
) -> Path:
    """Record ``steps`` of ``viewer`` into ``path`` with the arguments ``Viewer.record`` checked; returns ``path``."""
    suffix = path.suffix.lower()
    ffmpeg = _ffmpeg(required=True) if suffix == ".mp4" else None
    codecs = [] if suffix == ".png" else [_H264, *_CODECS] if ffmpeg else list(_CODECS)
    shown, playing = viewer.timeline.step, viewer.timeline.playing
    viewer.timeline.pause()
    try:
        with viewer.lock, _Page(viewer, size, camera, fov, gpu) as page:
            data, codec = page.record(steps, codecs=codecs, fps=fps, bitrate=bitrate)
    finally:
        viewer.show(shown)
        if playing:
            viewer.timeline.play()
    if ffmpeg is None:
        path.write_bytes(data)
        return path
    if codec == _H264:  # only the container changes; -r stamps the frames, the stream's own timing is the encoder's
        convert = ["-f", "h264", "-r", f"{fps:g}", "-i", "pipe:0", "-c", "copy"]
    else:  # a browser without an H.264 encoder: transcode its WebM, which takes longer than the rendering
        convert = [
            "-f",
            "webm",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    command = [ffmpeg, "-y", "-loglevel", "error", *convert, "-movflags", "+faststart", str(path)]
    run = subprocess.run(command, input=data, capture_output=True, check=False)
    if run.returncode != 0:  # the encoded video is the costly part: keep it
        kept = path.with_suffix(".h264" if codec == _H264 else ".webm")
        kept.write_bytes(data)
        reason = run.stderr.decode(errors="replace").strip()
        raise OptionalDependencyError(
            f"an ffmpeg that writes {path.name} ({ffmpeg} exited with code {run.returncode}"
            f"{f': {reason}' if reason else ''})",
            install=f"apt install ffmpeg, or point BETTER_ROBOT_FFMPEG at a working one; the video is kept in {kept}",
        )
    return path


def _chrome() -> str:
    """Path of a Chromium binary: ``$BETTER_ROBOT_CHROME``, else the first of ``_CHROME_NAMES`` on ``PATH``."""
    path = os.environ.get("BETTER_ROBOT_CHROME") or next(filter(None, map(shutil.which, _CHROME_NAMES)), None)
    if path is None:
        raise OptionalDependencyError(
            "Chromium (Viewer.record)",
            install="npx playwright install chromium-headless-shell, then put chrome-headless-shell on PATH or point "
            "BETTER_ROBOT_CHROME at it (chromium or google-chrome work too)",
        )
    return path


def _ffmpeg(*, required: bool) -> str | None:
    """Path of ``ffmpeg`` (``$BETTER_ROBOT_FFMPEG``, else on ``PATH``); when it is missing, ``None`` or the error."""
    path = os.environ.get("BETTER_ROBOT_FFMPEG") or shutil.which("ffmpeg")
    if path is None and required:
        raise OptionalDependencyError(
            "ffmpeg (Viewer.record to .mp4)",
            install="apt install ffmpeg, or set BETTER_ROBOT_FFMPEG to its path; or record to .webm, which needs nothing",
        )
    return path


@final
class _Page:
    """A headless Chromium on the viewer's page: canvas pinned to ``size``, the recorder in.

    ``camera`` is ``(position, look_at)``, each ``(3,)``, or ``(S, 3)`` for a camera that moves with the ``S`` recorded
    steps. A context manager: entering launches the browser and waits for viser's client, exiting kills it.
    """

    def __init__(
        self, viewer: Viewer, size: tuple[int, int], camera: tuple[np.ndarray, np.ndarray], fov: float, gpu: int | None
    ) -> None:
        self.viewer, self.size, self.camera, self.fov, self.gpu = viewer, size, camera, fov, gpu
        # A server bound to every interface is reached through loopback: 0.0.0.0 and :: are no destination on every
        # system, and Chromium is closing them.
        self.origin = viewer.url.replace("//0.0.0.0:", "//127.0.0.1:").replace("//[::]:", "//[::1]:")
        self._profile = tempfile.TemporaryDirectory(prefix="better_robot_record_")
        self._process: subprocess.Popen | None = None
        self._pipes: tuple[int, int] | None = None  # (write to Chromium, read from Chromium)
        self._buffer = b""
        self._id = 0
        self._session = ""
        self._prime = ""  # DRI_PRIME of the requested GPU

    def __enter__(self) -> _Page:
        """Launch Chromium on the page and prepare it; ``OptionalDependencyError`` or ``ArgumentError`` if it cannot."""
        width, height = self.size
        position, look_at = (",".join(f"{value:.9g}" for value in vector.reshape(-1, 3)[0]) for vector in self.camera)
        up = ",".join(f"{value:g}" for value in self.viewer.theme.up_vector)
        url = (
            f"{self.origin}/?fixedDpr=1&initialCameraPosition={position}&initialCameraLookAt={look_at}"
            f"&initialCameraUp={up}&initialCameraFov={self.fov:.9g}"
        )
        environment = dict(os.environ)
        if self.gpu is None:
            flags = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        else:
            environment["DRI_PRIME"] = self._prime = _dri_prime(self.gpu)
            flags = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]
        command = [
            _chrome(),
            "--headless",
            "--no-sandbox",
            "--remote-debugging-pipe",
            f"--user-data-dir={self._profile.name}",
            f"--window-size={width},{height}",
            f"--unsafely-treat-insecure-origin-as-secure={self.origin}",  # WebCodecs needs a secure origin
            "--hide-scrollbars",
            "--mute-audio",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            *flags,
            url,
        ]
        to_chrome, from_chrome = os.pipe(), os.pipe()
        # Chromium reads the protocol on fd 3 and writes it on fd 4: the pipes arrive as stdin and stdout, and the shell
        # moves them there (it redirects single-digit descriptors only) before it becomes Chromium.
        self._process = subprocess.Popen(
            ["/bin/sh", "-c", 'exec "$@" 3<&0 4>&1 </dev/null >/dev/null', "sh", *command],
            stdin=to_chrome[0],
            stdout=from_chrome[1],
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        os.close(to_chrome[0])
        os.close(from_chrome[1])
        self._pipes = (to_chrome[1], from_chrome[0])
        try:
            self._open()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        """Kill the browser and remove its profile."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        if self._pipes is not None:
            for fd in self._pipes:
                os.close(fd)
            self._pipes = None
        self._profile.cleanup()

    def _open(self) -> None:
        """Attach to the page, wait for viser's client and check its internals, install the recorder, pin the canvas."""
        targets = self._send("Target.getTargets")["targetInfos"]
        page = next(target for target in targets if target["type"] == "page")
        self._session = self._send("Target.attachToTarget", targetId=page["targetId"], flatten=True)["sessionId"]
        deadline = time.monotonic() + 60.0
        while (missing := self.eval(PROBE, check=False)) != []:
            if missing == ["window.__viserMutable"]:  # no viser client: the page may not have loaded at all
                shown = self.eval("location.href", check=False)
                if str(shown).startswith("chrome-error:") or time.monotonic() > deadline:
                    raise ArgumentError(
                        f"Chromium could not load the viewer's page from {self.origin} (it shows {shown!r}); the "
                        "server must be reachable at that address from this host"
                    )
            if time.monotonic() > deadline:
                raise OptionalDependencyError(
                    f"viser {version('viser')} with the client internals Viewer.record drives (missing: {missing})",
                    install="uv sync --extra viewer, which installs the viser version the lock pins",
                )
            time.sleep(0.05)
        self.eval(_RECORDER)
        width, height, limit, renderer = self.call("pin", *self.size)
        if self.gpu is not None and "NVIDIA" not in renderer:
            raise ArgumentError(f"gpu={self.gpu} is not reachable: Chromium renders on {renderer!r}; pass gpu=None")
        # NVIDIA's Vulkan driver keeps a few MiB on the first GPU of any Vulkan program; the rendering GPU must be listed.
        held = _gpus_of(self._process.pid) if self.gpu is not None and self._process is not None else set()
        if held and self._prime not in held:
            raise ArgumentError(
                f"gpu={self.gpu} is not reachable: Chromium holds {sorted(held)} instead; pass gpu=None"
            )
        if (width, height) != self.size:
            raise ArgumentError(f"size {self.size} is above this renderer's limit of {limit} pixels ({renderer})")

    def record(self, steps: Sequence[int], *, codecs: Sequence[str], fps: float, bitrate: float) -> tuple[bytes, str]:
        """``(data, codec)``: a video of ``steps`` at ``fps`` and ``bitrate`` bits per second in the first of ``codecs``
        the browser encodes (H.264 as its raw stream, VP9 or VP8 as a WebM), or with no ``codecs`` a PNG of
        ``steps[0]`` (codec ``"png"``)."""
        server, (position, look_at) = self.viewer.server, self.camera
        marker = server.scene.add_frame(_MARKER, show_axes=False, visible=False)

        def push(index: int, marker_id: int) -> list | None:
            """Send ``steps[index]``, then the marker at ``marker_id`` (never 0, where it starts); the frame's camera."""
            with server.atomic():
                self.viewer.show(steps[index])
                marker.position = (float(marker_id), 0.0, 0.0)
            server.flush()  # skip viser's 1/60 s send window
            return [position[index].tolist(), look_at[index].tolist()] if position.ndim == 2 else None

        try:
            self.call("settle", -1, push(0, -1))
            if not codecs:
                return base64.b64decode(self.call("png")), "png"
            width, height = self.size
            codec = self.call("start", list(codecs), width, height, round(bitrate), fps)
            if codec is None:
                raise ArgumentError(f"this browser encodes none of {list(codecs)} at {width}x{height}; record smaller")
            chunks = []
            for index in range(len(steps)):
                chunks += self.call("frame", index, push(index, index + 1))
            chunks += self.call("finish")
        finally:
            marker.remove()
        chunks = [(int(time), bool(key), base64.b64decode(data)) for time, key, data in chunks]
        if codec == _H264:
            return b"".join(data for _, _, data in chunks), codec
        return _webm(chunks, codec=_CODECS[codec], width=width, height=height, fps=fps), codec

    def call(self, method: str, *args: Any) -> Any:
        """Await ``window.__brRecorder.<method>(*args)`` in the page and return its JSON value."""
        return self.eval(f"window.__brRecorder.{method}({', '.join(map(json.dumps, args))})")

    def eval(self, expression: str, *, check: bool = True) -> Any:
        """Await ``expression`` in the page and return its JSON value; a page error raises, or returns ``None`` when
        ``check`` is off, as does a page between two documents, where DevTools finds no context to evaluate in."""
        try:
            result = self._send(
                "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True, session=self._session
            )
        except ArgumentError:  # DevTools' refusal while the page navigates: the wait polls again
            if check:
                raise
            return None
        if "exceptionDetails" in result:
            if not check:
                return None
            details = result["exceptionDetails"]
            raise ArgumentError(f"the recording page failed: {details.get('exception', details).get('description')}")
        return result["result"].get("value")

    def _send(self, method: str, *, session: str = "", **params: Any) -> dict:
        """Send one DevTools command and return its result; events in between are skipped."""
        assert self._pipes is not None and self._process is not None
        self._id += 1
        message = {"id": self._id, "method": method, "params": params} | ({"sessionId": session} if session else {})
        os.write(self._pipes[0], json.dumps(message).encode() + b"\0")
        while True:
            while b"\0" not in self._buffer:
                data = os.read(self._pipes[1], 1 << 20)
                if not data:
                    raise OptionalDependencyError(
                        f"a headless Chromium that starts (it exited with code {self._process.wait()})",
                        install="put libasound.so.2 on LD_LIBRARY_PATH (see better_robot.viewer.record), or point "
                        "BETTER_ROBOT_CHROME at a Chromium that starts headless",
                    )
                self._buffer += data
            raw, _, self._buffer = self._buffer.partition(b"\0")
            reply = json.loads(raw)
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise ArgumentError(f"{method}: {reply['error'].get('message')}")
                return reply["result"]


def _dri_prime(gpu: int) -> str:
    """``DRI_PRIME`` value that puts GPU ``gpu`` (``nvidia-smi`` index) first in Vulkan's device list."""
    layers = glob("/usr/share/vulkan/implicit_layer.d/*device_select*.json")
    if not layers + glob("/etc/vulkan/implicit_layer.d/*device_select*.json"):
        raise ArgumentError(
            f"gpu={gpu} is not reachable: Mesa's Vulkan device-select layer is missing (mesa-vulkan-drivers); "
            "pass gpu=None"
        )
    nvidia_smi = shutil.which("nvidia-smi")
    query = [nvidia_smi or "nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(gpu)]
    run = subprocess.run(query, capture_output=True, text=True, check=False) if nvidia_smi else None
    if run is None or run.returncode != 0:
        reason = "nvidia-smi is not on PATH" if run is None else (run.stdout.strip() or run.stderr.strip())
        raise ArgumentError(f"gpu={gpu} is not reachable: {reason}; pass gpu=None")
    return _prime(run.stdout.strip())


def _prime(bus: str) -> str:
    """``DRI_PRIME``'s spelling of the PCI bus ``bus`` as ``nvidia-smi`` prints it (``00000000:A5:00.0``)."""
    domain, number, device_function = bus.lower().split(":")
    return f"pci-{int(domain, 16):04x}_{number}_{device_function.replace('.', '_')}"


def _gpus_of(browser: int) -> set[str]:
    """GPUs (as ``DRI_PRIME`` spells them) on which ``nvidia-smi`` lists a descendant of process ``browser``; empty
    when it lists none of them (as inside a container)."""
    listing = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True, text=True, check=False).stdout
    found, bus = set(), ""
    for line in listing.splitlines():
        if line.startswith("GPU "):
            bus = line.split()[1]
        elif match := re.search(r"Process ID\s+:\s+(\d+)", line):
            pid = int(match.group(1))
            while pid > 1 and pid != browser:  # up the parents: Chromium's GPU process is a grandchild
                try:
                    pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
                except OSError:
                    break
            if pid == browser:
                found.add(_prime(bus))
    return found


def _webm(chunks: Sequence[tuple[int, bool, bytes]], *, codec: str, width: int, height: int, fps: float) -> bytes:
    """A WebM file of one video track (``codec`` ``V_VP9`` or ``V_VP8``) from ``chunks`` ``[(time_us, key, bytes)]``."""

    def element(element_id: int, payload: bytes) -> bytes:  # EBML: id, 8-byte size, payload
        size = (len(payload) | 1 << 56).to_bytes(8, "big")
        return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big") + size + payload

    def uint(element_id: int, value: int) -> bytes:
        return element(element_id, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))

    header = element(  # EBML version 1, ids up to 4 bytes, sizes up to 8, doctype webm version 4 (read 2)
        0x1A45DFA3,
        uint(0x4286, 1)
        + uint(0x42F7, 1)
        + uint(0x42F2, 4)
        + uint(0x42F3, 8)
        + element(0x4282, b"webm")
        + uint(0x4287, 4)
        + uint(0x4285, 2),
    )
    info = element(  # times in milliseconds, muxing and writing application, duration
        0x1549A966,
        uint(0x2AD7B1, 1_000_000)
        + element(0x4D80, b"better_robot")
        + element(0x5741, b"better_robot")
        + element(0x4489, struct.pack(">d", chunks[-1][0] / 1000.0 + 1000.0 / fps)),
    )
    track = uint(0xD7, 1) + uint(0x73C5, 1) + uint(0x83, 1) + element(0x86, codec.encode())
    track += uint(0x23E383, round(1e9 / fps)) + element(0xE0, uint(0xB0, width) + uint(0xBA, height))
    clusters, start, blocks = [], 0, b""
    for time_us, key, data in chunks:
        ms = round(time_us / 1000.0)
        if not blocks or key or ms - start > 30000:  # a block's time is 16 bits after its cluster's
            if blocks:
                clusters.append(element(0x1F43B675, uint(0xE7, start) + blocks))
            start, blocks = ms, b""
        blocks += element(0xA3, b"\x81" + struct.pack(">h", ms - start) + (b"\x80" if key else b"\x00") + data)
    clusters.append(element(0x1F43B675, uint(0xE7, start) + blocks))
    return header + element(0x18538067, info + element(0x1654AE6B, element(0xAE, track)) + b"".join(clusters))
