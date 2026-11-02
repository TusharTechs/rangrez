"""Camera sources without a camera: a phone streaming MJPEG as an IP webcam, and a video file.
Frames are real-looking photos with a little hand shake between them."""

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from PIL import Image

from rangrez.assisted import AssistedRig
from rangrez.camera import grab
from rangrez.colour import delta_e_2000
from rangrez.rig import Recipe, execute
from rangrez.sim import VirtualRig
from rangrez.vision import measure


def shaky_frames(jpeg: bytes, n: int = 15, seed: int = 0) -> list[bytes]:
    """A phone held over the strip: each frame shifted by a pixel or two."""
    rng = np.random.default_rng(seed)
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    out = []
    for _ in range(n):
        dy, dx = rng.integers(-2, 3, 2)
        f = np.roll(img, (dy, dx), axis=(0, 1))
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=85)
        out.append(buf.getvalue())
    return out


@pytest.fixture
def dyed():
    rig = VirtualRig(seed=6, scene="kitchen_tube")
    shot = execute(rig, Recipe((0.4, 4.0, 1.0)))
    return rig, shot.image_jpeg


@pytest.fixture
def ip_webcam(dyed):
    """An IP-webcam phone app: multipart MJPEG at /video."""
    frames = shaky_frames(dyed[1])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for f in frames * 4:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f))
                    self.wfile.write(f + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/video"
    server.shutdown()


def test_phone_as_ip_webcam_feeds_the_assisted_lab(dyed, ip_webcam):
    rig, _ = dyed
    jpeg = AssistedRig(camera=ip_webcam).capture()
    m = measure(jpeg, rig.segmenter())
    assert m.ok, m.flags
    assert float(delta_e_2000(m.lab, rig.last_swatch_lab)) < 2.0


def test_video_file_as_camera(dyed, tmp_path):
    rig, jpeg = dyed
    path = str(tmp_path / "clip.mp4")
    first = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for f in shaky_frames(jpeg, n=20, seed=1):
        out.write(cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR))
    out.release()
    m = measure(grab(path), rig.segmenter())
    assert float(delta_e_2000(m.lab, rig.last_swatch_lab)) < 2.0


def test_missing_camera_says_so():
    with pytest.raises(RuntimeError, match="could not open camera"):
        grab("/nonexistent/camera.mp4")
