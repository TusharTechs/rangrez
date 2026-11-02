"""Grab one frame from a camera: a webcam index, a phone streaming as an IP
webcam (e.g. http://192.168.1.7:8080/video), or a video file.

    RANGREZ_CAMERA=0                                # first webcam
    RANGREZ_CAMERA=1                                # Continuity Camera on a Mac, often
    RANGREZ_CAMERA=http://192.168.1.7:8080/video    # an Android phone running an IP-webcam app
"""

from __future__ import annotations

import os


def source() -> int | str:
    raw = os.environ.get("RANGREZ_CAMERA", "0").strip()
    return int(raw) if raw.isdigit() else raw


def grab(src: int | str | None = None, warmup: int = 10, width: int = 1920, height: int = 1080) -> bytes:
    """A JPEG of the current view. Warm-up frames let auto exposure settle."""
    import cv2

    src = source() if src is None else src
    cam = cv2.VideoCapture(src)
    try:
        if not cam.isOpened():
            raise RuntimeError(f"could not open camera {src!r}")
        if isinstance(src, int):
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(warmup + 1):
            ok, f = cam.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"camera {src!r} returned no frame")
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return jpeg.tobytes()
    finally:
        cam.release()
