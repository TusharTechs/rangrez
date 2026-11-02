"""Live lab dashboard: FastAPI + server-sent events.

    uv run rangrez-lab            # simulator rig, hosted SAM + Wonder if MODEL_API_KEY is set
    RANGREZ_RIG=serial RANGREZ_PORT=/dev/cu.usbserial-0001 uv run rangrez-lab
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from .agent import WONDER_MODEL, run_deterministic, run_with_wonder
from .campaign import Campaign
from .colour import hex_to_lab, lab_to_hex
from .photoreal import scene_names
from .rig import DYES, UNIT
from .sim import VirtualRig
from .vision import CallLog, MuseSamSegmenter, target_from_photo

load_dotenv()
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Rangrez")


class Lab:
    """One rig, one campaign at a time, many dashboard viewers."""

    def __init__(self) -> None:
        self.log = CallLog()
        self.subscribers: list[queue.Queue] = []
        self.campaign: Campaign | None = None
        self.thread: threading.Thread | None = None
        self.log.listeners.append(lambda call: self.publish({
            "kind": "api", "model": call.model, "purpose": call.purpose,
            "seconds": round(call.seconds, 2), "ok": call.ok, "detail": call.detail, "at": time.time(),
        }))

    @property
    def online(self) -> bool:
        return bool(os.environ.get("MODEL_API_KEY")) and os.environ.get("RANGREZ_OFFLINE") != "1"

    def make_rig(self, target_lab: np.ndarray | None):
        kind = os.environ.get("RANGREZ_RIG", "sim")
        if kind == "assisted":
            if not self.online:
                raise HTTPException(400, "assisted mode measures real photos with hosted SAM 3.1: set MODEL_API_KEY")
            from .assisted import AssistedRig

            return AssistedRig(
                on_instructions=lambda w: self.publish({"kind": "assist", **w, "at": time.time()}),
                camera_index=int(os.environ.get("RANGREZ_CAMERA", "0")),
            )
        if kind == "serial":
            from .serial_rig import SerialRig

            return SerialRig(os.environ["RANGREZ_PORT"])
        # A different realistic scene and paper each campaign; the colour card in view.
        seed = int(time.time()) % 10_000
        scene = os.environ.get("RANGREZ_SCENE") or scene_names()[seed % len(scene_names())]
        return VirtualRig(seed=seed, target_lab=target_lab, scene=scene)

    def make_segmenter(self, rig=None):
        if self.online and os.environ.get("RANGREZ_SEGMENTER", "sam") == "sam":
            return MuseSamSegmenter(self.log)
        if isinstance(rig, VirtualRig):
            return rig.segmenter()  # offline: the scene's cached SAM masks
        raise HTTPException(400, "reading a real photo needs hosted SAM 3.1: set MODEL_API_KEY")

    def publish(self, event: dict) -> None:
        event = _json_safe(event)
        for q in list(self.subscribers):
            q.put(event)

    def start(self, target_lab: np.ndarray, tolerance: float) -> None:
        if self.thread and self.thread.is_alive():
            raise HTTPException(409, "a campaign is already running")
        rig = self.make_rig(None)
        c = Campaign(target_lab, rig, self.make_segmenter(rig), tolerance=tolerance)
        c.listeners.append(self.publish)
        self.campaign = c
        self.publish({"kind": "start", "target_lab": target_lab.tolist(), "target_hex": lab_to_hex(target_lab),
                      "tolerance": tolerance, "agent": WONDER_MODEL if self.online else "offline policy",
                      "rig": type(rig).__name__, "at": time.time()})
        delay = float(os.environ.get("RANGREZ_STEP_DELAY", "1.5"))

        def work() -> None:
            from .assisted import Cancelled

            try:
                if self.online:
                    run_with_wonder(c, self.log)
                else:
                    run_deterministic(c, on_step=lambda: time.sleep(delay))
            except Cancelled as exc:
                c.status = "stopped"
                self.publish({"kind": "status", "status": "stopped", "reason": str(exc), "at": time.time()})
            except Exception as exc:  # surface failures on the dashboard, never silently
                c.status = "error"
                self.publish({"kind": "status", "status": "error", "reason": str(exc), "at": time.time()})

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()


def _json_safe(value):
    """Browsers reject NaN in JSON; a rejected measurement becomes null."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


LAB = Lab()


class StartBody(BaseModel):
    hex: str | None = None
    lab: list[float] | None = None
    tolerance: float = 1.5


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {"online": LAB.online, "agent": WONDER_MODEL if LAB.online else "offline policy",
            "rig": os.environ.get("RANGREZ_RIG", "sim"), "dyes": DYES, "unit": UNIT}


@app.post("/api/campaign")
def start(body: StartBody) -> dict:
    if body.lab:
        target = np.array(body.lab, dtype=float)
    elif body.hex:
        target = hex_to_lab(body.hex)
    else:
        raise HTTPException(400, "give a hex colour or a Lab triple")
    try:
        LAB.start(target, body.tolerance)
    except HTTPException:
        raise
    return {"ok": True, "target_hex": lab_to_hex(target)}


@app.post("/api/target-from-photo")
async def target_photo(photo: UploadFile = File(...), concept: str = Form("fabric")) -> dict:
    jpeg = await photo.read()
    try:
        from .assisted import normalise_photo

        lab, calibration = target_from_photo(normalise_photo(jpeg), concept, LAB.make_segmenter())
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc))
    return {"lab": [round(v, 2) for v in lab], "hex": lab_to_hex(lab), "calibration": calibration}


@app.post("/api/assist/photo")
async def assist_photo(photo: UploadFile = File(...)) -> dict:
    rig = getattr(LAB.campaign, "rig", None)
    if rig is None or not hasattr(rig, "provide"):
        raise HTTPException(409, "no assisted campaign is waiting for a photo")
    try:
        rig.provide(await photo.read())
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@app.post("/api/assist/webcam")
def assist_webcam() -> dict:
    rig = getattr(LAB.campaign, "rig", None)
    if rig is None or not hasattr(rig, "provide"):
        raise HTTPException(409, "no assisted campaign is waiting for a photo")
    try:
        rig.provide(rig.capture())
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@app.post("/api/stop")
def stop() -> dict:
    c = LAB.campaign
    if c is None:
        return {"ok": True}
    if hasattr(c.rig, "cancel"):
        c.rig.cancel()
    if c.status == "running":
        c.status = "stopped"
        LAB.publish({"kind": "status", "status": "stopped", "reason": "stopped from the dashboard", "at": time.time()})
    return {"ok": True}


@app.get("/api/state")
def state() -> dict:
    return LAB.campaign.state() if LAB.campaign else {"status": "idle"}


@app.get("/api/shot/{index}.jpg")
def shot_image(index: int, raw: bool = False) -> Response:
    c = LAB.campaign
    if not c or not 1 <= index <= len(c.shots):
        raise HTTPException(404)
    s = c.shots[index - 1]
    return Response(s.image_jpeg if raw else s.overlay_jpeg, media_type="image/jpeg")


@app.get("/api/events")
async def events() -> StreamingResponse:
    q: queue.Queue = queue.Queue()
    LAB.subscribers.append(q)

    async def stream():
        try:
            yield "retry: 1500\n\n"
            while True:
                try:
                    event = await asyncio.to_thread(q.get, True, 15)
                    yield f"data: {json.dumps(event, default=float)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            LAB.subscribers.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("RANGREZ_HOST", "127.0.0.1"), port=int(os.environ.get("RANGREZ_PORT_HTTP", "8000")))
