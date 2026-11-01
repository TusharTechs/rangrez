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
from .rig import DYES
from .sim import VirtualRig, ground_truth_masks
from .vision import CallLog, MuseSamSegmenter, OfflineSegmenter, target_from_photo

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
        if os.environ.get("RANGREZ_RIG", "sim") == "serial":
            from .serial_rig import SerialRig

            return SerialRig(os.environ["RANGREZ_PORT"])
        return VirtualRig(seed=int(time.time()) % 10_000, target_lab=target_lab)

    def make_segmenter(self):
        if self.online and os.environ.get("RANGREZ_SEGMENTER", "sam") == "sam":
            return MuseSamSegmenter(self.log)
        return OfflineSegmenter(ground_truth_masks())

    def publish(self, event: dict) -> None:
        for q in list(self.subscribers):
            q.put(event)

    def start(self, target_lab: np.ndarray, tolerance: float) -> None:
        if self.thread and self.thread.is_alive():
            raise HTTPException(409, "a campaign is already running")
        rig = self.make_rig(None)
        c = Campaign(target_lab, rig, self.make_segmenter(), tolerance=tolerance)
        c.listeners.append(self.publish)
        self.campaign = c
        self.publish({"kind": "start", "target_lab": target_lab.tolist(), "target_hex": lab_to_hex(target_lab),
                      "tolerance": tolerance, "agent": WONDER_MODEL if self.online else "offline policy",
                      "rig": type(rig).__name__, "at": time.time()})
        delay = float(os.environ.get("RANGREZ_STEP_DELAY", "1.5"))

        def work() -> None:
            try:
                if self.online:
                    run_with_wonder(c, self.log)
                else:
                    run_deterministic(c, on_step=lambda: time.sleep(delay))
            except Exception as exc:  # surface failures on the dashboard, never silently
                c.status = "error"
                self.publish({"kind": "status", "status": "error", "reason": str(exc), "at": time.time()})

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()


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
            "rig": os.environ.get("RANGREZ_RIG", "sim"), "dyes": DYES}


@app.post("/api/campaign")
def start(body: StartBody) -> dict:
    if body.lab:
        target = np.array(body.lab, dtype=float)
    elif body.hex:
        target = hex_to_lab(body.hex)
    else:
        raise HTTPException(400, "give a hex colour or a Lab triple")
    LAB.start(target, body.tolerance)
    return {"ok": True, "target_hex": lab_to_hex(target)}


@app.post("/api/target-from-photo")
async def target_photo(photo: UploadFile = File(...), concept: str = Form("fabric")) -> dict:
    jpeg = await photo.read()
    try:
        lab, calibrated = target_from_photo(jpeg, concept, LAB.make_segmenter())
    except RuntimeError as exc:
        raise HTTPException(422, str(exc))
    return {"lab": [round(v, 2) for v in lab], "hex": lab_to_hex(lab), "calibrated": calibrated}


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
