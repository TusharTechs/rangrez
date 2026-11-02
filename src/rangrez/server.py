"""Live lab dashboard: FastAPI + server-sent events.

    uv run rangrez-lab            # simulator rig, hosted SAM + Wonder if MODEL_API_KEY is set
    RANGREZ_RIG=serial RANGREZ_PORT=/dev/cu.usbserial-0001 uv run rangrez-lab
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import secrets
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from .agent import WONDER_MODEL, run_deterministic, run_with_wonder
from .campaign import Campaign
from .colour import hex_to_lab, lab_to_hex
from .memory import LabMemory, default_path
from .photoreal import scene_names
from .recorder import Recorder
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
        self.history: list[dict] = []  # this campaign's events, replayed to pages that join late
        self.lock = threading.Lock()
        self.since = time.time()
        self.campaign: Campaign | None = None
        self.thread: threading.Thread | None = None
        self.rec = Recorder()
        self.target_photo: bytes | None = None  # the customer's photo the current target came from
        self.target_concept = "garment"
        self.preview: dict | None = None  # Muse Image's bulk preview, measured and corrected
        self.order: dict | None = None  # the buyer's order, read by Muse Spark
        self.log.listeners.append(lambda call: self.publish({
            "kind": "api", "model": call.model, "purpose": call.purpose,
            "seconds": round(call.seconds, 2), "ok": call.ok, "detail": call.detail, "at": time.time(),
        }))

    @property
    def key_available(self) -> bool:
        return bool(os.environ.get("MODEL_API_KEY")) and os.environ.get("RANGREZ_OFFLINE") != "1"

    @property
    def online(self) -> bool:
        """Meta models by default; on a public demo (RANGREZ_LIVE_PASSCODE set) only with the passcode."""
        return self.key_available and not os.environ.get("RANGREZ_LIVE_PASSCODE")

    def live(self, passcode: str | None) -> bool:
        if self.online:
            return True
        code = os.environ.get("RANGREZ_LIVE_PASSCODE", "")
        return self.key_available and bool(code) and secrets.compare_digest((passcode or "").encode(), code.encode())

    def make_rig(self, target_lab: np.ndarray | None, live: bool = True):
        kind = os.environ.get("RANGREZ_RIG", "sim")
        if kind == "assisted":
            if not live:
                raise HTTPException(400, "assisted mode measures real photos with hosted SAM 3.1: set MODEL_API_KEY")
            from .assisted import AssistedRig

            return AssistedRig(
                on_instructions=lambda w: self.publish({"kind": "assist", **w, "at": time.time()}),
            )
        if kind == "serial":
            from .serial_rig import SerialRig

            port = os.environ["RANGREZ_PORT"]
            if port == "virtual":  # the firmware's protocol and limits, emulated: no parts needed
                from .esp32_emulator import start

                board, port = start(time_scale=float(os.environ.get("RANGREZ_TIME_SCALE", "0.02")))
                return SerialRig(port, camera=board.camera_path, boot_seconds=0.1)
            return SerialRig(port)
        # A different realistic scene and paper each campaign; the colour card in view.
        seed = int(time.time()) % 10_000
        scene = os.environ.get("RANGREZ_SCENE") or scene_names()[seed % len(scene_names())]
        return VirtualRig(seed=seed, target_lab=target_lab, scene=scene)

    def make_segmenter(self, rig=None, live: bool = True):
        if live and os.environ.get("RANGREZ_SEGMENTER", "sam") == "sam":
            return MuseSamSegmenter(self.log)
        if isinstance(rig, VirtualRig):
            return rig.segmenter()  # offline: the scene's cached SAM masks
        raise HTTPException(400, "reading a real photo needs hosted SAM 3.1: set MODEL_API_KEY, or enter the live passcode")

    def publish(self, event: dict) -> None:
        event = _json_safe(event)
        with self.lock:
            self.history.append(event)
            for q in list(self.subscribers):
                q.put(event)
        if self.campaign is None:
            return
        self.rec.event(event)
        if event.get("kind") == "shot" and self.campaign.shots:
            s = self.campaign.shots[-1]
            self.rec.save(f"dip{s.index:02d}_photo.jpg", s.image_jpeg)
            self.rec.save(f"dip{s.index:02d}_sam.jpg", s.overlay_jpeg)
        if event.get("kind") == "status" and event.get("status") not in ("running", "waiting"):
            self.rec.save("state.json", self.campaign.state())
            self.rec.save("report.html", self.report())

    def report(self) -> str:
        from .report import render

        c = self.campaign
        calls = sum(1 for e in self.history if e.get("kind") == "api")
        agent = next((e.get("agent") for e in self.history if e.get("kind") == "start"), "")
        return render(c, target_photo=self.target_photo, agent=agent, api_calls=calls, preview=self.preview,
                      order=self.order)

    def subscribe(self) -> queue.Queue:
        """A new page (a reload, the phone joining) first gets the campaign so far."""
        q: queue.Queue = queue.Queue()
        with self.lock:
            q.put({"kind": "reset"})
            for event in self.history:
                q.put(event)
            q.put({"kind": "replayed"})  # live events from here on (the page may act on them, e.g. speak)
            self.subscribers.append(q)
        return q

    def start(self, target_lab: np.ndarray, tolerance: float, live: bool | None = None) -> None:
        live = self.online if live is None else live
        if self.thread and self.thread.is_alive():
            raise HTTPException(409, "a campaign is already running")
        with self.lock:  # keep only the API calls made since the last start (e.g. reading the target photo)
            self.history = [e for e in self.history if e.get("kind") == "api" and e.get("at", 0) >= self.since]
            self.since = time.time()
        self.preview = None  # a new campaign, a new best dip
        rig = self.make_rig(None, live)
        path = default_path()
        memory = LabMemory(path) if path is not None else None  # re-read: another process may have added dips
        c = Campaign(target_lab, rig, self.make_segmenter(rig, live), tolerance=tolerance, memory=memory)
        self.rec.begin(type(rig).__name__, {
            "rig": type(rig).__name__, "scene": getattr(rig, "scene", None), "dyes": DYES, "unit": UNIT,
            "agent": WONDER_MODEL if live else "offline policy", "tolerance": tolerance,
            "target_lab": target_lab.tolist(), "target_hex": lab_to_hex(target_lab),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        # The API calls that read the target photo belong to this run too.
        for e in self.history:
            self.rec.event(e)
        c.listeners.append(self.publish)
        self.campaign = c
        self.publish({"kind": "start", "target_lab": target_lab.tolist(), "target_hex": lab_to_hex(target_lab),
                      "tolerance": tolerance, "agent": WONDER_MODEL if live else "offline policy",
                      "rig": type(rig).__name__, "memory_dips": len(memory.dips) if memory else 0,
                      "memory_campaigns": len(memory.campaigns) if memory else 0, "at": time.time()})
        delay = float(os.environ.get("RANGREZ_STEP_DELAY", "1.5"))

        def work() -> None:
            from .assisted import Cancelled

            try:
                if live:
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


class PublicQuota:
    """Free colour reads on a public demo: a few per visitor and a cap per day, so it can't run up a bill."""

    def __init__(self) -> None:
        self.per_day = int(os.environ.get("RANGREZ_PUBLIC_READS_PER_DAY", "0"))
        self.per_visitor = int(os.environ.get("RANGREZ_PUBLIC_READS_PER_VISITOR", "5"))
        self.day, self.total, self.by_ip = "", 0, {}
        self.lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        with self.lock:
            today = time.strftime("%Y-%m-%d")
            if today != self.day:
                self.day, self.total, self.by_ip = today, 0, {}
            if self.total >= self.per_day or self.by_ip.get(ip, 0) >= self.per_visitor:
                return False
            self.total += 1
            self.by_ip[ip] = self.by_ip.get(ip, 0) + 1
            return True

    def left(self) -> int:
        with self.lock:
            return max(0, self.per_day - (self.total if self.day == time.strftime("%Y-%m-%d") else 0))


QUOTA = PublicQuota()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


class StartBody(BaseModel):
    hex: str | None = None
    lab: list[float] | None = None
    tolerance: float = 1.5
    passcode: str | None = None  # unlocks the Meta models on a public demo


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {"online": LAB.online, "agent": WONDER_MODEL if LAB.online else "offline policy",
            "passcode": bool(os.environ.get("RANGREZ_LIVE_PASSCODE")) and LAB.key_available,
            "public_reads_left": QUOTA.left() if LAB.key_available else 0,
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
        LAB.start(target, body.tolerance, LAB.live(body.passcode))
    except HTTPException:
        raise
    return {"ok": True, "target_hex": lab_to_hex(target)}


@app.post("/api/target-from-photo")
async def target_photo(request: Request, photo: UploadFile = File(...), concept: str = Form("fabric"),
                       passcode: str = Form("")) -> dict:
    from .home import home_recipe

    jpeg = await photo.read()
    live = LAB.live(passcode) or (LAB.key_available and QUOTA.allow(client_ip(request)))
    if not live and LAB.key_available:
        raise HTTPException(429, "today's free colour reads are used up: try again tomorrow, or enter the live passcode")
    try:
        from .assisted import normalise_photo

        photo = normalise_photo(jpeg)
        lab, calibration = await asyncio.to_thread(target_from_photo, photo, concept, LAB.make_segmenter(live=live))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc))
    result = {"lab": [round(v, 2) for v in lab], "hex": lab_to_hex(lab), "calibration": calibration, "concept": concept,
              "home_recipe": home_recipe(lab)}
    LAB.rec.pending = []  # only the latest target photo goes with the next campaign
    LAB.rec.stash("target.jpg", photo)
    LAB.target_photo = photo
    LAB.target_concept = concept
    LAB.preview = None
    LAB.rec.stash("target.json", result)
    return result


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


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...), passcode: str = Form(""), lang: str = Form("en")) -> dict:
    """One spoken turn from the hands-free dashboard: transcribe it, then do at most one safe thing."""
    from .voice import SAY, instructions, intent, transcribe

    if not LAB.live(passcode):
        raise HTTPException(403, "voice needs the Meta models: set MODEL_API_KEY, or enter the live passcode")
    lang = lang if lang in SAY else "en"
    wav = await audio.read()
    try:
        text = await asyncio.to_thread(transcribe, wav, LAB.log, lang)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    act, say = intent(text), SAY[lang]
    c, result = LAB.campaign, ""
    rig = getattr(c, "rig", None)
    waiting = getattr(rig, "waiting", None)
    if act == "photo":
        if waiting and hasattr(rig, "capture"):
            try:
                rig.provide(await asyncio.to_thread(rig.capture))
                result = say["photo"]
            except (RuntimeError, ValueError):
                result = say["photo_failed"]
        else:
            result = say["no_photo"]
    elif act == "stop":
        stop()
        result = say["stopped"]
    elif act == "repeat":
        result = instructions(waiting, lang) if waiting else say["no_dip"]
    elif act == "note" and c is not None:
        c.note(text, who="dyer")
        result = say["noted"]
    return {"transcript": text, "intent": act, "result": result}


@app.post("/api/order")
async def read_order(message: str = Form(...), passcode: str = Form("")) -> dict:
    """The buyer's chat message, read by Muse Spark: what, how much, by when, how strict the shade."""
    from .order import parse_order

    if not LAB.live(passcode):
        raise HTTPException(403, "reading an order needs the Meta models: set MODEL_API_KEY, or enter the live passcode")
    try:
        LAB.order = await asyncio.to_thread(parse_order, message[:2000], LAB.log)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    LAB.rec.stash("order.json", {"message": message[:2000], "order": LAB.order})
    return LAB.order


@app.post("/api/preview")
async def preview(passcode: str = Form("")) -> dict:
    """The customer's garment in the best dip's shade: Muse Image recolours, SAM 3.1 checks and corrects."""
    from .preview import bulk_preview

    c = LAB.campaign
    if c is None or c.best is None or LAB.target_photo is None:
        raise HTTPException(409, "needs a campaign started from a customer's photo, with at least one dip")
    if not LAB.live(passcode):
        raise HTTPException(403, "the preview needs the Meta models: set MODEL_API_KEY, or enter the live passcode")
    try:
        p = await asyncio.to_thread(bulk_preview, LAB.target_photo, LAB.target_concept, c.best.measurement.lab, LAB.log)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    LAB.preview = {**p, "dip": c.best.index}
    LAB.rec.save("preview.jpg", p["jpeg"])
    LAB.rec.save("report.html", LAB.report())
    return {"dip": c.best.index, "measured_hex": p["measured_hex"], "raw_de00": p["raw_de00"], "de00": p["de00"]}


@app.get("/api/preview.jpg")
def preview_image() -> Response:
    if not LAB.preview:
        raise HTTPException(404)
    return Response(LAB.preview["jpeg"], media_type="image/jpeg")


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


@app.get("/api/report.html")
def report() -> HTMLResponse:
    if LAB.campaign is None or not LAB.campaign.shots:
        raise HTTPException(404, "no dips yet")
    return HTMLResponse(LAB.report())


@app.get("/api/memory")
def memory() -> dict:
    """How the lab has done with this dye lot, campaign by campaign (the night shift, live)."""
    path = default_path()
    m = LabMemory(path) if path is not None else None
    if m is None:
        return {"dips": 0, "campaigns": []}
    keep = ("status", "dips", "first_de00", "best_de00")
    return {"dips": len(m.dips), "campaigns": [{k: c.get(k) for k in keep} for c in m.campaigns[-60:]]}


@app.get("/api/shot/{index}.jpg")
def shot_image(index: int, raw: bool = False) -> Response:
    c = LAB.campaign
    if not c or not 1 <= index <= len(c.shots):
        raise HTTPException(404)
    s = c.shots[index - 1]
    return Response(s.image_jpeg if raw else s.overlay_jpeg, media_type="image/jpeg")


@app.get("/api/events")
async def events() -> StreamingResponse:
    q = LAB.subscribe()

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
