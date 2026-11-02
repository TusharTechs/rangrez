"""Hands-free lab: the dyer's gloves are wet with dye, so they talk instead of tapping.

Muse Voice Transcribe (Meta Model API, /v1/asr/transcribe) turns a spoken turn into text,
biased towards the lab's words; a small deterministic parser turns the text into one action.
Nothing spoken can dispense dye or change a recipe: commands only take the photo, repeat the
instructions or stop, and everything else becomes a note in the lab notebook for Wonder to read.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import uuid

from .profiles import PROFILE

MODEL = "muse-voice-transcribe-1.0"
URL = "https://api.meta.ai/v1/asr/transcribe"
KEYWORDS = ["done", "dip", "photo", "webcam", "repeat", "stop", "drops", "patchy", "darker", "lighter", "Rangrez",
            *PROFILE.names]

INTENTS = [  # first match wins; whole words only
    ("stop", r"\b(stop|cancel|abort)\b"),
    ("repeat", r"\b(repeat|again|say that|what was|instructions)\b"),
    ("photo", r"\b(done|ready|photo|picture|capture|webcam|shoot|snap|dry now|it'?s dry)\b"),
]


def transcribe(wav: bytes, log=None) -> str:
    """Text of one spoken turn (a 16 kHz mono WAV)."""
    from .vision import ApiCall

    boundary = uuid.uuid4().hex
    request = json.dumps({"model": MODEL, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK", "keywords": KEYWORDS})
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n"
            f"{request}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"turn.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {os.environ['MODEL_API_KEY']}", "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = json.loads(r.read()).get("transcript", "").strip()
    except Exception as exc:
        if log is not None:
            log.add(ApiCall(MODEL, "voice command", time.monotonic() - started, False, str(exc)[:200]))
        raise RuntimeError(f"could not transcribe: {exc}") from exc
    if log is not None:
        log.add(ApiCall(MODEL, "voice command", time.monotonic() - started, True, text[:80]))
    return text


def intent(text: str) -> str:
    """One of stop, repeat, photo, or note (anything else is something the dyer observed)."""
    t = text.lower().strip(" .!?")
    if not t:
        return "none"
    for name, pattern in INTENTS:
        if re.search(pattern, t):
            # "it looks done but patchy" is an observation with a command in it: a long turn is a note.
            if name == "photo" and len(t.split()) > 6:
                return "note"
            return name
    return "note"
