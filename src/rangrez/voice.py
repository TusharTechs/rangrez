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
KEYWORDS = {
    "en": ["done", "dip", "photo", "webcam", "repeat", "stop", "drops", "patchy", "darker", "lighter", "Rangrez", *PROFILE.names],
    "hi": [],  # tried: Devanagari keywords made Hindi worse ("फोटो लो" heard as "फट लो"); language bias alone is clean
}

INTENTS = [  # first match wins. English as whole words; Hindi (Devanagari or Hinglish) as phrases.
    ("stop", r"\b(stop|cancel|abort|band karo|ruk jao)\b|रुको|रुक जाओ|बंद करो|^बस\b"),
    ("repeat", r"\b(repeat|again|say that|what was|instructions|phir se|dobara)\b|फिर से|दोबारा|दुबारा|क्या था"),
    ("photo", r"\b(done|ready|photo|picture|capture|webcam|shoot|snap|dry now|it'?s dry|ho gaya|ho gya)\b"
              r"|हो गया|हो गई|फोटो|फ़ोटो|तस्वीर|खींचो|सूख गया"),
]
LANGUAGES = {"en": ["English"], "hi": ["Hindi", "English"]}


def transcribe(wav: bytes, log=None, lang: str = "en") -> str:
    """Text of one spoken turn (a 16 kHz mono WAV), biased towards the dyer's language."""
    from .vision import ApiCall

    boundary = uuid.uuid4().hex
    lang = lang if lang in LANGUAGES else "en"
    settings = {"model": MODEL, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK", "languageBias": LANGUAGES[lang]}
    if KEYWORDS[lang]:
        settings["keywords"] = KEYWORDS[lang]
    request = json.dumps(settings)
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
            if name == "photo" and len(t.split()) > 5:
                return "note"
            return name
    return "note"


# What the lab says back, in the dyer's language.
SAY = {
    "en": {"photo": "Photo taken. Measuring.", "no_photo": "Nothing to photograph right now.",
           "stopped": "Stopped.", "no_dip": "No dip is waiting.", "noted": "Noted for the colourist.",
           "photo_failed": "Could not take the photo. Send one from your phone."},
    "hi": {"photo": "फोटो ले ली। नाप रहे हैं।", "no_photo": "अभी फोटो लेने को कुछ नहीं है।",
           "stopped": "रोक दिया।", "no_dip": "अभी कोई डिप बाकी नहीं है।", "noted": "रंगसाज़ के लिए लिख लिया।",
           "photo_failed": "फोटो नहीं ले पाए। फ़ोन से फोटो भेजिए।"},
}
DYE_HI = {"blue": "नीला", "red": "लाल", "yellow": "पीला", "turquoise": "फ़िरोज़ी", "fuchsia": "रानी", "lemon yellow": "नींबू पीला"}
UNIT_HI = {"drops": "बूँद", "mL": "एमएल"}


def instructions(w: dict, lang: str = "en") -> str:
    """One dip's instructions as a sentence to read aloud."""
    amounts = [(d, v) for d, v in w["amounts"].items() if float(v) > 0]
    if lang == "hi":
        parts = ", ".join(f"{DYE_HI.get(d, d)} {v} {UNIT_HI.get(w['unit'], w['unit'])}" for d, v in amounts)
        return f"डिप {w['index']}। {parts}। {w['dip_seconds']} सेकंड डुबोइए। कपड़ा सूखकर कागज़ पर रखें, तो 'हो गया' बोलिए।"
    parts = ", ".join(f"{v} {w['unit']} of {d}" for d, v in amounts)
    return f"Dip {w['index']}. {parts}. Dip for {w['dip_seconds']} seconds. Say done when the strip is dry and on the paper."
