"""Glimmer: Muse Glimmer 30B fine-tuned on this lab's own dips, proposing the first recipe.

The supplier's model knows the dyes in general; Glimmer learned this dye lot through this
camera, from every recipe the lab dipped and the colour SAM 3.1 measured on the cloth
(eval/glimmer.py). The open LoRA weights run anywhere; on Fireworks set

    GLIMMER_MODEL=accounts/<account>/models/<model>#accounts/<account>/deployments/<deployment>

or serve the published weights on Modal (eval/glimmer_serve.py) and set GLIMMER_BASE_URL and
GLIMMER_API_KEY. Either way the first dip of every campaign gets a "glimmer" candidate.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .colour import lab_to_hex
from .profiles import PROFILE

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
SYSTEM = (
    f"You are Glimmer, the colourist of a small cotton dye lab. Dyes: {', '.join(PROFILE.names)} "
    f"({PROFILE.description}). Given a target colour measured in CIELAB (D65), reply with only the "
    f"recipe for the first dip as JSON with one key per dye, in {PROFILE.unit}, each between 0 and {PROFILE.max_conc:g}."
)


def prompt(lab) -> str:
    L, a, b = (float(v) for v in lab)
    return f"Target L* {L:.1f}, a* {a:.1f}, b* {b:.1f} ({lab_to_hex(np.array([L, a, b]))}). First dip recipe?"


def answer(conc) -> str:
    return json.dumps({d: round(float(c), 2) for d, c in zip(PROFILE.names, conc)})


def parse(text: str) -> np.ndarray | None:
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return np.clip([float(obj.get(d, 0.0)) for d in PROFILE.names], 0, PROFILE.max_conc)
    except (ValueError, json.JSONDecodeError, TypeError, AttributeError):
        return None


def _endpoint() -> tuple[str, str]:
    """Where Glimmer runs: Fireworks by default, or any OpenAI-style endpoint (eval/glimmer_serve.py on Modal)."""
    url = os.environ.get("GLIMMER_BASE_URL") or FIREWORKS_URL
    key = os.environ.get("GLIMMER_API_KEY") if os.environ.get("GLIMMER_BASE_URL") else os.environ.get("FIREWORKS_API_KEY")
    return url, (key or "").strip()


def configured() -> bool:
    return bool(os.environ.get("GLIMMER_MODEL") and _endpoint()[1])


def first_guess(target_lab, log=None, model: str | None = None) -> np.ndarray | None:
    """Glimmer's first recipe for target_lab, or None if it is not configured or answers badly."""
    from openai import OpenAI

    from .vision import ApiCall

    model = model or os.environ.get("GLIMMER_MODEL")
    if not model:
        return None
    url, key = _endpoint()
    client = OpenAI(base_url=url, api_key=key, timeout=120, max_retries=1)
    started = time.monotonic()
    try:
        r = client.chat.completions.create(model=model, temperature=0, max_tokens=400, messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(target_lab)}])
        conc = parse(r.choices[0].message.content or "")
    except Exception as exc:  # the lab works without Glimmer: never fail a campaign over it
        if log is not None:
            log.add(ApiCall("muse-glimmer-30b (tuned)", "first dip recipe", time.monotonic() - started, False, str(exc)[:200]))
        return None
    if log is not None:
        log.add(ApiCall("muse-glimmer-30b (tuned)", "first dip recipe", time.monotonic() - started, conc is not None))
    return conc
