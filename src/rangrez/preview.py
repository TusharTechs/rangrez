"""Bulk preview: the customer's garment shown in the shade the lab actually dyed.

Muse Image recolours the garment in the customer's photo (keeping its folds, buttons and light).
Image models are not colour instruments, so Rangrez then measures the preview the same way it
measures a dip (SAM 3.1 + the paper or card), nudges the garment's pixels so the measurement
matches the dip, and reports the residual. The preview is for the eye; the dip is the evidence.
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
from PIL import Image

from .colour import (
    delta_e_2000,
    lab_to_hex,
    lab_to_srgb,
    linear_to_srgb,
    srgb_to_linear,
)
from .vision import ApiCall, MuseSamSegmenter, decode, read_object, target_from_photo

IMAGE_MODEL = "muse-image-1.0"


def _edit(photo: bytes, concept: str, hex_colour: str, log) -> bytes:
    from openai import OpenAI

    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=240, max_retries=1)
    started = time.monotonic()
    try:
        r = client.images.edit(model=IMAGE_MODEL, image=("garment.jpg", photo, "image/jpeg"), prompt=(
            f"Recolour only the {concept} to a solid {hex_colour}, matte dyed cotton. Keep its folds, seams, buttons, "
            "shading and everything else in the photo exactly as it is."))
    except Exception as exc:
        log.add(ApiCall(IMAGE_MODEL, "bulk preview", time.monotonic() - started, False, str(exc)[:200]))
        raise RuntimeError(f"Muse Image could not make the preview: {exc}") from exc
    log.add(ApiCall(IMAGE_MODEL, "bulk preview", time.monotonic() - started, True))
    img = Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def bulk_preview(photo: bytes, concept: str, dip_lab, log, segmenter=None) -> dict:
    """{"jpeg", "measured_hex", "raw_de00", "de00"}: the corrected preview and how close it is to the dip."""
    dip_lab = np.asarray(dip_lab, dtype=float)
    edited = _edit(photo, concept, lab_to_hex(dip_lab), log)
    seg = segmenter or MuseSamSegmenter(log)
    raw_lab, _, mask = read_object(edited, concept, seg)  # the same garment the measurement chose
    raw_de = float(delta_e_2000(raw_lab, dip_lab))

    # Nudge the garment's pixels, per channel in linear light, so it measures as the dip.
    _, srgb = decode(edited)
    lin = srgb_to_linear(srgb)
    gain = srgb_to_linear(lab_to_srgb(dip_lab)) / np.maximum(srgb_to_linear(lab_to_srgb(raw_lab)), 1e-4)
    lin[mask] = np.clip(lin[mask] * np.clip(gain, 0.25, 4.0), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((linear_to_srgb(lin) * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=92)
    corrected = buf.getvalue()
    final_lab, _ = target_from_photo(corrected, concept, seg)
    return {"jpeg": corrected, "measured_hex": lab_to_hex(final_lab), "raw_de00": round(raw_de, 2),
            "de00": round(float(delta_e_2000(final_lab, dip_lab)), 2)}
