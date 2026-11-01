"""Camera frame -> calibrated CIELAB of the dyed strip, with an error bar.

SAM 3.1 finds the colour card and the cotton strip by text prompt. Everything
after the masks is deterministic: the card's 24 patches fit a colour
correction, glare and shadow pixels are dropped, and the strip's colour is
reported in CIELAB with a bootstrap uncertainty.
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

from .card import COLS, ROWS, reference_linear
from .colour import (
    apply_colour_correction,
    delta_e_2000,
    linear_rgb_to_xyz,
    srgb_to_linear,
    xyz_to_lab,
)

SAM_MODEL = "sam-3.1"
BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")

CARD_PROMPT = "color checker card"
STRIP_PROMPT = "fabric strip"
TARGET_PROMPT = "fabric swatch"


@dataclass
class Instance:
    mask: np.ndarray  # bool (H, W)
    box: tuple[int, int, int, int]  # left, top, right, bottom

    @property
    def aspect(self) -> float:
        l, t, r, b = self.box
        return (b - t) / max(1, r - l)

    @property
    def centre(self) -> tuple[float, float]:
        l, t, r, b = self.box
        return ((l + r) / 2, (t + b) / 2)


class Segmenter(Protocol):
    def segment(self, image: Image.Image, jpeg: bytes, concept: str) -> list[Instance]: ...


@dataclass
class ApiCall:
    model: str
    purpose: str
    seconds: float
    ok: bool
    detail: str = ""


@dataclass
class CallLog:
    calls: list[ApiCall] = field(default_factory=list)
    listeners: list = field(default_factory=list)

    def add(self, call: ApiCall) -> None:
        self.calls.append(call)
        for fn in self.listeners:
            fn(call)


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    l, t, r, b = box
    m[t:b, l:r] = True
    return m


class OfflineSegmenter:
    """Stand-in for SAM when there is no API key: knows the rig's fixed geometry."""

    def __init__(self, boxes: dict[str, tuple[int, int, int, int]]):
        self.boxes = boxes

    def segment(self, image: Image.Image, jpeg: bytes, concept: str) -> list[Instance]:
        key = {CARD_PROMPT: "card", STRIP_PROMPT: "swatch", TARGET_PROMPT: "target"}[concept]
        shape = (image.height, image.width)
        out = []
        for name in ([key] if key != "swatch" else ["swatch", "target"]):
            box = self.boxes.get(name)
            if box:
                out.append(Instance(_box_mask(shape, box), box))
        return out


class MuseSamSegmenter:
    """Hosted SAM 3.1 on the Meta Model API (Responses endpoint)."""

    def __init__(self, log: CallLog | None = None, timeout: float = 20.0):
        from openai import OpenAI

        key = os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError("Set MODEL_API_KEY to use hosted SAM 3.1")
        self.client = OpenAI(base_url=BASE_URL, api_key=key, timeout=timeout, max_retries=1)
        self.log = log or CallLog()

    def segment(self, image: Image.Image, jpeg: bytes, concept: str) -> list[Instance]:
        from meta_sam_parser import (
            CompletedOutcome,
            SegmentationMaskRecord,
            decode_mask_to_raster,
            image_segmentation_format,
        )

        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        started = time.monotonic()
        try:
            response = self.client.responses.create(
                model=SAM_MODEL,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": concept},
                            {"type": "input_image", "image_url": url},
                        ],
                    }
                ],
                metadata={"mask_encoding": "one_bit"},
                store=False,
            )
        except Exception as exc:
            self.log.add(ApiCall(SAM_MODEL, f"segment '{concept}'", time.monotonic() - started, False, str(exc)))
            raise
        self.log.add(ApiCall(SAM_MODEL, f"segment '{concept}'", time.monotonic() - started, True))

        text = getattr(response, "output_text", "") or ""
        if not text:
            return []
        parser = image_segmentation_format().create_parser()
        parser.push(text)
        result = parser.finish(CompletedOutcome()).result
        width, height = image.width, image.height
        instances = []
        for record in result.records:
            if not isinstance(record, SegmentationMaskRecord):
                continue
            raster = decode_mask_to_raster(record.mask)
            piece = Image.frombytes("L", (record.mask.width, record.mask.height), bytes(raster))
            left = max(0, math.floor(record.bounds.left))
            top = max(0, math.floor(record.bounds.top))
            right = min(width, math.ceil(record.bounds.right))
            bottom = min(height, math.ceil(record.bounds.bottom))
            if right <= left or bottom <= top:
                continue
            if piece.size != (right - left, bottom - top):
                piece = piece.resize((right - left, bottom - top), Image.Resampling.NEAREST)
            full = np.zeros((height, width), dtype=bool)
            full[top:bottom, left:right] = np.asarray(piece) > 0
            instances.append(Instance(full, (left, top, right, bottom)))
        return instances


@dataclass
class Measurement:
    lab: np.ndarray
    lab_sd: np.ndarray
    card_residual_de: float  # mean CIEDE2000 of corrected patches vs reference
    n_pixels: int
    flags: list[str]
    card_box: tuple[int, int, int, int]
    strip_box: tuple[int, int, int, int]
    target_lab: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return not any(f.startswith("REJECT") for f in self.flags)


def decode(jpeg: bytes) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    return img, np.asarray(img, dtype=float) / 255.0


def _patch_samples(lin: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Median linear RGB of the central 40% of each of the 24 card cells."""
    l, t, r, b = box
    cw, ch = (r - l) / COLS, (b - t) / ROWS
    out = []
    for row in range(ROWS):
        for col in range(COLS):
            x0 = int(l + (col + 0.3) * cw)
            x1 = int(l + (col + 0.7) * cw)
            y0 = int(t + (row + 0.3) * ch)
            y1 = int(t + (row + 0.7) * ch)
            out.append(np.median(lin[y0:y1, x0:x1].reshape(-1, 3), axis=0))
    return np.array(out)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(px):
        m[1:, :] &= mask[:-1, :]
        m[:-1, :] &= mask[1:, :]
        m[:, 1:] &= mask[:, :-1]
        m[:, :-1] &= mask[:, 1:]
        mask = m.copy()
    return m


def _card_inner_box(inst: Instance) -> tuple[int, int, int, int]:
    """The patch grid sits inside the card's black border (12 px at 64 px patches)."""
    l, t, r, b = inst.box
    pad_x = (r - l) * 12 / (COLS * 72 + 16)
    pad_y = (b - t) * 12 / (ROWS * 72 + 16)
    return (int(l + pad_x), int(t + pad_y), int(r - pad_x + 8 * (r - l) / (COLS * 72 + 16)), int(b - pad_y + 8 * (b - t) / (ROWS * 72 + 16)))


def _lab_of_region(lin: np.ndarray, mask: np.ndarray, M: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    flags = []
    core = _erode(mask, 4)
    px = lin[core]
    if len(px) < 200:
        return np.full(3, np.nan), np.full(3, np.nan), len(px), ["REJECT: strip mask too small"]
    glare = (px.max(axis=1) > 0.96)
    if glare.mean() > 0.15:
        flags.append(f"WARN: {glare.mean():.0%} glare pixels removed")
    px = px[~glare]
    lum = px.mean(axis=1)
    lo, hi = np.percentile(lum, [5, 95])
    px = px[(lum >= lo) & (lum <= hi)]
    corrected = np.clip(apply_colour_correction(M, px), 0, 1)
    lab = xyz_to_lab(linear_rgb_to_xyz(corrected.mean(axis=0)))
    boots = []
    for _ in range(64):
        idx = rng.integers(0, len(corrected), size=min(len(corrected), 2000))
        boots.append(xyz_to_lab(linear_rgb_to_xyz(corrected[idx].mean(axis=0))))
    return lab, np.std(boots, axis=0), len(px), flags


def measure(jpeg: bytes, segmenter: Segmenter, with_target: bool = False, seed: int = 0) -> Measurement:
    img, srgb = decode(jpeg)
    lin = srgb_to_linear(srgb)
    rng = np.random.default_rng(seed)
    flags: list[str] = []

    cards = segmenter.segment(img, jpeg, CARD_PROMPT)
    if not cards:
        raise RuntimeError("colour card not found in frame")
    card = max(cards, key=lambda i: i.mask.sum())
    inner = _card_inner_box(card)
    patches = _patch_samples(lin, inner)
    ref = reference_linear()
    from .colour import fit_colour_correction

    M = fit_colour_correction(patches, ref)
    corrected = np.clip(apply_colour_correction(M, patches), 0, 1)
    residual = float(np.mean(delta_e_2000(xyz_to_lab(linear_rgb_to_xyz(corrected)), xyz_to_lab(linear_rgb_to_xyz(ref)))))
    if residual > 4:
        flags.append(f"REJECT: card fit residual {residual:.1f} dE00")
    elif residual > 2:
        flags.append(f"WARN: card fit residual {residual:.1f} dE00")

    fabrics = segmenter.segment(img, jpeg, STRIP_PROMPT)
    fabrics = [f for f in fabrics if not (f.mask & card.mask).any()]
    if not fabrics:
        raise RuntimeError("fabric strip not found in frame")
    # The strip hangs vertically from the dip arm: the tallest, narrowest fabric.
    strip = max(fabrics, key=lambda i: i.aspect)
    lab, lab_sd, n, f2 = _lab_of_region(lin, strip.mask, M, rng)
    flags += f2

    target_lab = None
    if with_target:
        others = [f for f in fabrics if f is not strip]
        if not others:
            targets = segmenter.segment(img, jpeg, TARGET_PROMPT)
            others = [t for t in targets if not (t.mask & strip.mask).any() and not (t.mask & card.mask).any()]
        if others:
            target = max(others, key=lambda i: i.mask.sum())
            target_lab, _, _, _ = _lab_of_region(lin, target.mask, M, rng)
    return Measurement(lab, lab_sd, residual, n, flags, card.box, strip.box, target_lab)


def target_from_photo(jpeg: bytes, concept: str, segmenter: Segmenter) -> tuple[np.ndarray, bool]:
    """Lab of the named object in a customer's photo (e.g. "teal kurta").

    If the photo also shows the colour card, the result is calibrated;
    otherwise it is the photo's own sRGB, which is only as good as their camera.
    Returns (lab, calibrated).
    """
    img, srgb = decode(jpeg)
    lin = srgb_to_linear(srgb)
    rng = np.random.default_rng(0)
    objects = segmenter.segment(img, jpeg, concept)
    if not objects:
        raise RuntimeError(f"could not find '{concept}' in the photo")
    obj = max(objects, key=lambda i: i.mask.sum())
    M = np.vstack([np.eye(3), np.zeros((1, 3))])
    calibrated = False
    try:
        cards = segmenter.segment(img, jpeg, CARD_PROMPT)
    except Exception:
        cards = []
    if cards:
        from .colour import fit_colour_correction

        card = max(cards, key=lambda i: i.mask.sum())
        M = fit_colour_correction(_patch_samples(lin, _card_inner_box(card)), reference_linear())
        calibrated = True
    lab, _, _, _ = _lab_of_region(lin, obj.mask, M, rng)
    return lab, calibrated


def overlay(jpeg: bytes, m: Measurement) -> bytes:
    """Frame with the card grid and strip box drawn on it, for the dashboard."""
    from PIL import ImageDraw

    img, _ = decode(jpeg)
    draw = ImageDraw.Draw(img)
    draw.rectangle(m.card_box, outline=(255, 214, 0), width=3)
    draw.rectangle(m.strip_box, outline=(0, 229, 255), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
