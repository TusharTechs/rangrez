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

import cv2
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
STRIP_PROMPTS = ("fabric", "cloth", STRIP_PROMPT, "cotton tape", "rectangle")
PAPER_PROMPT = "white paper"
PAPER_PROMPTS = (PAPER_PROMPT, "sheet of paper", "white rectangle")
_paper_prompt_order = list(PAPER_PROMPTS)
PAPER_SRGB = 0.94  # office paper under a lamp reads close to, not exactly, white
_strip_prompt_order = list(STRIP_PROMPTS)


@dataclass
class Instance:
    mask: np.ndarray  # bool (H, W)
    box: tuple[int, int, int, int]  # left, top, right, bottom

    @property
    def aspect(self) -> float:
        l, t, r, b = self.box
        return (b - t) / max(1, r - l)

    @property
    def elongation(self) -> float:
        """Long side over short side, whichever way the strip lies."""
        l, t, r, b = self.box
        w, h = max(1, r - l), max(1, b - t)
        return max(w, h) / min(w, h)

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


def _white_balance(lin: np.ndarray, paper: Instance | None, fabrics: list[Instance]) -> tuple[np.ndarray, list[str]]:
    """Single-reference correction from bare paper (paper minus anything lying on it)."""
    if paper is None:
        return np.vstack([np.eye(3), np.zeros((1, 3))]), ["WARN: no colour card or white paper in frame; camera colours, uncalibrated"]
    bare = paper.mask.copy()
    for f in fabrics:
        bare &= ~f.mask
    bare = _erode(bare, 6)
    if bare.sum() < 500:
        return np.vstack([np.eye(3), np.zeros((1, 3))]), ["WARN: too little bare paper visible; camera colours, uncalibrated"]
    white = np.median(lin[bare], axis=0)
    target = srgb_to_linear(np.full(3, PAPER_SRGB))
    M = np.vstack([np.diag(target / np.maximum(white, 1e-4)), np.zeros((1, 3))])
    return M, ["WARN: white-paper balance, no colour card. Absolute colour may be off by up to ~5 dE00, but dips and target shot on the same paper under the same light compare fairly"]


def _find_paper(img: Image.Image, jpeg: bytes, segmenter: Segmenter) -> Instance | None:
    for prompt in list(_paper_prompt_order):
        papers = segmenter.segment(img, jpeg, prompt)
        if papers:
            _paper_prompt_order.remove(prompt)
            _paper_prompt_order.insert(0, prompt)
            return max(papers, key=lambda i: i.mask.sum())
    return None


def _on_paper(obj: Instance, paper: Instance | None) -> bool:
    """Cloth being measured lies on the paper; a bedsheet or tablecloth spreads beyond it."""
    if paper is None:
        return True
    h, w = paper.mask.shape
    l, t, r, b = paper.box
    tol_x, tol_y = 0.03 * w, 0.03 * h
    ol, ot, or_, ob = obj.box
    return ol >= l - tol_x and ot >= t - tol_y and or_ <= r + tol_x and ob <= b + tol_y


def _cloth_in_view(objects: list[Instance], paper: Instance | None) -> list[Instance]:
    """Drop background cloth SAM also calls "fabric", unless nothing else is left."""
    on = [o for o in objects if _on_paper(o, paper)]
    if paper is None:  # no paper: background cloth is what runs off three or more edges of the frame
        h, w = objects[0].mask.shape if objects else (0, 0)
        on = [o for o in objects if sum([o.box[0] <= 2, o.box[1] <= 2, o.box[2] >= w - 2, o.box[3] >= h - 2]) < 3]
    return on or objects


def _bare_paper(paper: Instance, objects: list[Instance], card_zone: np.ndarray | None) -> np.ndarray:
    bare = paper.mask.copy()
    for f in objects:
        bare &= ~f.mask
    if card_zone is not None:
        bare &= ~card_zone
    return _erode(bare, 6)


def _flat_field(lin: np.ndarray, bare: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Even out the light across the frame, using the paper as a white reference everywhere.

    A desk lamp, a window or a phone flash lights the strip and the card unequally.
    Fits a smooth (cubic in log) light field per channel to the bare paper,
    ignoring cast shadows and stray threads, and divides it out.
    """
    if bare.sum() < 2000:
        return lin, []
    h, w = bare.shape
    ys, xs = np.nonzero(bare)
    pick = np.random.default_rng(0).choice(len(ys), size=min(len(ys), 6000), replace=False)
    ys, xs = ys[pick], xs[pick]

    def basis(y, x):
        y, x = 2 * y / h - 1, 2 * x / w - 1
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3], axis=-1)

    A = basis(ys.astype(float), xs.astype(float))
    lum = np.log(np.maximum(lin[ys, xs], 1e-4))
    grey = lum.mean(axis=1)
    keep = grey > np.median(grey) + np.log(0.5)  # cast shadows are far darker than lit paper
    for _ in range(4):  # robust: drop shadows and outliers, refit
        coef, *_ = np.linalg.lstsq(A[keep], lum[keep], rcond=None)
        r = (lum - A @ coef).mean(axis=1)
        sd = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))  # MAD: shadows can't inflate it
        keep = (r > -3 * sd) & (r < 3 * sd)
    gy, gx = np.mgrid[0:h:8, 0:w:8]  # the field is smooth: evaluate coarse, then resize
    # Beyond the paper there is no evidence: hold the edge value instead of extrapolating a cubic.
    gy = np.clip(gy, ys.min(), ys.max()).astype(float)
    gx = np.clip(gx, xs.min(), xs.max()).astype(float)
    coarse = (basis(gy, gx) @ coef).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR).astype(float)
    field -= np.median(field[bare], axis=0)
    gain = np.clip(np.exp(field), 0.4, 2.5)
    spread = float(np.exp(np.percentile(field[bare].mean(axis=1), 95) - np.percentile(field[bare].mean(axis=1), 5)))
    if spread > 6:  # a phone flash or a lamp right over the paper: too steep to even out well
        flags = [f"WARN: very uneven light ({spread:.0f}x across the paper; flash?); colour may be off by several dE00, retake in even light without flash"]
    else:
        flags = [f"NOTE: evened out uneven light ({spread:.1f}x across the paper)"] if spread > 1.25 else []
    return lin / gain, flags


def _card_correction(lin: np.ndarray, card: Instance) -> tuple[np.ndarray, float, list[str]]:
    from .colour import fit_colour_correction

    patches = _patch_samples(lin, _card_inner_box(card))
    ref = reference_linear()
    M = fit_colour_correction(patches, ref)
    corrected = np.clip(apply_colour_correction(M, patches), 0, 1)
    residual = float(np.mean(delta_e_2000(xyz_to_lab(linear_rgb_to_xyz(corrected)), xyz_to_lab(linear_rgb_to_xyz(ref)))))
    if residual > 4:
        return M, residual, [f"REJECT: card fit residual {residual:.1f} dE00; light on the card is uneven or glary, retake the photo"]
    if residual > 2:
        return M, residual, [f"WARN: card fit residual {residual:.1f} dE00"]
    return M, residual, []


def measure(jpeg: bytes, segmenter: Segmenter, with_target: bool = False, seed: int = 0) -> Measurement:
    img, srgb = decode(jpeg)
    lin = srgb_to_linear(srgb)
    rng = np.random.default_rng(seed)
    flags: list[str] = []

    # References: the 24-patch card (colour), the sheet of white paper (light across the frame).
    cards = segmenter.segment(img, jpeg, CARD_PROMPT)
    card = max(cards, key=lambda i: i.mask.sum()) if cards else None
    paper = _find_paper(img, jpeg, segmenter)
    residual = float("nan")

    # A margin around the card: people often lay it overlapping the strip.
    card_zone = ~_erode(~card.mask, 6) if card is not None else None

    def keep(f: Instance) -> bool:
        if card is not None and (f.mask & card.mask).sum() >= 0.5 * f.mask.sum():
            return False  # mostly card: it is the card
        # The strip lies on the paper; drop only the paper itself.
        return paper is None or f.mask.sum() < 0.6 * paper.mask.sum()

    def trim(f: Instance) -> Instance:
        if card_zone is None or not (f.mask & card_zone).any():
            return f
        mask = f.mask & ~card_zone
        ys, xs = np.nonzero(mask)
        return Instance(mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    fabrics: list[Instance] = []
    for prompt in list(_strip_prompt_order):  # cloth words first; shape words when the scene is plain
        fabrics = [trim(f) for f in _cloth_in_view(segmenter.segment(img, jpeg, prompt), paper) if keep(f)]
        if fabrics:
            # The rig and lighting don't change between dips: try what worked first next time.
            _strip_prompt_order.remove(prompt)
            _strip_prompt_order.insert(0, prompt)
            break
    if not fabrics:
        raise RuntimeError("fabric strip not found in frame")
    if paper is not None:
        lin, f0 = _flat_field(lin, _bare_paper(paper, fabrics, card_zone))
        flags += f0
    if card is not None:
        M, residual, f1 = _card_correction(lin, card)
        if f1 and f1[0].startswith("REJECT") and paper is not None:
            M, f2 = _white_balance(lin, paper, fabrics)
            flags += [f"WARN: colour card unreadable (fit residual {residual:.1f} dE00, uneven light or glare on it); used the white paper instead"] + f2
        else:
            flags += f1
    else:
        M, f1 = _white_balance(lin, paper, fabrics)
        flags += f1
    # The test strip is the most elongated piece of cloth, lying either way.
    strip = max(fabrics, key=lambda i: i.elongation)
    lab, lab_sd, n, f2 = _lab_of_region(lin, strip.mask, M, rng)
    flags += f2

    target_lab = None
    if with_target:
        others = [f for f in fabrics if f is not strip]
        if not others:
            targets = segmenter.segment(img, jpeg, TARGET_PROMPT)
            others = [trim(t) for t in _cloth_in_view(targets, paper) if not (t.mask & strip.mask).any() and keep(t)]
        if others:
            target = max(others, key=lambda i: i.mask.sum())
            target_lab, _, _, _ = _lab_of_region(lin, target.mask, M, rng)
    ref_box = card.box if card is not None else (paper.box if paper is not None else (0, 0, 0, 0))
    return Measurement(lab, lab_sd, residual, n, flags, ref_box, strip.box, target_lab)


def target_from_photo(jpeg: bytes, concept: str, segmenter: Segmenter) -> tuple[np.ndarray, str]:
    """Lab of the named object in a customer's photo (e.g. "teal kurta").

    If the photo also shows the colour card, the result is calibrated;
    with white paper in view it is white-balanced the same way dips are; otherwise
    it is the photo's own sRGB. Returns (lab, calibration) with calibration one of
    "card", "paper" or "none".
    """
    img, srgb = decode(jpeg)
    lin = srgb_to_linear(srgb)
    rng = np.random.default_rng(0)
    objects: list[Instance] = []
    # The customer's own word first ("kurta"), then generic cloth words SAM knows well.
    for prompt in dict.fromkeys([concept, "fabric", "cloth", "garment"]):
        objects = segmenter.segment(img, jpeg, prompt)
        if objects:
            break
    if not objects:
        raise RuntimeError(f"could not find '{concept}' or any fabric in the photo")
    paper = _find_paper(img, jpeg, segmenter)
    if paper is not None and all((paper.mask & o.mask).sum() > 0.5 * paper.mask.sum() for o in objects):
        paper = None  # that "paper" is the garment itself
    # The garment on the paper, not the bedsheet under it.
    obj = max(_cloth_in_view(objects, paper), key=lambda i: i.mask.sum())
    cards = segmenter.segment(img, jpeg, CARD_PROMPT)
    card = max(cards, key=lambda i: i.mask.sum()) if cards else None
    if paper is not None:
        card_zone = ~_erode(~card.mask, 6) if card is not None else None
        lin, _ = _flat_field(lin, _bare_paper(paper, objects, card_zone))
    cc = _card_correction(lin, card) if card is not None else None
    if cc is not None and not any(f.startswith("REJECT") for f in cc[2]):
        M, mode = cc[0], "card"
    elif cc is not None and paper is None:
        raise RuntimeError("the colour card in this photo is unreadable (uneven light or glare); retake it in even light")
    else:
        M, _ = _white_balance(lin, paper, objects)
        mode = "paper" if paper is not None else "none"
    lab, _, _, _ = _lab_of_region(lin, obj.mask, M, rng)
    return lab, mode


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
