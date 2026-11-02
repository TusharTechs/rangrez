"""Photo-realistic test photos with known true colours.

A Muse Image scene (undyed strip on paper) is recoloured so the strip reflects
a known colour under the scene's own light, keeping the cloth's real texture,
folds and shadows. The paper's true whiteness and tint are randomised and NOT
told to the pipeline, the way real printer paper is never exactly white.
Optionally the printable colour card is placed in the scene under the same light.
"""
from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from rangrez.assisted import AssistedRig
from rangrez.colour import lab_to_srgb, linear_to_srgb, srgb_to_linear
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig

ROOT = os.path.join(os.path.dirname(__file__), "scenes")
SCENES = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/*.jpg"))
CARD = srgb_to_linear(np.asarray(Image.open(os.path.join(os.path.dirname(__file__), "..", "hardware", "colour_card.png")).convert("RGB"), dtype=float) / 255)


def _mask(name: str, kind: str) -> np.ndarray | None:
    p = f"{ROOT}/masks/{name}_{kind}.png"
    return np.asarray(Image.open(p)) > 127 if os.path.exists(p) else None


@dataclass
class Scene:
    name: str
    lin: np.ndarray
    strip: np.ndarray
    paper: np.ndarray
    target: np.ndarray | None

    @classmethod
    def load(cls, name: str) -> Scene:
        rgb = np.asarray(Image.open(f"{ROOT}/{name}.jpg").convert("RGB"), dtype=float) / 255
        return cls(name, srgb_to_linear(rgb), _mask(name, "strip"), _mask(name, "paper"), _mask(name, "target"))


def _shading(lin: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel texture and shadow of a white cloth, relative to its median."""
    px = lin[mask]
    return lin[mask] / np.maximum(np.median(px, axis=0), 1e-4)


def render(scene: Scene, strip_lab, target_lab=None, card: bool = False, seed: int = 0, paper_seed: int | None = None, overlap: bool = False) -> tuple[bytes, dict]:
    """One phone photo. The same paper_seed means the same sheet of paper across photos."""
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed if paper_seed is None else paper_seed)
    lin = scene.lin.copy()
    bare = scene.paper & ~scene.strip & (~scene.target if scene.target is not None else True)
    # True paper reflectance, unknown to the pipeline: 0.86-0.96, slightly bluish (optical brighteners).
    paper_true = prng.uniform(0.86, 0.96) * np.array([0.985, 0.995, 1.02]) * prng.normal(1, 0.01, 3)
    light = np.median(lin[bare], axis=0) / paper_true  # illuminant consistent with that paper
    for mask, lab in ((scene.strip, strip_lab), (scene.target, target_lab)):
        if mask is None or lab is None:
            continue
        refl = srgb_to_linear(lab_to_srgb(np.asarray(lab, dtype=float)))
        lin[mask] = _shading(lin, mask) * light * refl
    if card:
        ch, cw = CARD.shape[:2]
        scale = 0.22 * lin.shape[1] / cw
        card_img = Image.fromarray((linear_to_srgb(CARD) * 255).astype(np.uint8)).resize((int(cw * scale), int(ch * scale)))
        c = srgb_to_linear(np.asarray(card_img, dtype=float) / 255) * light
        y0, x0 = _card_spot(bare, c.shape[:2], overlap)
        lin[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    # Camera: auto exposure and white balance wobble, sensor noise, JPEG.
    lin = lin * rng.uniform(0.9, 1.1) * rng.normal(1, 0.02, 3) + rng.normal(0, 0.004, lin.shape)
    out = (linear_to_srgb(lin) * 255).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=int(rng.integers(82, 95)))
    return buf.getvalue(), {"scene": scene.name, "paper_true": paper_true.round(3).tolist(), "card": card}


def _card_spot(bare: np.ndarray, size: tuple[int, int], overlap: bool) -> tuple[int, int]:
    """Top-left for the card: fully on bare paper, or (overlap=True) half over the cloth's edge."""
    h, w = size
    ii = np.pad(bare.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, spot = -1.0, (0, 0)
    for y in range(0, bare.shape[0] - h, 16):
        for x in range(0, bare.shape[1] - w, 16):
            frac = (ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]) / (h * w)
            score = -abs(frac - 0.7) if overlap else frac
            if score > best:
                best, spot = score, (y, x)
    return spot


@dataclass
class PhotoRig(VirtualRig):
    """The simulator's hidden chemistry, photographed as a realistic phone photo."""

    scene_name: str = "lamp_desk"
    card: bool = False
    photo_seed: int = 0
    quantize = AssistedRig.quantize
    log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._scene = Scene.load(self.scene_name)

    def render(self) -> bytes:
        self.photo_seed += 1
        jpeg, meta = render(self._scene, self.last_swatch_lab, card=self.card, seed=self.seed * 1000 + self.photo_seed, paper_seed=self.seed)
        self.log.append({**meta, "true_lab": None if self.last_swatch_lab is None else self.last_swatch_lab.tolist()})
        return jpeg
