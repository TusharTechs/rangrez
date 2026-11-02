"""A camera that takes realistic phone photos of the simulator's cloth.

Scenes are real-looking photos (made with Muse Image) of undyed white cotton on
printer paper, under desk lamps, daylight, tube lights, phone flash and more.
SAM 3.1 masked each scene once (eval/scene_masks.py) and the masks are cached
in scenes/masks. To photograph a dyed strip, the strip's pixels are recoloured
to its true colour under the scene's own light, keeping the cloth's real folds,
weave and shadows. The paper's true whiteness and tint are randomised and NOT
told to the pipeline, the way real printer paper is never exactly white.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .colour import lab_to_srgb, linear_to_srgb, srgb_to_linear
from .vision import CARD_PROMPT, PAPER_PROMPTS, TARGET_PROMPT, Instance

SCENE_DIR = Path(__file__).parent / "scenes"
COTTON_LIN = 0.80  # linear reflectance of undyed cotton, to infer the light when there is no paper


def scene_names(kind: str = "strip") -> list[str]:
    """Scenes with a test strip ("strip"), or with only a garment to read ("target")."""
    masks = SCENE_DIR / "masks"
    strips = sorted(p.name[: -len("_strip.png")] for p in masks.glob("*_strip.png"))
    if kind == "strip":
        return strips
    return sorted(n for n in (p.name[: -len("_target.png")] for p in masks.glob("*_target.png")) if n not in strips)


@cache
def _card_lin() -> np.ndarray:
    from .make_card import render as card_image  # the same card people print

    return srgb_to_linear(np.asarray(card_image(), dtype=float) / 255)


def _mask(name: str, kind: str) -> np.ndarray | None:
    p = SCENE_DIR / "masks" / f"{name}_{kind}.png"
    return np.asarray(Image.open(p)) > 127 if p.exists() else None


def _instance(mask: np.ndarray) -> Instance:
    ys, xs = np.nonzero(mask)
    return Instance(mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


@dataclass(frozen=True, eq=False)
class Scene:
    name: str
    lin: np.ndarray
    strip: np.ndarray | None
    paper: np.ndarray | None
    target: np.ndarray | None
    background: np.ndarray | None  # bedsheet or tablecloth SAM also calls "fabric"

    @staticmethod
    @cache
    def load(name: str) -> Scene:
        rgb = np.asarray(Image.open(SCENE_DIR / f"{name}.jpg").convert("RGB"), dtype=float) / 255
        return Scene(name, srgb_to_linear(rgb), _mask(name, "strip"), _mask(name, "paper"), _mask(name, "target"),
                     _mask(name, "background"))

    @property
    def cloth(self) -> np.ndarray:
        out = np.zeros(self.lin.shape[:2], bool)
        for m in (self.strip, self.target):
            if m is not None:
                out |= m
        return out

    @property
    def bare(self) -> np.ndarray:
        """Where a card can lie: the paper, or the table when there is no paper."""
        base = self.paper if self.paper is not None else np.ones(self.lin.shape[:2], bool)
        return base & ~self.cloth

    def light(self) -> np.ndarray:
        """Linear RGB of white under this scene's light, assuming nominal paper / cotton."""
        if self.paper is not None:
            return np.median(self.lin[self.bare], axis=0)
        return np.median(self.lin[self.cloth], axis=0) / COTTON_LIN

    @cached_property
    def illumination(self) -> np.ndarray:
        """Relative light across the frame (1 = median over the paper), smoothed from the bare paper.

        This is how the scene was lit: lamp falloff, window light, flash hotspot.
        Cast shadows beside the cloth are left out so they do not darken the cloth itself.
        """
        h, w = self.lin.shape[:2]
        if self.paper is None:
            return np.ones((h, w, 3))
        small, bare = self.lin[::4, ::4], self.bare[::4, ::4].astype(float)
        weight = bare
        for _ in range(2):
            num = gaussian_filter(small * weight[..., None], (25, 25, 0))
            den = gaussian_filter(weight, 25)[..., None]
            field = num / np.maximum(den, 1e-9)
            weight = bare * (small.mean(axis=2) > 0.8 * field.mean(axis=2))  # drop shadows, refit
        field /= np.median(field[bare > 0], axis=0)
        return cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(float)

    def card_box(self, overlap: bool = False) -> tuple[int, int, int, int]:
        """Card placement: fully on bare paper, or (overlap=True) partly over the cloth's edge."""
        return _card_box(self.name, overlap)


@cache
def _card_box(name: str, overlap: bool) -> tuple[int, int, int, int]:
    scene = Scene.load(name)
    bare = scene.bare
    ii = np.pad(bare.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    for width in (0.22, 0.18, 0.15):  # a smaller print when the paper margin is narrow
        cw = int(width * scene.lin.shape[1])
        ch = int(cw * _card_lin().shape[0] / _card_lin().shape[1])
        best, spot = -np.inf, (0, 0)
        for y in range(0, bare.shape[0] - ch, 16):
            for x in range(0, bare.shape[1] - cw, 16):
                frac = (ii[y + ch, x + cw] - ii[y, x + cw] - ii[y + ch, x] + ii[y, x]) / (ch * cw)
                score = -abs(frac - 0.7) if overlap else frac
                if score > best:
                    best, spot = score, (y, x)
        if overlap or best > 0.97:
            break
    y, x = spot
    return (x, y, x + cw, y + ch)


def render(scene: Scene, strip_lab=None, target_lab=None, card: bool = False, seed: int = 0,
           paper_seed: int | None = None, overlap: bool = False) -> tuple[bytes, dict[str, np.ndarray]]:
    """One phone photo and the masks of what is in it. Same paper_seed: same sheet of paper.

    strip_lab=None leaves the strip undyed. Returns the JPEG and {"strip", "target",
    "paper", "card"} masks (absent keys: not in the photo).
    """
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed if paper_seed is None else paper_seed)
    lin = scene.lin.copy()
    # True paper (or cotton) whiteness, unknown to the pipeline: 0.86-0.96, slightly bluish (optical brighteners).
    tint = prng.uniform(0.86, 0.96) * np.array([0.985, 0.995, 1.02]) * prng.normal(1, 0.01, 3)
    light = scene.light() / tint if scene.paper is not None else scene.light()  # true illuminant
    masks: dict[str, np.ndarray] = {}
    for kind, lab in (("strip", strip_lab), ("target", target_lab)):
        mask = getattr(scene, kind)
        if mask is None:
            continue
        masks[kind] = mask
        if lab is not None:
            refl = srgb_to_linear(lab_to_srgb(np.asarray(lab, dtype=float)))
            lit = scene.illumination[mask]
            texture = scene.lin[mask] / lit  # folds, weave and shading of the white cloth
            texture /= np.median(texture, axis=0)
            lin[mask] = texture * lit * light * refl
    if scene.paper is not None:
        masks["paper"] = scene.paper
    if scene.background is not None:
        masks["background"] = scene.background
    if card:
        l, t, r, b = scene.card_box(overlap)
        img = Image.fromarray((linear_to_srgb(_card_lin()) * 255).astype(np.uint8)).resize((r - l, b - t))
        # The card is lit by the light where it lies, like everything else in the scene.
        card_lin = srgb_to_linear(np.asarray(img, dtype=float) / 255)
        lin[t:b, l:r] = card_lin * light * scene.illumination[t:b, l:r]
        cm = np.zeros(lin.shape[:2], bool)
        cm[t:b, l:r] = True
        masks["card"] = cm
    # Camera: auto exposure and white balance wobble, sensor noise, JPEG.
    lin = lin * rng.uniform(0.9, 1.1) * rng.normal(1, 0.02, 3) + rng.normal(0, 0.004, lin.shape)
    out = (linear_to_srgb(lin) * 255).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=int(rng.integers(82, 95)))
    return buf.getvalue(), masks


class SceneSegmenter:
    """Offline stand-in for hosted SAM 3.1: answers prompts from the scene's cached SAM masks."""

    def __init__(self, masks: dict[str, np.ndarray]):
        self.masks = masks

    def segment(self, image: Image.Image, jpeg: bytes, concept: str) -> list[Instance]:
        if concept == CARD_PROMPT:
            kinds = ["card"]
        elif concept in PAPER_PROMPTS:
            kinds = ["paper"]
        elif concept == TARGET_PROMPT:
            kinds = ["target"]
        else:  # "fabric", "cloth", "kurta"...: SAM returns every piece of cloth, bedsheet included
            kinds = ["strip", "target", "background"]
        return [_instance(self.masks[k]) for k in kinds if k in self.masks]
