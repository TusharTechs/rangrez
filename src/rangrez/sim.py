"""A virtual dye lab: hidden 'real' dye chemistry plus a camera that renders frames.

The agent never sees anything in this module except the JPEG frames it returns
and the pump plan it asked for. Reality differs from the supplier model on
purpose: shifted absorption peaks, different uptake per dye lot, saturation at
high concentration, pump volume noise, lighting and white-balance drift.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .card import COLS, PATCH_SRGB_8BIT, ROWS
from .colour import lab_to_srgb, srgb_to_linear, linear_to_srgb
from .km import DyeModel, absorption_band, reflectance_to_lab
from .rig import DYES, Inventory, PumpPlan, Recipe, RigLimits, ShotResult, now

W, H = 1280, 720

# Fixed rig geometry: the camera is clamped, so everything sits in known places.
CARD_ORIGIN = (60, 170)
PATCH = 64
GAP = 8
SWATCH_BOX = (640, 150, 760, 590)  # left, top, right, bottom
TARGET_BOX = (860, 230, 1110, 510)
CUP_CENTRE, CUP_R = (1180, 110), 62
PAPER_BOX = (590, 110, 1150, 640)  # a sheet of printer paper under the strip and target


def card_patch_boxes() -> list[tuple[int, int, int, int]]:
    boxes = []
    x0, y0 = CARD_ORIGIN
    for r in range(ROWS):
        for c in range(COLS):
            left = x0 + c * (PATCH + GAP)
            top = y0 + r * (PATCH + GAP)
            boxes.append((left, top, left + PATCH, top + PATCH))
    return boxes


def card_box() -> tuple[int, int, int, int]:
    boxes = card_patch_boxes()
    return (boxes[0][0] - 12, boxes[0][1] - 12, boxes[-1][2] + 12, boxes[-1][3] + 12)


def true_world_model() -> DyeModel:
    """Reality: this dye lot is not what the supplier sheet says."""
    unit_ks = np.stack(
        [
            absorption_band(652, 62, 0.55),
            absorption_band(548, 44, 0.60),
            absorption_band(424, 40, 0.50),
        ]
    )
    model = DyeModel(list(DYES), unit_ks)
    model.gains = np.array([1.35, 0.78, 1.18])
    return model


@dataclass
class Camera:
    wb_gain: np.ndarray = field(default_factory=lambda: np.array([1.07, 1.0, 0.9]))
    exposure: float = 0.93
    noise: float = 0.006
    drift_per_shot: float = 0.004

    def drift(self, rng: np.random.Generator) -> None:
        self.wb_gain = self.wb_gain * (1 + rng.normal(0, self.drift_per_shot, 3))
        self.exposure *= 1 + rng.normal(0, self.drift_per_shot)


@dataclass
class VirtualRig:
    limits: RigLimits = field(default_factory=RigLimits)
    inventory: Inventory = field(default_factory=Inventory)
    seed: int = 7
    target_lab: np.ndarray | None = None  # physical target cloth in the light box
    world: DyeModel = field(default_factory=true_world_model)
    camera: Camera = field(default_factory=Camera)
    saturation_g_per_l: float = 14.0
    show_card: bool = True
    paper: bool = False
    pump_cv: float = 0.02

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.last_swatch_lab: np.ndarray | None = None
        self.last_liquid_rgb = np.array([0.85, 0.88, 0.9])

    # -- chemistry -------------------------------------------------------
    def fixed_dye(self, recipe: Recipe, dye_ml: np.ndarray) -> np.ndarray:
        """g/L that actually fixes on the strip, given what the pumps delivered."""
        conc = dye_ml / self.limits.bath_ml * self.limits.stock_g_per_l
        saturated = conc / (1 + conc / self.saturation_g_per_l)
        t = recipe.dip_seconds
        time_factor = (1 - np.exp(-t / 45)) / (1 - np.exp(-90 / 45))
        return saturated * time_factor

    def true_swatch_lab(self, recipe: Recipe) -> np.ndarray:
        dye_ml = recipe.as_array() / self.limits.stock_g_per_l * self.limits.bath_ml
        return reflectance_to_lab(self.world.reflectance(self.fixed_dye(recipe, dye_ml)))

    # -- rig protocol ----------------------------------------------------
    def run_shot(self, recipe: Recipe) -> ShotResult:
        started = now()
        planned = recipe.as_array() / self.limits.stock_g_per_l * self.limits.bath_ml
        delivered = planned * self.rng.normal(1.0, self.pump_cv, size=planned.shape)
        fixed = self.fixed_dye(recipe, np.maximum(delivered, 0))
        self.last_swatch_lab = reflectance_to_lab(self.world.reflectance(fixed))
        # Liquid in the cup: a rough transmission colour, for the picture only.
        self.last_liquid_rgb = np.clip(
            1 - 0.06 * (recipe.as_array() @ np.array([[0.9, 0.1, 0.0], [0.1, 0.85, 0.3], [0.0, 0.15, 0.9]])),
            0.05,
            1,
        )
        self.camera.drift(self.rng)
        image = self.render()
        plan = PumpPlan(tuple(float(v) for v in planned), float(self.limits.bath_ml - planned.sum()))
        return ShotResult(image, plan, started, now(), {"delivered_ml": delivered.tolist()})

    def capture(self) -> bytes:
        return self.render()

    # -- camera ----------------------------------------------------------
    def render(self) -> bytes:
        canvas = np.full((H, W, 3), 0.42, dtype=float)  # light-box floor, linear
        if self.show_card:
            cl, ct, cr, cb = card_box()
            canvas[ct:cb, cl:cr] = 0.012  # black card body
            ref_lin = srgb_to_linear(PATCH_SRGB_8BIT / 255.0)
            for (l, t, r, b), lin in zip(card_patch_boxes(), ref_lin):
                canvas[t:b, l:r] = lin
        if self.paper:
            pl, pt, pr, pb = PAPER_BOX
            canvas[pt:pb, pl:pr] = srgb_to_linear(0.94)

        if self.last_swatch_lab is not None:
            self._paint_cloth(canvas, SWATCH_BOX, self.last_swatch_lab)
        if self.target_lab is not None:
            self._paint_cloth(canvas, TARGET_BOX, self.target_lab)

        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = CUP_CENTRE
        cup = (xx - cx) ** 2 + (yy - cy) ** 2 <= CUP_R**2
        canvas[cup] = srgb_to_linear(self.last_liquid_rgb)

        # Camera: white balance, exposure, vignetting, sensor noise, gamma, JPEG.
        vignette = 1 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
        raw = canvas * self.camera.wb_gain * self.camera.exposure * vignette[..., None]
        raw += self.rng.normal(0, self.camera.noise, raw.shape)
        out = (linear_to_srgb(raw) * 255).round().clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    def _paint_cloth(self, canvas: np.ndarray, box: tuple[int, int, int, int], lab: np.ndarray) -> None:
        l, t, r, b = box
        base = srgb_to_linear(lab_to_srgb(lab))
        h, w = b - t, r - l
        yy, xx = np.mgrid[0:h, 0:w]
        weave = 1 + 0.025 * np.sin(xx * 1.9) * np.sin(yy * 1.7) + self.rng.normal(0, 0.01, (h, w))
        canvas[t:b, l:r] = base * weave[..., None]


def ground_truth_masks(card: bool = True, paper: bool = False) -> dict[str, tuple[int, int, int, int]]:
    """Boxes the offline segmenter uses in place of hosted SAM (tests, no API key)."""
    boxes = {"swatch": SWATCH_BOX, "target": TARGET_BOX}
    if card:
        boxes["card"] = card_box()
    if paper:
        boxes["paper"] = PAPER_BOX
    return boxes
