"""A virtual dye lab: hidden 'real' dye chemistry photographed by a realistic phone camera.

The agent never sees anything in this module except the JPEG photos it returns
and the pump plan it asked for. Reality differs from the supplier model on
purpose: shifted absorption peaks, different uptake per dye lot, saturation at
high concentration, pump volume noise. Photos are real-looking phone shots
(see photoreal.py): uneven light, unknown paper whiteness, exposure and white
balance wobble, JPEG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .km import DyeModel, absorption_band, reflectance_to_lab
from .photoreal import Scene, SceneSegmenter, render
from .rig import DYES, Inventory, PumpPlan, Recipe, RigLimits, ShotResult, now


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
class VirtualRig:
    limits: RigLimits = field(default_factory=RigLimits)
    inventory: Inventory = field(default_factory=Inventory)
    seed: int = 7
    target_lab: np.ndarray | None = None  # target cloth lying beside the strip (scenes that have one)
    world: DyeModel = field(default_factory=true_world_model)
    saturation_g_per_l: float = 14.0
    scene: str = "lamp_desk"
    card: bool = True  # printed colour card in the photo; else white paper only
    card_overlap: bool = False  # card laid partly over the strip
    printed_card: bool = False  # card printed at home: patches a few dE00 off the reference
    pump_cv: float = 0.02

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.last_swatch_lab: np.ndarray | None = None
        self.photos = 0
        self.masks: dict[str, np.ndarray] = {}

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
        image = self.render()
        plan = PumpPlan(tuple(float(v) for v in planned), float(self.limits.bath_ml - planned.sum()))
        return ShotResult(image, plan, started, now(), {"delivered_ml": delivered.tolist()})

    def capture(self) -> bytes:
        return self.render()

    # -- camera ----------------------------------------------------------
    def render(self) -> bytes:
        self.photos += 1
        jpeg, masks = render(
            Scene.load(self.scene), self.last_swatch_lab, target_lab=self.target_lab, card=self.card,
            seed=self.seed * 1000 + self.photos, paper_seed=self.seed, overlap=self.card_overlap,
            printed_card=self.printed_card,
        )
        self.masks.clear()
        self.masks.update(masks)  # in place: segmenters handed out earlier see the latest photo
        return jpeg

    def segmenter(self) -> SceneSegmenter:
        """Offline stand-in for SAM 3.1 that knows where things are in this rig's photos."""
        return SceneSegmenter(self.masks)
