"""Recipes, the rig interface, and the deterministic safety gate.

Model output never reaches a pump directly. The agent proposes a Recipe; the
gate checks it against hard limits; only then does the executor turn it into
pump volumes. The firmware enforces the same limits a second time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .profiles import PROFILE

DYES = list(PROFILE.names)
UNIT = PROFILE.unit


@dataclass(frozen=True)
class RigLimits:
    bath_ml: float = 20.0  # liquor per shot
    stock_g_per_l: float = PROFILE.stock_per_l  # stock concentration, in UNIT
    max_conc_g_per_l: float = PROFILE.max_conc  # per dye, in the bath, in UNIT
    max_dye_fraction: float = 0.9  # dye stock may fill at most this share of the bath
    dip_seconds: tuple[float, float] = (30.0, 300.0)
    min_stock_ml: float = PROFILE.min_stock_ml  # refuse to run a bottle dry


@dataclass(frozen=True)
class Recipe:
    conc: tuple[float, ...]  # g/L in the bath, one per dye in DYES order
    dip_seconds: float = 90.0

    def as_array(self) -> np.ndarray:
        return np.array(self.conc, dtype=float)

    def describe(self) -> str:
        parts = [f"{name} {c:.2f} {UNIT}" for name, c in zip(DYES, self.conc)]
        return ", ".join(parts) + f", dip {self.dip_seconds:.0f} s"


@dataclass
class PumpPlan:
    dye_ml: tuple[float, ...]
    water_ml: float


class GateError(ValueError):
    """A recipe the gate refused. The message says why, for the agent to read."""


@dataclass
class Inventory:
    stock_ml: list[float] = field(default_factory=lambda: [PROFILE.bottle_ml] * len(PROFILE.names))
    strips: int = 60


def check_recipe(recipe: Recipe, limits: RigLimits, inventory: Inventory) -> PumpPlan:
    """Deterministic gate. Returns the pump plan or raises GateError."""
    conc = recipe.as_array()
    if conc.shape != (len(DYES),):
        raise GateError(f"recipe needs {len(DYES)} concentrations, got {conc.shape[0]}")
    if not np.all(np.isfinite(conc)):
        raise GateError("recipe has a non-finite concentration")
    if np.any(conc < 0):
        raise GateError("concentrations must be >= 0")
    if np.any(conc > limits.max_conc_g_per_l):
        raise GateError(f"each dye is capped at {limits.max_conc_g_per_l} {UNIT}")
    lo, hi = limits.dip_seconds
    if not lo <= recipe.dip_seconds <= hi:
        raise GateError(f"dip time must be between {lo:.0f} and {hi:.0f} s")
    dye_ml = conc / limits.stock_g_per_l * limits.bath_ml
    if dye_ml.sum() > limits.max_dye_fraction * limits.bath_ml:
        raise GateError("total dye stock exceeds the bath volume; lower the concentrations")
    for name, need, have in zip(DYES, dye_ml, inventory.stock_ml):
        if have - need < limits.min_stock_ml:
            raise GateError(f"{name} stock is too low ({have:.0f} mL); refill before continuing")
    if inventory.strips <= 0:
        raise GateError("out of cotton strips")
    return PumpPlan(tuple(float(v) for v in dye_ml), float(limits.bath_ml - dye_ml.sum()))


@dataclass
class ShotResult:
    image_jpeg: bytes
    plan: PumpPlan
    started_at: float
    finished_at: float
    extra: dict = field(default_factory=dict)


class Rig(Protocol):
    limits: RigLimits
    inventory: Inventory

    def run_shot(self, recipe: Recipe) -> ShotResult: ...

    def capture(self) -> bytes: ...


def execute(rig: Rig, recipe: Recipe) -> ShotResult:
    """Gate, then run. The only path from a recipe to the pumps."""
    plan = check_recipe(recipe, rig.limits, rig.inventory)
    result = rig.run_shot(recipe)
    for i, ml in enumerate(plan.dye_ml):
        rig.inventory.stock_ml[i] -= ml
    rig.inventory.strips -= 1
    result.plan = plan
    return result


def now() -> float:
    return time.time()
