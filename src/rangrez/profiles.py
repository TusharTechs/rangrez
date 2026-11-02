"""Dye sets the lab can run. Choose with RANGREZ_DYES=reactive|food.

reactive: cold fibre-reactive dyes on soda-ash cotton, as a dye house uses.
food:     supermarket food colours on plain cotton; a stand-in anyone can buy
          tonight. Recipes are in % of the bath that is food colour.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .km import DyeModel, absorption_band


@dataclass(frozen=True)
class Profile:
    key: str
    names: tuple[str, ...]
    unit: str  # unit of a recipe entry, as shown to people and to the agent
    stock_per_l: float  # stock concentration in the same unit, so mL = c / stock * bath
    max_conc: float
    description: str
    nominal: Callable[[], DyeModel]
    bottle_ml: float = 250.0  # stock on hand per dye
    min_stock_ml: float = 5.0
    step_ml: float = 0.1  # smallest amount a person can measure in assisted mode


def nominal_reactive() -> DyeModel:
    """Supplier's nominal turquoise, fuchsia, lemon yellow (cold reactive)."""
    unit_ks = np.stack(
        [
            absorption_band(640, 55, 0.55),  # turquoise absorbs red
            absorption_band(540, 40, 0.60),  # fuchsia absorbs green
            absorption_band(430, 35, 0.50),  # lemon yellow absorbs blue
        ]
    )
    return DyeModel(["turquoise", "fuchsia", "lemon yellow"], unit_ks)


def nominal_food() -> DyeModel:
    """Brilliant blue, allura red and tartrazine food colours staining cotton."""
    unit_ks = np.stack(
        [
            absorption_band(630, 45, 0.30),  # blue (brilliant blue FCF peaks near 630 nm)
            absorption_band(505, 45, 0.30),  # red (allura red peaks near 504 nm)
            absorption_band(427, 35, 0.28),  # yellow (tartrazine peaks near 427 nm)
        ]
    )
    return DyeModel(["blue", "red", "yellow"], unit_ks)


def nominal_home() -> DyeModel:
    """Household blue, red and yellow inks, per drop in a 20 mL bath."""
    unit_ks = np.stack(
        [
            absorption_band(620, 55, 0.10),
            absorption_band(515, 50, 0.10),
            absorption_band(430, 40, 0.09),
        ]
    )
    return DyeModel(["blue", "red", "yellow"], unit_ks)


PROFILES = {
    "reactive": Profile(
        "reactive", ("turquoise", "fuchsia", "lemon yellow"), "g/L", 20.0, 10.0,
        "cold fibre-reactive dyes on soda-ash-treated cotton; concentrations are g/L of dye in a 20 mL bath",
        nominal_reactive,
    ),
    "food": Profile(
        "food", ("blue", "red", "yellow"), "%", 100.0, 15.0,
        "liquid food colours on white cotton; amounts are % of the 20 mL bath that is food colour (the rest is water); a person measures them with syringes to 0.1 mL",
        nominal_food, bottle_ml=30.0, min_stock_ml=2.0,
    ),
    # Household inks (sketch-pen ink, food colour, haldi...) counted in drops of about 0.05 mL
    # into a 20 mL (4 teaspoon) bath: c drops -> c * 0.05 mL, so stock = 20 / 0.05 = 400.
    "home": Profile(
        "home", ("blue", "red", "yellow"), "drops", 400.0, 40.0,
        "household colours (sketch-pen ink, food colour) on white cotton; amounts are drops of each colour in 4 teaspoons (20 mL) of water, counted by a person",
        nominal_home, bottle_ml=20.0, min_stock_ml=0.5, step_ml=0.05,
    ),
}

PROFILE = PROFILES[os.environ.get("RANGREZ_DYES", "reactive")]
