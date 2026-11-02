"""A first home recipe for any colour: drops of blue, red and yellow household colour.

The public demo's take-home: photograph your outfit, SAM 3.1 reads its colour, and the household
dye model turns it into drops for a 4-teaspoon bath. It is a first-dip estimate (inks and food
colours vary by brand); assisted mode then corrects it dip by dip.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .colour import delta_e_2000, lab_to_hex
from .km import solve_recipe
from .profiles import PROFILES

HOME = PROFILES["home"]
REACHABLE_DE00 = 3.0  # the model's own predicted error above which it says it can't get there


def home_recipe(target_lab) -> dict:
    model = HOME.nominal()
    target = np.asarray(target_lab, dtype=float)
    conc, _ = solve_recipe(model, target, max_conc=HOME.max_conc)  # drops per 20 mL
    best = None
    # Pale shades need less than a drop: a bigger bath (8 or 16 teaspoons) gives finer steps.
    for scale in (1, 2, 4):
        options = itertools.product(*[sorted({math.floor(c * scale), math.ceil(c * scale)}) for c in conc])
        for o in options:
            d = np.array(o, dtype=float)
            lab = model.predict_lab(d / scale)
            de = float(delta_e_2000(lab, target))
            if best is None or de < best[0] - 1e-9:
                best = (de, scale, d, lab)
        if best[0] <= 1.0:
            break
    de, scale, d, lab = best
    return {
        "drops": {name: int(n) for name, n in zip(HOME.names, d)},
        "water_ml": 20 * scale,
        "teaspoons": 4 * scale,
        "predicted_de00": round(de, 1),
        "reachable": bool(de <= REACHABLE_DE00),
        "closest_hex": lab_to_hex(lab),
    }
