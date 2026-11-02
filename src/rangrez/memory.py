"""Lab memory: every dip this lab has made with this dye lot, kept across campaigns.

A colourist who has dyed a hundred shades with this lot does not start the next one from the
supplier's sheet. Neither does Rangrez: each campaign starts from a dye model fitted to every dip
remembered, and is offered the most similar dip ever made, corrected towards the new target.
The memory is a JSON file per dye set and lot (RANGREZ_MEMORY to move it, "off" to disable).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .colour import delta_e_2000
from .km import DyeModel, fit_gains
from .profiles import PROFILE

RECENT_DIPS = 200  # a dye lot drifts slowly: fit to the most recent dips


def default_path() -> Path | None:
    raw = os.environ.get("RANGREZ_MEMORY", f"lab/memory_{PROFILE.key}.json")
    return None if raw.lower() == "off" else Path(raw)


class LabMemory:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.data = {"dye_set": PROFILE.key, "dips": [], "campaigns": []}
        if path is not None and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @property
    def dips(self) -> list[dict]:
        return self.data["dips"]

    @property
    def campaigns(self) -> list[dict]:
        return self.data["campaigns"]

    def record_dip(self, conc, lab, campaign: str) -> None:
        self.dips.append({"conc": [round(float(v), 4) for v in conc], "lab": [round(float(v), 3) for v in lab],
                          "campaign": campaign, "at": time.time()})
        self.save()

    def record_campaign(self, summary: dict) -> None:
        self.campaigns.append({**summary, "at": time.time()})
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)

    def recent(self, n: int = RECENT_DIPS) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.array(d["conc"]), np.array(d["lab"])) for d in self.dips[-n:]]

    def warm_model(self, nominal: DyeModel) -> DyeModel:
        """The dye model fitted to what this lot actually did, not what the supplier says."""
        data = self.recent()
        return fit_gains(nominal, data) if len(data) >= 3 else nominal.copy()

    def nearest(self, target_lab) -> tuple[np.ndarray, np.ndarray, float] | None:
        """The remembered dip closest in colour to target_lab: (recipe, measured Lab, dE00)."""
        if not self.dips:
            return None
        labs = np.array([d["lab"] for d in self.dips])
        des = delta_e_2000(labs, np.broadcast_to(np.asarray(target_lab, dtype=float), labs.shape))
        i = int(np.argmin(des))
        return np.array(self.dips[i]["conc"]), labs[i], float(des[i])
