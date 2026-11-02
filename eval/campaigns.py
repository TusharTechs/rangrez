"""Assisted-mode campaigns on photo-realistic photos, live SAM 3.1 and Muse Wonder.

The person is simulated: they follow the dashboard's drop counts exactly, the
hidden chemistry dyes the strip, and each dried strip comes back as a realistic
phone photo. The target is a real-looking cloth photographed on the same paper
first. Reports the agent's measured dE00 and the TRUE dE00 of the best dip.

    RANGREZ_DYES=home uv run python eval/campaigns.py
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("RANGREZ_DYES", "home")
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np

from rangrez.agent import run_with_wonder
from rangrez.assisted import AssistedRig
from rangrez.campaign import Campaign
from rangrez.colour import delta_e_2000, lab_to_hex
from rangrez.photoreal import Scene, render
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig
from rangrez.vision import CallLog, MuseSamSegmenter, target_from_photo


class PersonWithPhone(VirtualRig):
    """Hidden chemistry; drops counted the way a person can; every dip a realistic phone photo."""

    quantize = AssistedRig.quantize

    def run_shot(self, recipe):
        shot = super().run_shot(recipe)
        self.truths = [*getattr(self, "truths", []), self.last_swatch_lab]
        return shot


# (scene, colour card in view, the drops that dyed the customer's target cloth)
CASES = [
    ("lamp_desk", False, (9, 3, 4)),
    ("daylight_window", True, (4, 8, 2)),
    ("shadowy", False, (2, 2, 12)),
    ("tilted_phone", True, (6, 6, 6)),
    ("evening_warm", False, (3, 9, 6)),
    ("kitchen_tube", True, (12, 2, 3)),
    ("crumpled_strip", True, (2, 6, 9)),
    ("vertical_strip", False, (7, 1, 10)),
]


def one(case, seed):
    scene_name, card, drops = case
    rig = PersonWithPhone(seed=seed, scene=scene_name, card=card)
    true_target = rig.true_swatch_lab(Recipe(tuple(float(d) for d in drops)))
    # The person photographs the target cloth on the same paper, under the same light.
    photo, _ = render(Scene.load(scene_name), true_target, card=card, seed=seed * 7 + 1, paper_seed=seed)
    log = CallLog()
    seg = MuseSamSegmenter(log)
    target, mode = target_from_photo(photo, "fabric", seg)
    c = Campaign(target, rig, seg, tolerance=2.0, max_shots=8)
    t0 = time.time()
    run_with_wonder(c, log)
    true_de = [float(delta_e_2000(t, true_target)) for t in getattr(rig, "truths", [])]
    best = int(np.argmin([s.delta_e for s in c.shots])) if c.shots else None
    return {
        "case": f"{scene_name} ({'card' if card else 'paper'})",
        "target": lab_to_hex(true_target),
        "target_read_err": float(delta_e_2000(target, true_target)),
        "mode": mode,
        "status": c.status,
        "measured": [round(s.delta_e, 2) for s in c.shots],
        "true": [round(d, 2) for d in true_de],
        "best_true": None if best is None else round(true_de[best], 2),
        "calls": len(log.calls),
        "failed_calls": sum(not a.ok for a in log.calls),
        "secs": round(time.time() - t0),
    }


def main() -> None:
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, CASES, range(41, 41 + len(CASES))))
    for r in rows:
        print(f"{r['case']:26s} target {r['target']} read via {r['mode']} (err {r['target_read_err']:.2f})  -> {r['status']}")
        print(f"   measured dE00 per dip {r['measured']}")
        print(f"   TRUE dE00 per dip     {r['true']}   best dip true dE00 {r['best_true']}")
        print(f"   api calls {r['calls']} failed {r['failed_calls']}  {r['secs']}s")
    print(f"\nmatched {sum(r['status'] == 'matched' for r in rows)}/{len(rows)}; "
          f"median dips {np.median([len(r['measured']) for r in rows]):.0f}")


if __name__ == "__main__":
    main()
