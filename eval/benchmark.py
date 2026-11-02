"""Offline benchmark of the campaign tools: many reachable shades, every dip a realistic
photo in a different room, the deterministic policy (same tools Wonder uses).

    uv run python eval/benchmark.py                    # reactive dyes, robot rig
    RANGREZ_DYES=home uv run python eval/benchmark.py  # household drops, counted by a person
"""

from __future__ import annotations

import os

import numpy as np

from rangrez.agent import run_deterministic
from rangrez.assisted import AssistedRig
from rangrez.campaign import Campaign
from rangrez.colour import delta_e_2000
from rangrez.photoreal import scene_names
from rangrez.profiles import PROFILE
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig

HOME = os.environ.get("RANGREZ_DYES") == "home"
N = int(os.environ.get("N", "24"))


class PersonRig(VirtualRig):
    quantize = AssistedRig.quantize


def main() -> None:
    rng = np.random.default_rng(0)
    scenes = [s for s in scene_names() if s not in {"phone_flash", "no_paper"}]
    rows = []
    for i in range(N):
        recipe = tuple(float(v) for v in np.round(rng.uniform(0, 0.5, 3) * PROFILE.max_conc, 1))
        rig = (PersonRig if HOME else VirtualRig)(seed=i, scene=scenes[i % len(scenes)])
        target = rig.true_swatch_lab(Recipe(recipe))
        c = Campaign(target, rig, rig.segmenter(), tolerance=2.0 if HOME else 1.5, max_shots=10)
        run_deterministic(c)
        true = float(delta_e_2000(rig.true_swatch_lab(c.best.recipe), target)) if c.best else float("nan")
        rows.append((c.status, len(c.shots), true))
        print(f"{rig.scene:16s} {recipe} -> {c.status:12s} dips {len(c.shots):2d}  "
              f"{[round(s.delta_e, 1) for s in c.shots]}  true best {true:.2f}")
    matched = [r for r in rows if r[0] == "matched"]
    print(f"\n{PROFILE.key}: matched {len(matched)}/{len(rows)}, median dips {np.median([r[1] for r in matched]):.0f}, "
          f"median true dE00 of the matched {np.median([r[2] for r in matched]):.2f}")


if __name__ == "__main__":
    main()
