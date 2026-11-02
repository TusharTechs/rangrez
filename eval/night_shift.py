"""The night shift: campaign after campaign on one dye lot, with and without lab memory.

Same targets in the same order, every dip a realistic photo in a different room, the
deterministic policy (Wonder's tools and rules). With memory the lab starts each campaign
from a model fitted to every dip so far and is offered the most similar past dip, corrected.

    uv run python eval/night_shift.py            # 40 campaigns -> eval/night_shift.{json,png}
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rangrez.agent import run_deterministic
from rangrez.campaign import Campaign
from rangrez.colour import delta_e_2000
from rangrez.memory import LabMemory
from rangrez.photoreal import scene_names
from rangrez.profiles import PROFILE
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig

N = int(os.environ.get("N", "40"))
OUT = Path(__file__).parent


def targets() -> list[tuple[float, ...]]:
    rng = np.random.default_rng(42)
    return [tuple(float(v) for v in np.round(rng.uniform(0, 1, 3) ** 2 * PROFILE.max_conc * 0.6, 2)) for _ in range(N)]


def shift(memory: LabMemory | None) -> list[dict]:
    scenes = [s for s in scene_names() if s not in {"phone_flash", "no_paper"}]
    rows = []
    for i, recipe in enumerate(targets()):
        rig = VirtualRig(seed=500 + i, scene=scenes[i % len(scenes)])
        target = rig.true_swatch_lab(Recipe(recipe))
        c = Campaign(target, rig, rig.segmenter(), tolerance=1.5, max_shots=8, memory=memory)
        truths = []
        run = rig.run_shot

        def tracked(r, _run=run, _rig=rig):
            res = _run(r)
            truths.append(_rig.last_swatch_lab.copy())
            return res

        rig.run_shot = tracked
        run_deterministic(c)
        first_true = float(delta_e_2000(truths[0], target)) if truths else None
        rows.append({"campaign": i + 1, "status": c.status, "dips": len(c.shots), "first_true_de00": first_true})
        print(f"{'memory' if memory else 'fresh '} #{i + 1:2d} {c.status:12s} dips {len(c.shots)}  first dip true dE00 {first_true:.2f}")
    return rows


def chart(fresh: list[dict], mem: list[dict], path: Path) -> None:
    W, H, pad = 1600, 900, 90
    img = Image.new("RGB", (W, H), "#111214")
    d = ImageDraw.Draw(img)
    try:
        f, fb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26), ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    except OSError:
        f = fb = ImageFont.load_default()
    d.text((pad, 30), "The night shift: one dye lot, campaign after campaign", font=fb, fill="#eceae6")

    def panel(top: int, key: str, label: str, top_val: float) -> None:
        h = 330
        d.text((pad, top), label, font=f, fill="#9a968f")
        y0, y1 = top + 40, top + h

        def xy(i, v):
            return pad + (W - 2 * pad) * i / (N - 1), y1 - (y1 - y0) * min(v, top_val) / top_val

        for v in np.linspace(0, top_val, 5):
            d.line([(pad, xy(0, v)[1]), (W - pad, xy(0, v)[1])], fill="#26282c")
            d.text((pad - 10, xy(0, v)[1] - 12), f"{v:.0f}", font=f, fill="#6b6862", anchor="ra")
        for k, (rows, colour, name) in enumerate(((fresh, "#9a968f", "every campaign from the supplier's model"), (mem, "#e3a857", "with lab memory"))):
            vals = np.array([r[key] for r in rows], dtype=float)
            smooth = np.convolve(vals, np.ones(5) / 5, mode="valid")
            pts = [xy(i + 2, v) for i, v in enumerate(smooth)]
            d.line(pts, fill=colour, width=5)
            for i, v in enumerate(vals):
                x, y = xy(i, v)
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour)
            d.text((W - pad, top + 34 * k), name, font=f, fill=colour, anchor="ra")

    panel(110, "dips", "dips to a match (5-campaign average)", max(8.0, max(r["dips"] for r in fresh)))
    panel(500, "first_true_de00", "first dip's true error, dE00", 12.0)
    d.text((W / 2, H - 30), "campaign number", font=f, fill="#6b6862", anchor="mm")
    img.save(path)


def main() -> None:
    fresh = shift(None)
    with tempfile.TemporaryDirectory() as tmp:
        mem = shift(LabMemory(Path(tmp) / "memory.json"))
    summary = {}
    for name, rows in (("fresh", fresh), ("memory", mem)):
        late = rows[N // 2:]
        summary[name] = {
            "matched": sum(r["status"] == "matched" for r in rows),
            "mean_dips": round(float(np.mean([r["dips"] for r in rows])), 2),
            "mean_dips_second_half": round(float(np.mean([r["dips"] for r in late])), 2),
            "median_first_de00_second_half": round(float(np.median([r["first_true_de00"] for r in late])), 2),
        }
    print(json.dumps(summary, indent=2))
    (OUT / "night_shift.json").write_text(json.dumps({"summary": summary, "fresh": fresh, "memory": mem}, indent=2))
    chart(fresh, mem, OUT / "night_shift.png")


if __name__ == "__main__":
    main()
