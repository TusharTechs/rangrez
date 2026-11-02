"""Measurement accuracy on photo-realistic scenes, live SAM 3.1, against known truth.

    uv run python eval/accuracy.py
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from realistic import SCENES, Scene, render  # noqa: E402

from rangrez.colour import delta_e_2000  # noqa: E402
from rangrez.vision import CallLog, MuseSamSegmenter, measure  # noqa: E402

# Shades a home dyer would actually get: teal, rose, mustard, lavender, olive, terracotta.
SHADES = [(55, -25, -12), (58, 35, 4), (72, 4, 52), (65, 14, -18), (52, -12, 30), (50, 32, 30)]


def one(args):
    name, k, card = args
    scene = Scene.load(name)
    truth = np.array(SHADES[(k + SCENES.index(name)) % len(SHADES)], dtype=float)
    tgt = np.array(SHADES[(k + 3) % len(SHADES)], dtype=float) if scene.target is not None else None
    jpeg, meta = render(scene, truth, target_lab=tgt, card=card, seed=100 * k + SCENES.index(name))
    log = CallLog()
    try:
        m = measure(jpeg, MuseSamSegmenter(log), with_target=tgt is not None)
    except Exception as exc:  # noqa: BLE001
        return name, card, truth, None, None, str(exc), len(log.calls)
    err = float(delta_e_2000(m.lab, truth))
    rel = None
    if tgt is not None and m.target_lab is not None:
        rel = abs(float(delta_e_2000(m.lab, m.target_lab)) - float(delta_e_2000(truth, tgt)))
    return name, card, truth, err, rel, "; ".join(m.flags), len(log.calls)


def main() -> None:
    jobs = [(n, k, card) for n in SCENES for k in range(2) for card in (True, False)]
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(one, jobs))
    for card in (True, False):
        errs = [r[3] for r in rows if r[1] == card and r[3] is not None]
        fails = [r for r in rows if r[1] == card and r[3] is None]
        print(f"\n== {'colour card' if card else 'white paper only'}: n={len(errs)} fail={len(fails)} "
              f"median dE00 {np.median(errs):.2f}  p90 {np.percentile(errs, 90):.2f}  max {max(errs):.2f}")
        for r in rows:
            if r[1] == card:
                rel = f" | strip-vs-target dE error {r[4]:.2f}" if r[4] is not None else ""
                e = f"{r[3]:.2f}" if r[3] is not None else "FAIL"
                print(f"  {r[0]:16s} truth {tuple(int(v) for v in r[2])} dE00 {e}{rel}  calls {r[6]}  {r[5][:90]}")


if __name__ == "__main__":
    main()
