"""Find and cache strip / target / paper masks for each scene with hosted SAM 3.1."""
import glob, io, os
from dotenv import load_dotenv
load_dotenv(".env")
import numpy as np
from PIL import Image
from rangrez.vision import MuseSamSegmenter

sam = MuseSamSegmenter()
for path in sorted(glob.glob("eval/scenes/*.jpg")):
    name = os.path.basename(path)[:-4]
    jpeg = open(path, "rb").read()
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    papers = sam.segment(img, jpeg, "white paper") or sam.segment(img, jpeg, "white rectangle")
    paper = max(papers, key=lambda i: i.mask.sum()) if papers else None
    cloth = []
    for p in ("fabric", "cloth", "white cloth strip", "fabric strip"):
        cloth = [c for c in sam.segment(img, jpeg, p) if paper is None or c.mask.sum() < 0.6 * paper.mask.sum()]
        if cloth:
            break
    if not cloth or paper is None:
        print(name, "MISSING", "paper" if paper is None else "", "cloth" if not cloth else "")
        continue
    strip = max(cloth, key=lambda i: i.elongation)
    others = [c for c in cloth if c is not strip and not (c.mask & strip.mask).any()]
    Image.fromarray(strip.mask.astype(np.uint8) * 255).save(f"eval/scenes/masks/{name}_strip.png")
    Image.fromarray(paper.mask.astype(np.uint8) * 255).save(f"eval/scenes/masks/{name}_paper.png")
    if others:
        tgt = max(others, key=lambda i: i.mask.sum())
        Image.fromarray(tgt.mask.astype(np.uint8) * 255).save(f"eval/scenes/masks/{name}_target.png")
    print(name, "strip", strip.box, "elong", round(strip.elongation, 1), "| paper", paper.box, "| target", bool(others), "| prompt", p)
