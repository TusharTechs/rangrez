"""Find and cache strip / target / paper masks for each scene with hosted SAM 3.1.

    uv run python eval/scene_masks.py [scene ...]
"""
import glob, io, os, sys
from dotenv import load_dotenv
load_dotenv(".env")
import numpy as np
from PIL import Image
from rangrez.photoreal import SCENE_DIR
from rangrez.vision import MuseSamSegmenter, _on_paper

sam = MuseSamSegmenter()


def save(mask, name, kind):
    Image.fromarray(mask.astype(np.uint8) * 255).save(f"{SCENE_DIR}/masks/{name}_{kind}.png")


names = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{SCENE_DIR}/*.jpg"))
for name in names:
    jpeg = open(f"{SCENE_DIR}/{name}.jpg", "rb").read()
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    papers = sam.segment(img, jpeg, "white paper") or sam.segment(img, jpeg, "white rectangle")
    paper = max(papers, key=lambda i: i.mask.sum()) if papers else None
    if name.endswith("_target"):  # a garment photographed on its own: the whole cloth is the target
        prompts = ("kurta", "shirt", "garment", "fabric")
    else:
        prompts = ("fabric", "cloth", "white cloth strip", "fabric strip")
    cloth, background = [], []
    for p in prompts:
        found = sam.segment(img, jpeg, p)
        # SAM also returns a bedsheet or tablecloth under the paper: keep it, tests must cope with it.
        background = [c for c in found if not _on_paper(c, paper)]
        cloth = [c for c in found if c not in background]
        if cloth:
            break
    if background:
        save(max(background, key=lambda i: i.mask.sum()).mask, name, "background")
    if not cloth:
        print(name, "MISSING cloth")
        continue
    if paper is not None:
        save(paper.mask, name, "paper")
    if name.endswith("_target"):
        tgt = max(cloth, key=lambda i: i.mask.sum())
        save(tgt.mask, name, "target")
        print(name, "target", tgt.box, "| paper", None if paper is None else paper.box, "| prompt", p)
        continue
    strip = max(cloth, key=lambda i: i.elongation)
    others = [c for c in cloth if c is not strip and not (c.mask & strip.mask).any()]
    save(strip.mask, name, "strip")
    if others:
        save(max(others, key=lambda i: i.mask.sum()).mask, name, "target")
    print(name, "strip", strip.box, "elong", round(strip.elongation, 1), "| paper", None if paper is None else paper.box, "| target", bool(others), "| prompt", p)
