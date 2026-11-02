"""A labelled simulation of a home dye run, cut from a recorded household campaign.

The kitchen scenes (counter, dropper, dipping, phone) are Muse Image pictures in assets/sim/.
Everything else comes from the recording: the customer's photo and SAM 3.1's reading of it,
each recipe Muse Wonder chose and why, each dip's photo with SAM's masks, the measured error,
and the count of Meta API calls. The dye chemistry is simulated, and every frame says so.

    uv run python eval/home_reel.py recordings/<run> --true-final 1.50   # writes <run>/home_reel.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H, FPS, FADE = 1920, 1080, 30, 0.45
SIM = Path(__file__).parent.parent / "assets" / "sim"
SERIF = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
SANS = "/System/Library/Fonts/Avenir Next.ttc"
INK, CREAM, DYE = "#fbf6ec", "#efe5d3", "#2f7f8f"
NAMES = {"sam-3.1": "SAM 3.1", "muse-wonder-1.0": "Muse Wonder", "muse-spark-1.3": "Muse Spark", "muse-image-1.0": "Muse Image"}


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.load_default(size)


F_TITLE, F_HEAD, F_BODY, F_SMALL = font(SERIF, 76), font(SERIF, 46), font(SANS, 30, 0), font(SANS, 24, 0)
F_PILL = font(SANS, 22, 2)


def cover(img: Image.Image, zoom: float, dx: float, dy: float) -> Image.Image:
    """Crop img to 16:9 at the given zoom and offset (fractions of the slack) and scale to the frame."""
    iw, ih = img.size
    cw = min(iw, ih * W / H) / zoom
    ch = cw * H / W
    x = (iw - cw) * (0.5 + dx / 2)
    y = (ih - ch) * (0.5 + dy / 2)
    return img.resize((W, H), Image.BICUBIC, box=(x, y, x + cw, y + ch))


def contain(img: Image.Image) -> Image.Image:
    """The whole photo, on a blurred, darkened copy of itself: a dip photo is evidence, so nothing is cropped."""
    bg = ImageEnhance.Brightness(cover(img, 1.0, 0, 0).filter(ImageFilter.GaussianBlur(36))).enhance(0.45)
    fg = img.copy()
    fg.thumbnail((W, H - 40))
    bg.paste(fg, ((W - fg.width) // 2, 20))
    return bg


def arrow(d: ImageDraw.ImageDraw, x: float, y: float, colour: str, size: float = 1.0) -> None:
    d.line((x, y, x + 26 * size, y), fill=colour, width=int(3 * size))
    d.polygon([(x + 34 * size, y), (x + 22 * size, y - 8 * size), (x + 22 * size, y + 8 * size)], fill=colour)


# Wonder's notes use the tools' names; the video says them in words.
PLAIN = [("model_inverse", "the refitted dye model"), ("colourist_correction", "a colourist's correction"),
         ("glimmer/lab_memory", "Glimmer or lab memory"), ("lab_memory", "lab memory"), ("dE00", "ΔE00"), ("2.61-dE", "2.61 ΔE00"),
         ("b*", "blue–yellow balance"), ("shot", "dip"), ("Best within 6", "Best dip within 6")]


def plain(text: str) -> str:
    for a, b in PLAIN:
        text = text.replace(a, b)
    return text


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def badge(o: ImageDraw.ImageDraw) -> None:
    text = "SIMULATION  ·  scenes by Muse Image, simulated dye chemistry"
    w = o.textlength(text, font=F_PILL)
    o.rounded_rectangle((W - w - 76, 36, W - 36, 84), 24, fill=(20, 14, 8, 170))
    o.ellipse((W - w - 58, 53, W - w - 44, 67), fill="#e4572e")
    o.text((W - w - 34, 47), text, font=F_PILL, fill=INK)


def caption(o: ImageDraw.ImageDraw, head: str, body: str = "", chips: list[str] | None = None, target: str | None = None) -> None:
    for i in range(320):  # a soft dark band at the bottom for the words
        o.line((0, H - 320 + i, W, H - 320 + i), fill=(12, 9, 6, int(205 * (i / 320) ** 1.4)))
    lines = wrap(o, body, F_BODY, 1180)[:3] if body else []
    y = H - 140 - 44 * len(lines)
    o.text((80, y), head, font=F_HEAD, fill=INK)
    for k, line in enumerate(lines):
        o.text((80, y + 66 + k * 42), line, font=F_BODY, fill=CREAM)
    if chips is not None:  # the tape so far, ending at the target
        x = W - 80 - 70 * (len(chips) + 1) - 40
        for k, hexc in enumerate(chips):
            o.rounded_rectangle((x + k * 70, H - 150, x + k * 70 + 62, H - 88), 8, fill=hexc)
            o.text((x + k * 70 + 31, H - 78), str(k + 1), font=F_SMALL, fill=CREAM, anchor="mt")
        tx = x + len(chips) * 70 + 40
        if target:
            arrow(o, tx - 36, H - 119, CREAM, 0.8)
            o.rounded_rectangle((tx, H - 156, tx + 74, H - 82), 10, fill=target, outline=INK, width=3)
            o.text((tx + 37, H - 72), "target", font=F_SMALL, fill=CREAM, anchor="mt")


def card(lines: list[tuple[str, ImageFont.FreeTypeFont, str]], chips: list[str] | None = None, target: str | None = None) -> Image.Image:
    img = Image.new("RGB", (W, H), "#17120d")
    d = ImageDraw.Draw(img)
    total = sum(f.size * 1.5 for _, f, _ in lines) + (150 if chips else 0)
    y = (H - total) / 2
    if chips:
        x = (W - (len(chips) * 96 + 140)) / 2
        for k, hexc in enumerate(chips):
            d.rounded_rectangle((x + k * 96, y, x + k * 96 + 86, y + 96), 12, fill=hexc)
        arrow(d, x + len(chips) * 96 + 8, y + 48, CREAM, 1.2)
        d.rounded_rectangle((x + len(chips) * 96 + 60, y - 6, x + len(chips) * 96 + 168, y + 102), 14, fill=target, outline=INK, width=4)
        y += 150
    for text, f, colour in lines:
        for line in wrap(d, text, f, 1640):
            d.text((W / 2, y), line, font=f, fill=colour, anchor="mt")
            y += f.size * 1.5
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--true-final", type=float, help="the last dip's true error, known only because the chemistry is simulated")
    args = ap.parse_args()
    run = Path(args.run)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    target = json.loads((run / "target.json").read_text())
    dispenses = [e for e in events if e["kind"] == "dispense"]
    shots = [e for e in events if e["kind"] == "shot"]
    status = next((e for e in reversed(events) if e["kind"] == "status"), {})
    calls = [e for e in events if e["kind"] == "api"]
    models = sorted({e.get("model", "") for e in calls} - {""})

    def drops(recipe: str) -> str:
        return " · ".join(f"{int(float(n))} {d.strip()}" for d, n in re.findall(r"([a-z ]+?) ([\d.]+) drops", recipe))

    def brief(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return plain(text)

    # (image or card, seconds, head, body, zoom from/to, pan, show tape)
    plan = []
    plan.append((card([("A home dye run", F_TITLE, INK),
                       (("Simulated. The kitchen scenes are Muse Image pictures and the dye chemistry is simulated. "
                        "Every reading, recipe and decision is the real Rangrez pipeline, live on SAM 3.1 and Muse Wonder."), F_BODY, CREAM)]),
                 4.5, "", "", (1, 1), (0, 0), False))
    plan.append((Image.open(SIM / "counter.jpg"), 4.2, "Three household colours, a dropper, white cotton, a phone",
                 "No lab, no spectrophotometer: what a small job worker already has.", (1.0, 1.08), (-0.3, 0.2), False))
    plan.append((Image.open(run / "target.jpg"), 4.6, f"The customer's kurta: SAM 3.1 reads {target['hex']}",
                 f"Photographed on white paper beside the printed card; calibrated with the {target['calibration']}.", (1.0, 1.1), (0, -0.2), "target"))
    for i, (disp, shot) in enumerate(zip(dispenses, shots), start=1):
        head = f"Dip {i} · Muse Wonder: {drops(disp['recipe'])} drops"
        if i == 1:
            plan.append((Image.open(SIM / "dropper.jpg"), 3.4, head, brief(disp.get("rationale", "")), (1.0, 1.12), (0.3, -0.2), False))
            plan.append((Image.open(SIM / "dip.jpg"), 3.0, "Dip the cotton strip for 90 seconds, then dry it",
                         "Rangrez reads the drops aloud and listens for “done”, in English or Hindi.", (1.05, 1.15), (0, 0.3), False))
            plan.append((Image.open(SIM / "phone.jpg"), 3.0, "Photograph it on the paper, beside the card",
                         "The phone is the only instrument.", (1.0, 1.1), (0.4, 0.2), False))
        else:
            plan.append((None, 0, drops(disp["recipe"]), brief(disp.get("rationale", "")), None, None, i))
        de = shot["delta_e"]
        plan.append((contain(Image.open(run / f"dip{i:02d}_sam.jpg")), 3.6 if i == 1 else 3.4,
                     f"Dip {i} · SAM 3.1 measures {shot['hex']} · ΔE00 {de:.2f}",
                     "" if i == 1 else None, (1.0, 1.04), (0, 0), i))
    final = shots[-1]["delta_e"]
    truth = f", {args.true_final:.2f} in the simulation's hidden truth" if args.true_final else ""
    plan.append((card([(f"{status.get('status', 'finished').capitalize()} in {len(shots)} dips", F_TITLE, INK),
                       ((f"Measured ΔE00 {final:.2f}{truth}. {len(calls)} Meta API calls ({', '.join(NAMES.get(m, m) for m in models)}), "
                        f"{sum(not e.get('ok', True) for e in calls)} failed."), F_BODY, CREAM),
                       ("Not yet run with real dye on real cloth: this shows the workflow, not a field result.", F_SMALL, "#b9ab95")],
                      chips=[s["hex"] for s in shots], target=target["hex"]), 6.5, "", "", (1, 1), (0, 0), False))

    # Fold each "recipe" step for dips 2+ into the photo that follows: its head shows on top, the measurement below.
    shots_plan = []
    pending = None
    for p in plan:
        if p[0] is None:
            pending = p
            continue
        if pending is not None:
            img, secs, head, _, zoom, pan, tape = p
            p = (img, secs, head, f"Muse Wonder chose {pending[2]} drops. {pending[3]}", zoom, pan, tape)
            pending = None
        shots_plan.append(p)

    out = run / "home_reel.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    prev = None
    for img, secs, head, body, zoom, pan, tape in shots_plan:
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        o = ImageDraw.Draw(overlay)
        if head:
            chips = None
            if tape == "target":
                chips = []
            elif tape is not False and isinstance(tape, int):
                chips = [s["hex"] for s in shots[:tape]]
            caption(o, head, body or "", chips, target["hex"] if chips is not None else None)
        badge(o)
        n = int(secs * FPS)
        last = None
        for k in range(n):
            t = k / max(n - 1, 1)
            ease = t * t * (3 - 2 * t)
            z = zoom[0] + (zoom[1] - zoom[0]) * ease
            base = cover(img, z, pan[0] * ease, pan[1] * ease).convert("RGBA")
            frame = Image.alpha_composite(base, overlay).convert("RGB")
            if prev is not None and k < FADE * FPS:
                frame = Image.blend(prev, frame, k / (FADE * FPS))
            ff.stdin.write(np.asarray(frame).tobytes())
            last = frame
        prev = last
    ff.stdin.close()
    ff.wait()
    print(out, f"{sum(p[1] for p in shots_plan):.1f}s")


if __name__ == "__main__":
    main()
