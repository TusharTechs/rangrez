"""Turn a recorded campaign into a 1080p video clip for the demo.

Everything on screen comes from the recording: the customer's photo and SAM's reading
of it, each dip's photo with SAM's masks, the recipe Wonder chose, the measured error,
and the Meta API calls (model, purpose, seconds) made for that dip.

    uv run python eval/demo_reel.py recordings/<run>            # writes recordings/<run>/reel.mp4
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG, PANEL, INK, MUTED, ACCENT, PASS = "#111214", "#1a1b1e", "#eceae6", "#9a968f", "#e3a857", "#6fcf97"
SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((w, h))
    return img


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def header(d: ImageDraw.ImageDraw, meta: dict, right: str = "") -> None:
    d.text((60, 40), "Rangrez", font=font(BOLD, 44), fill=INK)
    d.text((262, 56), "a self-driving dye lab on Meta models", font=font(SANS, 26), fill=MUTED)
    tag = f"{meta.get('agent', '')}  ·  SAM 3.1  ·  {meta.get('rig', '')}"
    d.text((W - 60, 52), tag + (f"  ·  {right}" if right else ""), font=font(SANS, 24), fill=MUTED, anchor="ra")


def chart(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], des: list[float], tol: float) -> None:
    l, t, r, b = box
    d.rounded_rectangle(box, 14, fill=PANEL)
    top = max(max(des, default=10), tol * 2) * 1.1
    y = lambda v: b - 30 - (b - t - 110) * min(v, top) / top
    d.line([(l + 30, y(tol)), (r - 30, y(tol))], fill=PASS, width=2)
    d.text((l + 30, y(tol) + 6), f"pass {tol}", font=font(SANS, 20), fill=PASS)
    d.text((l + 24, t + 16), "error to target (CIEDE2000)", font=font(SANS, 20), fill=MUTED)
    if not des:
        return
    n = max(len(des), 2)
    xs = [l + 50 + (r - l - 100) * i / (n - 1) for i in range(len(des))]
    pts = [(x, y(v)) for x, v in zip(xs, des)]
    if len(pts) > 1:
        d.line(pts, fill=ACCENT, width=4)
    for (x, yy), v in zip(pts, des):
        d.ellipse([x - 8, yy - 8, x + 8, yy + 8], fill=ACCENT)
        d.text((x, yy - 16), f"{v:.1f}", font=font(BOLD, 22), fill=INK, anchor="mb")


def frames(run: Path) -> list[tuple[Image.Image, float]]:
    meta = json.loads((run / "meta.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    tol = float(meta.get("tolerance", 1.5))
    out: list[tuple[Image.Image, float]] = []

    img, d = canvas()
    header(d, meta)
    d.text((W // 2, 420), "Match this colour on real cotton", font=font(BOLD, 72), fill=INK, anchor="mm")
    d.rounded_rectangle([W // 2 - 160, 500, W // 2 + 160, 620], 18, fill=meta["target_hex"])
    d.text((W // 2, 680), f"target {meta['target_hex']}  ·  pass at ΔE00 ≤ {tol}", font=font(SANS, 32), fill=MUTED, anchor="mm")
    out.append((img, 2.5))

    if (run / "target.jpg").exists():
        tj = json.loads((run / "target.json").read_text()) if (run / "target.json").exists() else {}
        img, d = canvas()
        header(d, meta, "reading the customer's photo")
        photo = fit(Image.open(run / "target.jpg").convert("RGB"), 1180, 860)
        img.paste(photo, (60, 140))
        x = 60 + photo.width + 60
        d.text((x, 160), "SAM 3.1 finds the", font=font(SANS, 34), fill=MUTED)
        d.text((x, 205), f"“{tj.get('concept', 'garment')}”", font=font(BOLD, 48), fill=INK)
        d.rounded_rectangle([x, 290, x + 300, 440], 16, fill=tj.get("hex", meta["target_hex"]))
        d.text((x, 470), tj.get("hex", ""), font=font(BOLD, 40), fill=INK)
        d.text((x, 530), f"calibrated with the {tj.get('calibration', '')}", font=font(SANS, 28), fill=MUTED)
        calls = [e for e in events if e.get("kind") == "api" and e.get("at", 0) <= events_start(events)]
        for i, c in enumerate(calls[-6:]):
            d.text((x, 620 + 40 * i), f"{c['model']}  {c['purpose']}  {c['seconds']}s", font=font(MONO, 22), fill=MUTED)
        out.append((img, 3.5))

    des: list[float] = []
    since = events_start(events)
    pending_calls: list[dict] = []
    note = ""
    for e in events:
        if e.get("at", 0) < since:
            continue
        kind = e.get("kind")
        if kind == "api":
            pending_calls.append(e)
        elif kind in ("thought", "note") and e.get("text"):
            note = e["text"]
        elif kind == "shot":
            de = e.get("delta_e")
            des.append(float(de) if de is not None else float("nan"))
            img, d = canvas()
            header(d, meta, f"dip {e['index']}")
            sam = run / f"dip{e['index']:02d}_sam.jpg"
            if sam.exists():
                photo = fit(Image.open(sam).convert("RGB"), 1150, 750)
                img.paste(photo, (60, 140))
            x = 1250
            d.text((x, 140), f"Dip {e['index']}", font=font(BOLD, 56), fill=INK)
            d.text((x, 215), e.get("recipe", "")[:80], font=font(SANS, 22), fill=MUTED)
            d.rounded_rectangle([x, 270, x + 150, 400], 14, fill=meta["target_hex"])
            d.rounded_rectangle([x + 170, 270, x + 320, 400], 14, fill=e.get("hex", "#000000"))
            d.text((x + 75, 415), "target", font=font(SANS, 20), fill=MUTED, anchor="ma")
            d.text((x + 245, 415), "this dip", font=font(SANS, 20), fill=MUTED, anchor="ma")
            ok = de is not None and de <= tol
            d.text((x + 350, 270), f"{de:.2f}" if de is not None else "—", font=font(BOLD, 96), fill=PASS if ok else ACCENT)
            d.text((x + 352, 385), "ΔE00 · matched" if ok else "ΔE00", font=font(SANS, 24), fill=MUTED)
            chart(d, (x, 460, W - 60, 760), [v for v in des if math.isfinite(v)], tol)
            for i, c in enumerate(pending_calls[-4:]):
                d.text((60, 960 + 28 * i), f"{c['model']:<18} {c['purpose'][:48]:<48} {c['seconds']:>5}s",
                       font=font(MONO, 20), fill=MUTED if c.get("ok", True) else "#ff7b72")
            if note:
                wrap(d, (x, 790), note, 600, font(SANS, 22), MUTED, lines=4)
            d.text((60, 920), f"Live Meta API calls for this dip: {len(pending_calls)}", font=font(SANS, 24), fill=INK)
            out.append((img, 3.2))
            pending_calls, note = [], ""
        elif kind == "status" and e.get("status") not in ("running", "waiting", None):
            img, d = canvas()
            header(d, meta, e["status"])
            words = {"matched": "Matched on cotton", "out_of_gamut": "Out of this dye set's reach",
                     "escalated": "Handed to a human", "stopped": "Stopped"}
            d.text((W // 2, 300), words.get(e["status"], e["status"]), font=font(BOLD, 80), fill=PASS if e["status"] == "matched" else ACCENT, anchor="mm")
            seq = "  →  ".join(f"{v:.1f}" for v in des if math.isfinite(v))
            d.text((W // 2, 420), f"ΔE00  {seq}", font=font(BOLD, 48), fill=INK, anchor="mm")
            api = [x for x in events if x.get("kind") == "api"]
            secs = events[-1]["at"] - events[0]["at"] if events else 0
            d.text((W // 2, 510), f"{len(des)} dips · {len(api)} live Meta API calls ({sum(not a.get('ok', True) for a in api)} failed) · {secs / 60:.1f} min",
                   font=font(SANS, 34), fill=MUTED, anchor="mm")
            if e.get("reason"):
                wrap(d, (360, 600), e["reason"], 1200, font(SANS, 28), MUTED, lines=4)
            chart(d, (360, 760, W - 360, 1030), [v for v in des if math.isfinite(v)], tol)
            out.append((img, 4.5))
    return out


def events_start(events: list[dict]) -> float:
    for e in events:
        if e.get("kind") == "start":
            return e["at"]
    return 0.0


def wrap(d: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, f, fill, lines: int = 3) -> None:
    words, line, out = text.split(), "", []
    for w in words:
        if d.textlength(line + " " + w, font=f) > width and line:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    out.append(line)
    for i, ln in enumerate(out[:lines]):
        d.text((xy[0], xy[1] + i * (f.size + 8)), ln + ("…" if i == lines - 1 and len(out) > lines else ""), font=f, fill=fill)


def main() -> None:
    run = Path(sys.argv[1])
    out = run / "reel.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        lines = []
        shots = frames(run)
        for i, (img, secs) in enumerate(shots):
            p = Path(tmp) / f"{i:03d}.png"
            img.save(p)
            lines += [f"file '{p}'", f"duration {secs}"]
        lines.append(f"file '{Path(tmp) / f'{len(shots) - 1:03d}.png'}'")
        (Path(tmp) / "list.txt").write_text("\n".join(lines))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(Path(tmp) / "list.txt"),
                        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20", str(out)], check=True)
        shots[0][0].save(run / "reel_cover.png")
    print(out)


if __name__ == "__main__":
    main()
