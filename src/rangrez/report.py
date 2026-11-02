"""The lab-dip report: what a dye house sends a buyer to approve a shade.

One self-contained HTML page (prints to PDF): target against the approved dip, the colour
difference split the way spectrophotometer QC reports do (lightness, chroma, hue, with plain
words), a pass or fail against the tolerance, the recipe scaled for production, every dip with
its photo and SAM's masks, and a line for the buyer's signature.
"""

from __future__ import annotations

import base64
import html
import math
import time

import numpy as np

from .colour import delta_e_2000, lab_to_hex
from .profiles import PROFILE
from .rig import DYES, UNIT

LIQUOR_RATIO = 10  # litres of dye bath per kg of cotton (1:10), typical for exhaust dyeing


def difference(target, sample) -> dict:
    """CIELAB difference of sample from target: dL*, dC*, dH* and how a colourist would say it."""
    L1, a1, b1 = (float(v) for v in target)
    L2, a2, b2 = (float(v) for v in sample)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    dL, dC, da, db = L2 - L1, C2 - C1, a2 - a1, b2 - b1
    dH2 = max(da * da + db * db - dC * dC, 0.0)
    h1 = math.atan2(b1, a1)
    sign = 1.0 if (a1 * db - b1 * da) >= 0 else -1.0  # counter-clockwise on the hue circle
    dH = sign * math.sqrt(dH2)

    def size(x: float) -> str:
        x = abs(x)
        return "" if x < 0.5 else "slightly " if x < 1.5 else "" if x < 3 else "much "

    words = []
    if abs(dL) >= 0.5:
        words.append(f"{size(dL)}{'lighter' if dL > 0 else 'darker'}")
    if abs(dC) >= 0.5:
        words.append(f"{size(dC)}{'brighter' if dC > 0 else 'duller'}")
    if abs(dH) >= 0.5:
        hue_dir = math.degrees(h1 + (math.pi / 2 if dH > 0 else -math.pi / 2)) % 360
        towards = "redder" if hue_dir < 45 or hue_dir >= 315 else "yellower" if hue_dir < 135 else "greener" if hue_dir < 225 else "bluer"
        words.append(f"{size(dH)}{towards}")
    return {"dE00": float(delta_e_2000(np.array(sample), np.array(target))), "dL": dL, "dC": dC, "dH": dH,
            "words": ", ".join(words) or "no visible difference"}


def _img(jpeg: bytes, width: int) -> str:
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}" width="{width}" alt="">'


def render(campaign, target_photo: bytes | None = None, agent: str = "", api_calls: int | None = None) -> str:
    c = campaign
    best = c.best
    tol = c.tolerance
    passed = best is not None and best.delta_e <= tol
    diff = difference(c.target_lab, best.measurement.lab) if best else None
    when = time.strftime("%d %b %Y, %H:%M")
    ref = f"RZ-{c.campaign_id[-8:].upper()}"
    rows = []
    for s in c.shots:
        d = difference(c.target_lab, s.measurement.lab) if s.measurement.ok else None
        recipe = " · ".join(f"{v:g}" for v in s.recipe.conc)
        rows.append(f"""<tr><td>{s.index}</td><td>{_img(s.overlay_jpeg, 150)}</td>
<td><span class="sw" style="background:{lab_to_hex(s.measurement.lab)}"></span>{lab_to_hex(s.measurement.lab)}</td>
<td>{recipe}</td><td class="num">{"rejected" if d is None else f"{s.delta_e:.2f}"}</td>
<td>{"" if d is None else html.escape(d["words"])}</td></tr>""")
    if best is not None:
        bath = best.recipe.as_array() / PROFILE.stock_per_l * 1000  # mL of stock per litre of bath
        per_kg = bath * LIQUOR_RATIO
        prod = "".join(f"<tr><td>{html.escape(n)}</td><td class='num'>{v:g} {UNIT}</td><td class='num'>{m:.0f} mL</td></tr>"
                       for n, v, m in zip(DYES, best.recipe.conc, per_kg))
    verdict = ("PASS" if passed else "NOT YET") if best else "NO DIP"
    flags = sorted({f for s in c.shots for f in s.measurement.flags})
    notes = "".join(f"<li>{html.escape(f)}</li>" for f in flags) or "<li>No measurement warnings.</li>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Lab-dip report {ref}</title>
<style>
:root {{ --ink:#1b1b1d; --muted:#6b6862; --line:#e4e1dc; --pass:#1f7a4d; --fail:#a8452a; }}
body {{ font: 15px/1.45 -apple-system, "Segoe UI", Arial, sans-serif; color:var(--ink); background:#fff; max-width: 960px; margin: 0 auto; padding: 32px 20px; }}
h1 {{ font-size: 26px; margin: 0; }} h2 {{ font-size: 17px; margin: 28px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
.meta {{ color: var(--muted); }} .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
.chip {{ height: 120px; border-radius: 10px; border: 1px solid var(--line); }}
.verdict {{ font-size: 28px; font-weight: 700; color: {"var(--pass)" if passed else "var(--fail)"}; }}
table {{ border-collapse: collapse; width: 100%; }} td, th {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: middle; }}
th {{ color: var(--muted); font-weight: 600; }} .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sw {{ display:inline-block; width: 16px; height: 16px; border-radius: 4px; margin-right: 6px; vertical-align: -3px; border: 1px solid var(--line); }}
img {{ border-radius: 6px; max-width: 100%; height: auto; }} .sign {{ display:flex; gap: 40px; margin-top: 40px; }} .sign div {{ flex:1; border-top: 1px solid var(--ink); padding-top: 6px; color: var(--muted); }}
@media print {{ body {{ padding: 0; }} }}
</style></head><body>
<h1>Lab-dip report</h1>
<div class="meta">Rangrez self-driving dye lab · ref {ref} · {when} · {html.escape(PROFILE.description)}</div>

<h2>Shade</h2>
<div class="grid">
  <div><div class="chip" style="background:{lab_to_hex(c.target_lab)}"></div>
    <b>Target</b> {lab_to_hex(c.target_lab)} · L*a*b* {", ".join(f"{v:.1f}" for v in c.target_lab)}
    {("<div>" + _img(target_photo, 300) + "</div>") if target_photo else ""}</div>
  <div><div class="chip" style="background:{lab_to_hex(best.measurement.lab) if best else "#fff"}"></div>
    <b>Submitted dip {best.index if best else ""}</b> {lab_to_hex(best.measurement.lab) if best else ""}
    {("· L*a*b* " + ", ".join(f"{v:.1f}" for v in best.measurement.lab)) if best else ""}</div>
  <div><div class="verdict">{verdict}</div>
    {f'<div>ΔE00 <b>{diff["dE00"]:.2f}</b> against a tolerance of {tol}</div><div>ΔL* {diff["dL"]:+.2f} · ΔC* {diff["dC"]:+.2f} · ΔH* {diff["dH"]:+.2f}</div><div>The dip is <b>{html.escape(diff["words"])}</b> than the target.</div>' if diff else ""}</div>
</div>

{f'''<h2>Recipe</h2>
<table><tr><th>Dye</th><th class="num">Bath concentration</th><th class="num">Stock for 1 kg cotton (1:{LIQUOR_RATIO})</th></tr>{prod}</table>
<p class="meta">Dip {best.recipe.dip_seconds:.0f} s. Stock at {PROFILE.stock_per_l:g} {UNIT if UNIT != "drops" else "drops"} per litre. Scale-up assumes the same liquor ratio, temperature and fixation as the lab dip; confirm with a bulk trial.</p>''' if best else ""}

<h2>Dips</h2>
<table><tr><th>#</th><th>Photo, SAM 3.1 masks</th><th>Measured</th><th>Recipe ({" · ".join(DYES)}, {UNIT})</th><th class="num">ΔE00</th><th>Against the target</th></tr>
{"".join(rows)}</table>

<h2>How it was measured</h2>
<ul class="meta">
<li>Each dip is dried and photographed on white paper; SAM 3.1 segments the strip and the 24-patch colour card, and a colour-correction matrix turns pixels into CIELAB (D65). Uneven light is evened out using the paper.</li>
<li>Colour difference is CIEDE2000; ΔL*, ΔC*, ΔH* are CIELAB components (positive: lighter, brighter, counter-clockwise in hue).</li>
<li>Campaign run by {html.escape(agent or "Rangrez")}{f", {api_calls} live Meta API calls" if api_calls is not None else ""}; {len(c.shots)} dips; status {html.escape(c.status)}.</li>
{notes}
</ul>

<div class="sign"><div>Dyer</div><div>Buyer approval</div><div>Date</div></div>
</body></html>"""
