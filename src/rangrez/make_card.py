"""Render the printable 24-patch colour card used in every frame.

    uv run rangrez-card            # writes hardware/colour_card.png (300 dpi, 112 x 76 mm)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .card import COLS, PATCH_SRGB_8BIT, ROWS

DPI = 300
PATCH_MM, GAP_MM, BORDER_MM = 16, 2, 3


def render() -> Image.Image:
    mm = DPI / 25.4
    patch, gap, border = round(PATCH_MM * mm), round(GAP_MM * mm), round(BORDER_MM * mm)
    w = 2 * border + COLS * patch + (COLS - 1) * gap
    h = 2 * border + ROWS * patch + (ROWS - 1) * gap
    img = Image.new("RGB", (w, h), (8, 8, 8))
    draw = ImageDraw.Draw(img)
    for i, rgb in enumerate(PATCH_SRGB_8BIT.astype(int)):
        r, c = divmod(i, COLS)
        x, y = border + c * (patch + gap), border + r * (patch + gap)
        draw.rectangle([x, y, x + patch - 1, y + patch - 1], fill=tuple(int(v) for v in rgb))
    return img


def main() -> None:
    out = Path("hardware/colour_card.png")
    out.parent.mkdir(exist_ok=True)
    render().save(out, dpi=(DPI, DPI))
    print(f"wrote {out} — print at 100% scale on matte photo paper")


if __name__ == "__main__":
    main()
