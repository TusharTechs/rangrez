"""The printed reference card that sits in every frame.

A 24-patch ColorChecker-style layout (4 rows x 6 columns). The reference
values are the published sRGB approximations of the Classic chart; print the
card in `hardware/colour_card.pdf` and every shot is colour-corrected against it,
so lighting drift and camera auto-white-balance cancel out.
"""

from __future__ import annotations

import numpy as np

from .colour import srgb_to_linear

PATCH_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black",
]

PATCH_SRGB_8BIT = np.array(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
        [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
        [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=float,
)

ROWS, COLS = 4, 6


def reference_linear() -> np.ndarray:
    """(24, 3) linear sRGB reference values of the card patches."""
    return srgb_to_linear(PATCH_SRGB_8BIT / 255.0)
