"""Colour science: sRGB <-> CIELAB, CIEDE2000, and camera colour correction.

Everything colour-related that has a right answer lives here, so the agent
never has to reason about it in prose.
"""

from __future__ import annotations

import numpy as np

# D65 reference white for the sRGB path (CIE 1931 2 deg), Y normalised to 100.
D65_WHITE = np.array([95.047, 100.0, 108.883])

# Linear sRGB -> XYZ (D65), IEC 61966-2-1.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> linear light in [0, 1]."""
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def linear_rgb_to_xyz(lin: np.ndarray) -> np.ndarray:
    return np.asarray(lin, dtype=float) @ _RGB_TO_XYZ.T * 100.0


def xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz, dtype=float) / 100.0) @ _XYZ_TO_RGB.T


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = D65_WHITE) -> np.ndarray:
    t = np.asarray(xyz, dtype=float) / white
    delta = 6 / 29
    f = np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray = D65_WHITE) -> np.ndarray:
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    delta = 6 / 29
    f = np.stack([fx, fy, fz], axis=-1)
    t = np.where(f > delta, f**3, 3 * delta**2 * (f - 4 / 29))
    return t * white


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIELAB (D65)."""
    return xyz_to_lab(linear_rgb_to_xyz(srgb_to_linear(rgb)))


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(xyz_to_linear_rgb(lab_to_xyz(lab)))


def hex_to_lab(hex_colour: str) -> np.ndarray:
    h = hex_colour.lstrip("#")
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return srgb_to_lab(rgb)


def lab_to_hex(lab: np.ndarray) -> str:
    rgb = np.round(lab_to_srgb(lab) * 255).astype(int)
    return "#" + "".join(f"{int(v):02x}" for v in np.clip(rgb, 0, 255))


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), kL = kC = kH = 1."""
    lab1 = np.asarray(lab1, dtype=float)
    lab2 = np.asarray(lab2, dtype=float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2
    hsum = h1p + h2p
    hp_bar = np.where(
        C1p * C2p == 0,
        hsum,
        np.where(
            np.abs(h1p - h2p) <= 180,
            hsum / 2,
            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
        ),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )
    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    R_C = 2 * np.sqrt(Cp_bar**7 / (Cp_bar**7 + 25**7))
    S_L = 1 + 0.015 * (Lp_bar - 50) ** 2 / np.sqrt(20 + (Lp_bar - 50) ** 2)
    S_C = 1 + 0.045 * Cp_bar
    S_H = 1 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2 * d_theta)) * R_C
    return np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


def fit_colour_correction(measured_lin: np.ndarray, reference_lin: np.ndarray) -> np.ndarray:
    """Least-squares affine map (4x3) from camera linear RGB to reference linear sRGB.

    measured_lin, reference_lin: (n_patches, 3). Returns M such that
    [rgb, 1] @ M ~= reference.
    """
    X = np.hstack([measured_lin, np.ones((len(measured_lin), 1))])
    M, *_ = np.linalg.lstsq(X, reference_lin, rcond=None)
    return M


def apply_colour_correction(M: np.ndarray, lin: np.ndarray) -> np.ndarray:
    lin = np.asarray(lin, dtype=float)
    X = np.concatenate([lin, np.ones(lin.shape[:-1] + (1,))], axis=-1)
    return X @ M
