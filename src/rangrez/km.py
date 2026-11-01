"""Spectral Kubelka-Munk model of reactive dye on cotton.

This is the colourist's model: each dye adds absorption (K/S) to the fibre in
proportion to how much of it fixes, and reflectance follows from total K/S.
The agent holds one of these as its hypothesis and refits it after every shot;
the simulator holds a different, hidden one as "reality".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .colour import delta_e_2000, xyz_to_lab

WAVELENGTHS = np.arange(400, 701, 20)  # 16 bands, nm

# CIE 1931 2-degree colour matching functions at 20 nm.
_CMF = np.array(
    [
        [0.0143, 0.0004, 0.0679], [0.1344, 0.0040, 0.6456], [0.3483, 0.0230, 1.7471],
        [0.2908, 0.0600, 1.6692], [0.0956, 0.1390, 0.8130], [0.0049, 0.3230, 0.2720],
        [0.0633, 0.7100, 0.0782], [0.2904, 0.9540, 0.0203], [0.5945, 0.9950, 0.0039],
        [0.9163, 0.8700, 0.0017], [1.0622, 0.6310, 0.0008], [0.8544, 0.3810, 0.0002],
        [0.4479, 0.1750, 0.0000], [0.1649, 0.0610, 0.0000], [0.0468, 0.0170, 0.0000],
        [0.0114, 0.0041, 0.0000],
    ]
)

# CIE D65 relative spectral power at 20 nm.
_D65 = np.array(
    [82.75, 93.43, 104.86, 117.81, 115.92, 109.35, 104.79, 104.41,
     100.00, 95.79, 90.01, 87.70, 83.70, 80.21, 78.28, 71.61]
)

_WEIGHTS = _CMF * _D65[:, None]
_NORM = 100.0 / _WEIGHTS[:, 1].sum()
SPECTRAL_WHITE = _WEIGHTS.sum(axis=0) * _NORM  # XYZ of a perfect white


def reflectance_to_xyz(R: np.ndarray) -> np.ndarray:
    return (np.asarray(R, dtype=float) @ _WEIGHTS) * _NORM


def reflectance_to_lab(R: np.ndarray) -> np.ndarray:
    return xyz_to_lab(reflectance_to_xyz(R), white=SPECTRAL_WHITE)


def ks_to_reflectance(ks: np.ndarray) -> np.ndarray:
    ks = np.maximum(np.asarray(ks, dtype=float), 0.0)
    return 1 + ks - np.sqrt(ks**2 + 2 * ks)


def reflectance_to_ks(R: np.ndarray) -> np.ndarray:
    R = np.clip(np.asarray(R, dtype=float), 1e-4, 1.0)
    return (1 - R) ** 2 / (2 * R)


def absorption_band(peak_nm: float, width_nm: float, strength: float) -> np.ndarray:
    """A Gaussian absorption band, as K/S per g/L of fixed dye."""
    return strength * np.exp(-0.5 * ((WAVELENGTHS - peak_nm) / width_nm) ** 2)


# Bleached cotton: near white, slightly warm.
COTTON_KS = reflectance_to_ks(np.linspace(0.80, 0.90, len(WAVELENGTHS)))


@dataclass
class DyeModel:
    """K/S of cotton = substrate + sum(gain_i * c_i * unit_ks_i)."""

    names: list[str]
    unit_ks: np.ndarray  # (n_dyes, n_bands) K/S per g/L in the bath, at gain 1
    substrate_ks: np.ndarray = field(default_factory=lambda: COTTON_KS.copy())
    gains: np.ndarray | None = None  # per-dye uptake multipliers, fitted from shots

    def __post_init__(self) -> None:
        if self.gains is None:
            self.gains = np.ones(len(self.names))

    def ks(self, conc: np.ndarray) -> np.ndarray:
        conc = np.maximum(np.asarray(conc, dtype=float), 0.0)
        return self.substrate_ks + (conc * self.gains) @ self.unit_ks

    def reflectance(self, conc: np.ndarray) -> np.ndarray:
        return ks_to_reflectance(self.ks(conc))

    def predict_lab(self, conc: np.ndarray) -> np.ndarray:
        return reflectance_to_lab(self.reflectance(conc))

    def copy(self) -> DyeModel:
        return DyeModel(list(self.names), self.unit_ks.copy(), self.substrate_ks.copy(), self.gains.copy())


def nominal_procion_cmy() -> DyeModel:
    """The supplier's nominal model of turquoise, fuchsia and lemon yellow.

    This is the agent's prior; real dye lots differ, which is why it refits.
    """
    unit_ks = np.stack(
        [
            absorption_band(640, 55, 0.55),  # turquoise absorbs red
            absorption_band(540, 40, 0.60),  # fuchsia absorbs green
            absorption_band(430, 35, 0.50),  # lemon yellow absorbs blue
        ]
    )
    return DyeModel(["turquoise", "fuchsia", "lemon yellow"], unit_ks)


def fit_gains(model: DyeModel, shots: list[tuple[np.ndarray, np.ndarray]]) -> DyeModel:
    """Refit per-dye uptake gains to measured (concentration, Lab) shots.

    Minimises CIEDE2000 between predicted and measured shades; returns a new model.
    """
    if not shots:
        return model.copy()
    concs = np.array([c for c, _ in shots])
    labs = np.array([lab for _, lab in shots])

    def residual(log_gains: np.ndarray) -> np.ndarray:
        trial = model.copy()
        trial.gains = np.exp(log_gains)
        pred = np.array([trial.predict_lab(c) for c in concs])
        # Mild prior pulling gains towards 1 keeps the fit sane with few shots.
        return np.concatenate([delta_e_2000(pred, labs), 0.5 * log_gains])

    result = least_squares(residual, np.log(model.gains), bounds=(np.log(0.2), np.log(5.0)))
    fitted = model.copy()
    fitted.gains = np.exp(result.x)
    return fitted


def solve_recipe(
    model: DyeModel,
    target_lab: np.ndarray,
    max_conc: float = 20.0,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Concentrations (g/L) that the model predicts will match target_lab.

    Returns (concentrations, predicted CIEDE2000).
    """
    n = len(model.names)
    starts = [start] if start is not None else []
    starts += [np.full(n, v) for v in (0.5, 2.0, 6.0)]
    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), 0.0, max_conc)
        r = least_squares(
            lambda c: np.atleast_1d(delta_e_2000(model.predict_lab(c), target_lab)),
            x0,
            bounds=(0.0, max_conc),
        )
        de = float(delta_e_2000(model.predict_lab(r.x), target_lab))
        if best is None or de < best[1]:
            best = (r.x, de)
    return best
