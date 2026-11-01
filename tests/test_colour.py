import numpy as np
import pytest

from rangrez.colour import (
    apply_colour_correction,
    delta_e_2000,
    fit_colour_correction,
    hex_to_lab,
    lab_to_hex,
    lab_to_srgb,
    srgb_to_lab,
    srgb_to_linear,
)

# Sharma, Wu & Dalal (2005) CIEDE2000 test pairs.
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA)
def test_ciede2000_reference_pairs(lab1, lab2, expected):
    assert delta_e_2000(np.array(lab1), np.array(lab2)) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_is_symmetric_and_zero_on_identity():
    a, b = np.array([55.0, 30.0, -12.0]), np.array([48.0, 10.0, 22.0])
    assert delta_e_2000(a, b) == pytest.approx(delta_e_2000(b, a))
    assert delta_e_2000(a, a) == pytest.approx(0.0)


def test_white_is_L100():
    assert srgb_to_lab(np.array([1.0, 1.0, 1.0])) == pytest.approx([100.0, 0.0, 0.0], abs=0.01)


def test_srgb_lab_round_trip():
    rgb = np.array([0.2, 0.55, 0.7])
    assert lab_to_srgb(srgb_to_lab(rgb)) == pytest.approx(rgb, abs=1e-6)


def test_hex_round_trip():
    assert lab_to_hex(hex_to_lab("#1f8a8a")) == "#1f8a8a"


def test_colour_correction_recovers_affine_distortion():
    rng = np.random.default_rng(0)
    ref = srgb_to_linear(rng.uniform(0.05, 0.95, size=(24, 3)))
    distortion = np.array([[0.8, 0.1, 0.0], [0.05, 0.9, 0.05], [0.0, 0.1, 1.1]])
    camera = ref @ distortion.T + np.array([0.02, 0.01, 0.03])
    M = fit_colour_correction(camera, ref)
    assert apply_colour_correction(M, camera) == pytest.approx(ref, abs=1e-9)
