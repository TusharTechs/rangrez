import numpy as np
import pytest

from rangrez.colour import delta_e_2000
from rangrez.km import (
    fit_gains,
    ks_to_reflectance,
    nominal_procion_cmy,
    reflectance_to_ks,
    reflectance_to_lab,
    solve_recipe,
)


def test_ks_reflectance_round_trip():
    R = np.linspace(0.05, 0.95, 16)
    assert ks_to_reflectance(reflectance_to_ks(R)) == pytest.approx(R, abs=1e-9)


def test_perfect_white_is_L100():
    assert reflectance_to_lab(np.ones(16)) == pytest.approx([100.0, 0.0, 0.0], abs=1e-6)


def test_undyed_cotton_is_near_white():
    L, a, b = nominal_procion_cmy().predict_lab(np.zeros(3))
    assert L > 90 and abs(a) < 3 and 0 < b < 6


def test_each_dye_moves_hue_the_expected_way():
    m = nominal_procion_cmy()
    turquoise = m.predict_lab([4, 0, 0])
    fuchsia = m.predict_lab([0, 4, 0])
    yellow = m.predict_lab([0, 0, 4])
    assert turquoise[1] < 0 and turquoise[2] < 0  # green-blue
    assert fuchsia[1] > 20  # red-purple
    assert yellow[2] > 30  # yellow


def test_more_dye_is_darker():
    m = nominal_procion_cmy()
    assert m.predict_lab([1, 1, 1])[0] > m.predict_lab([5, 5, 5])[0]


def test_solve_recipe_hits_a_reachable_target():
    m = nominal_procion_cmy()
    target = m.predict_lab([2.5, 0.8, 1.2])
    conc, de = solve_recipe(m, target)
    assert de < 0.5
    assert delta_e_2000(m.predict_lab(conc), target) < 0.5


def test_fit_gains_recovers_hidden_uptake():
    truth = nominal_procion_cmy()
    truth.gains = np.array([1.6, 0.7, 1.2])
    shots = [(c, truth.predict_lab(c)) for c in ([2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 1], [3, 1, 0.5])]
    fitted = fit_gains(nominal_procion_cmy(), shots)
    assert fitted.gains == pytest.approx(truth.gains, rel=0.1)
