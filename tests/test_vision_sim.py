import numpy as np
import pytest

from rangrez.colour import delta_e_2000
from rangrez.rig import GateError, Inventory, Recipe, RigLimits, check_recipe, execute
from rangrez.sim import VirtualRig, ground_truth_masks
from rangrez.vision import OfflineSegmenter, measure


@pytest.fixture
def rig():
    return VirtualRig(seed=3)


@pytest.mark.parametrize(
    "conc",
    [(0.0, 0.0, 0.0), (3.0, 0.5, 0.5), (0.4, 4.0, 1.0), (0.2, 0.3, 5.0), (2.0, 2.0, 2.0), (6.0, 1.0, 3.0)],
)
def test_measurement_matches_hidden_truth(rig, conc):
    shot = execute(rig, Recipe(conc))
    m = measure(shot.image_jpeg, OfflineSegmenter(ground_truth_masks()))
    assert m.ok, m.flags
    assert m.card_residual_de < 2.0
    assert float(delta_e_2000(m.lab, rig.last_swatch_lab)) < 1.5
    assert np.all(m.lab_sd < 0.5)


def test_measures_physical_target_cloth():
    target = np.array([55.0, -28.0, -12.0])
    rig = VirtualRig(seed=1, target_lab=target)
    shot = execute(rig, Recipe((1.0, 1.0, 1.0)))
    m = measure(shot.image_jpeg, OfflineSegmenter(ground_truth_masks()), with_target=True)
    assert float(delta_e_2000(m.target_lab, target)) < 1.5


def test_gate_refuses_unsafe_recipes():
    limits, inv = RigLimits(), Inventory()
    with pytest.raises(GateError):
        check_recipe(Recipe((11.0, 0.0, 0.0)), limits, inv)
    with pytest.raises(GateError):
        check_recipe(Recipe((-1.0, 0.0, 0.0)), limits, inv)
    with pytest.raises(GateError):
        check_recipe(Recipe((9.0, 9.0, 9.0)), limits, inv)  # more stock than bath
    with pytest.raises(GateError):
        check_recipe(Recipe((1.0, 1.0, 1.0), dip_seconds=5), limits, inv)
    with pytest.raises(GateError):
        check_recipe(Recipe((1.0, 1.0, 1.0)), limits, Inventory(stock_ml=[4.0, 250, 250]))
    plan = check_recipe(Recipe((2.0, 1.0, 0.5)), limits, inv)
    assert plan.dye_ml == pytest.approx((2.0, 1.0, 0.5))
    assert plan.water_ml == pytest.approx(16.5)


def test_execute_spends_inventory(rig):
    before = list(rig.inventory.stock_ml)
    execute(rig, Recipe((2.0, 0.0, 0.0)))
    assert rig.inventory.stock_ml[0] == pytest.approx(before[0] - 2.0)
    assert rig.inventory.strips == 59


def test_card_free_measurement_with_white_paper():
    rig = VirtualRig(seed=8, show_card=False, paper=True)
    shot = execute(rig, Recipe((2.5, 1.0, 0.8)))
    m = measure(shot.image_jpeg, OfflineSegmenter(ground_truth_masks(card=False, paper=True)))
    assert any("white-paper balance" in f for f in m.flags)
    assert float(delta_e_2000(m.lab, rig.last_swatch_lab)) < 3.0


def test_no_reference_at_all_still_measures_but_warns():
    rig = VirtualRig(seed=8, show_card=False, paper=False)
    shot = execute(rig, Recipe((2.5, 1.0, 0.8)))
    m = measure(shot.image_jpeg, OfflineSegmenter(ground_truth_masks(card=False, paper=False)))
    assert any("uncalibrated" in f for f in m.flags)
    assert m.ok
