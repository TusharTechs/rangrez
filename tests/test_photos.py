"""Measurement on real-looking phone photos, against known truth.

Every photo is a Muse Image scene (desk lamp, daylight, tube light, warm bulb,
flash, crumpled cloth, no paper...) whose cotton is recoloured to a known true
colour under the scene's own uneven light, on paper of unknown whiteness. SAM 3.1
masked each scene once; the cached masks stand in for it here (see photoreal.py).
"""

import numpy as np
import pytest

from rangrez.colour import delta_e_2000
from rangrez.photoreal import Scene, SceneSegmenter, render, scene_names
from rangrez.rig import GateError, Inventory, Recipe, RigLimits, check_recipe, execute
from rangrez.sim import VirtualRig
from rangrez.vision import measure, target_from_photo

SCENES = [s for s in scene_names() if s not in {"phone_flash", "no_paper"}]  # usable light, paper in view
DYED = [(3.0, 0.5, 0.5), (0.4, 4.0, 1.0), (0.2, 0.3, 5.0), (2.0, 2.0, 2.0)]


def shoot(scene, conc, seed=3, **kw):
    rig = VirtualRig(seed=seed, scene=scene, **kw)
    m = measure(execute(rig, Recipe(conc)).image_jpeg, rig.segmenter(), with_target=rig.target_lab is not None)
    return rig, m


def de(a, b):
    return float(delta_e_2000(a, b))


@pytest.mark.parametrize("conc", DYED[:3])
@pytest.mark.parametrize("scene", SCENES)
def test_colour_card_photos_match_truth(scene, conc):
    rig, m = shoot(scene, conc)
    assert m.ok, m.flags
    assert de(m.lab, rig.last_swatch_lab) < 2.5
    assert np.all(m.lab_sd < 0.5)


def test_colour_card_accuracy_across_all_scenes():
    errs = [de(m.lab, rig.last_swatch_lab) for s in SCENES for rig, m in [shoot(s, c, seed=i) for i, c in enumerate(DYED)]]
    assert np.median(errs) < 1.0


@pytest.mark.parametrize("scene", SCENES)
def test_white_paper_only_photos(scene):
    for i, conc in enumerate(DYED[:2]):
        rig, m = shoot(scene, conc, seed=10 + i, card=False)
        assert m.ok, m.flags
        assert any("white-paper balance" in f for f in m.flags)
        assert de(m.lab, rig.last_swatch_lab) < 4.5  # the paper's true whiteness is unknown


def test_paper_only_still_compares_strip_and_target_fairly():
    """Absolute colour is off by the paper's unknown tint, but strip and target share it."""
    for seed, (strip, target) in enumerate([((2.0, 1.0, 0.5), (2.5, 0.8, 0.6)), ((0.5, 3.0, 1.0), (0.5, 3.0, 2.0))]):
        rig = VirtualRig(seed=seed, scene="with_target", card=False)
        rig.target_lab = rig.true_swatch_lab(Recipe(target))
        m = measure(execute(rig, Recipe(strip)).image_jpeg, rig.segmenter(), with_target=True)
        assert abs(de(m.lab, m.target_lab) - de(rig.last_swatch_lab, rig.target_lab)) < 1.0


def test_uneven_lamp_light_is_evened_out():
    rig, m = shoot("lamp_desk", (2.0, 2.0, 2.0))
    assert any(f.startswith("NOTE: evened out uneven light") for f in m.flags)
    assert de(m.lab, rig.last_swatch_lab) < 2.0


def test_hard_phone_shadow_on_the_paper_is_ignored():
    rig, m = shoot("shadowy", (0.4, 4.0, 1.0))
    assert m.ok, m.flags
    assert de(m.lab, rig.last_swatch_lab) < 1.5


def test_flash_photo_is_flagged_for_retake_and_card_falls_back_to_paper():
    _, m = shoot("phone_flash", (2.0, 2.0, 2.0))
    assert any("without flash" in f for f in m.flags)
    assert any("colour card unreadable" in f for f in m.flags)  # the hotspot bends the card's light
    assert m.ok


@pytest.mark.parametrize("scene", ["lamp_desk", "with_target", "kitchen_tube"])
def test_card_laid_over_the_strip_is_trimmed_not_fatal(scene):
    rig, m = shoot(scene, (1.5, 2.0, 0.5), card_overlap=True)
    assert m.ok, m.flags
    assert de(m.lab, rig.last_swatch_lab) < 2.5


def test_card_on_a_bare_table_without_paper():
    rig, m = shoot("no_paper", (3.0, 0.5, 0.5))
    assert m.ok, m.flags
    assert de(m.lab, rig.last_swatch_lab) < 1.0


def test_no_reference_at_all_still_measures_but_warns():
    _, m = shoot("no_paper", (2.5, 1.0, 0.8), card=False)
    assert any("uncalibrated" in f for f in m.flags)
    assert m.ok


def test_crumpled_strip():
    rig, m = shoot("crumpled_strip", (0.4, 4.0, 1.0))
    assert m.ok, m.flags
    assert de(m.lab, rig.last_swatch_lab) < 2.0


@pytest.mark.parametrize("card", [True, False])
def test_reads_a_customers_kurta(card):
    truth = np.array([55.0, -28.0, -12.0])
    scene = Scene.load("kurta_target")
    jpeg, masks = render(scene, target_lab=truth, card=card, seed=4)
    lab, mode = target_from_photo(jpeg, "kurta", SceneSegmenter(masks))
    assert mode == ("card" if card else "paper")
    # The kurta covers half the paper, so the light under it is interpolated from the paper
    # around it: a few dE00 even with the card. Paper only: the flagged ~5 dE00 bound.
    assert de(lab, truth) < (4.0 if card else 5.0)


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


def test_execute_spends_inventory():
    rig = VirtualRig(seed=3)
    before = list(rig.inventory.stock_ml)
    execute(rig, Recipe((2.0, 0.0, 0.0)))
    assert rig.inventory.stock_ml[0] == pytest.approx(before[0] - 2.0)
    assert rig.inventory.strips == 59


@pytest.mark.parametrize("scene", ["evening_warm", "vertical_strip"])
def test_target_photo_ignores_the_bedsheet_under_the_paper(scene):
    """SAM's "fabric" also finds the bedsheet; the cloth on the paper is the target."""
    truth = np.array([52.0, -12.0, 30.0])
    jpeg, masks = render(Scene.load(scene), truth, card=False, seed=2)
    assert "background" in masks
    lab, mode = target_from_photo(jpeg, "fabric", SceneSegmenter(masks))
    assert mode == "paper"
    assert de(lab, truth) < 5.0
