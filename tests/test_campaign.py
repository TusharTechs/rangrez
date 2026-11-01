import numpy as np
import pytest

from rangrez.agent import best_delta_e, dispatch, run_deterministic
from rangrez.campaign import Campaign
from rangrez.colour import delta_e_2000, hex_to_lab
from rangrez.km import nominal_procion_cmy
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig, ground_truth_masks
from rangrez.vision import OfflineSegmenter


def make_campaign(target_lab, seed=11, **kw):
    rig = VirtualRig(seed=seed)
    return Campaign(np.asarray(target_lab, dtype=float), rig, OfflineSegmenter(ground_truth_masks()), **kw)


# Shades reachable with turquoise / fuchsia / lemon yellow on cotton.
TARGETS = {
    "teal": lambda rig: rig.true_swatch_lab(Recipe((3.2, 0.6, 0.9))),
    "maroon": lambda rig: rig.true_swatch_lab(Recipe((0.6, 5.0, 2.0))),
    "olive": lambda rig: rig.true_swatch_lab(Recipe((0.9, 0.7, 4.5))),
    "navy": lambda rig: rig.true_swatch_lab(Recipe((6.0, 2.5, 0.4))),
}


@pytest.mark.parametrize("name", list(TARGETS))
def test_campaign_converges_on_reachable_shades(name):
    target = TARGETS[name](VirtualRig(seed=99))
    c = make_campaign(target)
    first_guess = nominal_procion_cmy().predict_lab  # the supplier model alone
    run_deterministic(c)
    assert c.status == "matched", c.state()
    assert best_delta_e(c) <= 1.5
    assert len(c.shots) <= 6
    # The supplier model alone would have missed: the loop is doing the work.
    assert float(delta_e_2000(c.shots[0].measurement.lab, target)) > best_delta_e(c)
    assert first_guess is not None


def test_out_of_gamut_target_escalates_not_loops():
    c = make_campaign(hex_to_lab("#ff2a00"), max_shots=10)  # saturated orange-red: not reachable cleanly
    run_deterministic(c)
    assert c.status in {"escalated", "matched", "finished"}
    assert len(c.shots) <= 10


def test_run_shot_refuses_unsafe_recipe_without_dispensing():
    c = make_campaign([50, 0, 0])
    out = dispatch(c, "run_shot", {"recipe_g_per_l": {"turquoise": 50, "fuchsia": 0, "lemon yellow": 0}, "rationale": "test"})
    assert "refused" in out
    assert c.shots == []
    assert c.rig.inventory.strips == 60


def test_state_is_json_ready():
    import json

    c = make_campaign([50, 10, 10])
    dispatch(c, "run_shot", {"recipe_g_per_l": {"turquoise": 1, "fuchsia": 1, "lemon yellow": 1}, "rationale": "probe"})
    json.dumps(c.state())
