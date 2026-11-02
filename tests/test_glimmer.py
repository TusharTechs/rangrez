"""Glimmer's place in the loop: it proposes the first dip; the lab never depends on it."""

import numpy as np

from rangrez import glimmer
from rangrez.agent import run_deterministic
from rangrez.campaign import Campaign
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig


def test_parse_reads_the_recipe_and_clips_it():
    assert np.allclose(glimmer.parse('{"turquoise": 3.2, "fuchsia": 0.6, "lemon yellow": 0.9}'), [3.2, 0.6, 0.9])
    assert np.allclose(glimmer.parse('Sure: {"turquoise": 40, "fuchsia": -1}'), [10.0, 0.0, 0.0])
    assert glimmer.parse("I think more blue") is None


def test_prompt_and_answer_round_trip():
    assert "L* 55.0" in glimmer.prompt([55, -20, -10])
    assert np.allclose(glimmer.parse(glimmer.answer([1.234, 0, 5])), [1.23, 0, 5])


def test_first_dip_uses_glimmer_when_configured(monkeypatch):
    rig = VirtualRig(seed=2, scene="kitchen_tube")
    truth = (3.2, 0.6, 0.9)
    target = rig.true_swatch_lab(Recipe(truth))
    monkeypatch.setattr(glimmer, "configured", lambda: True)
    monkeypatch.setattr(glimmer, "first_guess", lambda lab, log=None: np.array(truth))
    c = Campaign(target, rig, rig.segmenter(), max_shots=6)
    run_deterministic(c)
    assert np.allclose(c.shots[0].recipe.as_array(), truth, atol=0.01)
    assert c.status == "matched" and len(c.shots) <= 2


def test_lab_runs_without_glimmer(monkeypatch):
    monkeypatch.setattr(glimmer, "configured", lambda: True)
    monkeypatch.setattr(glimmer, "first_guess", lambda lab, log=None: None)  # unreachable or unparsable
    rig = VirtualRig(seed=2, scene="kitchen_tube")
    c = Campaign(rig.true_swatch_lab(Recipe((0.4, 4.0, 1.0))), rig, rig.segmenter(), max_shots=8)
    run_deterministic(c)
    assert c.status == "matched"
