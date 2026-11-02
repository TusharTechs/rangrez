"""Lab memory: the lab gets better at this dye lot campaign after campaign."""

import numpy as np

from rangrez.agent import run_deterministic
from rangrez.campaign import Campaign
from rangrez.memory import LabMemory
from rangrez.rig import Recipe
from rangrez.sim import VirtualRig


def campaign(recipe, seed, memory):
    rig = VirtualRig(seed=seed, scene="kitchen_tube")
    c = Campaign(rig.true_swatch_lab(Recipe(recipe)), rig, rig.segmenter(), max_shots=8, memory=memory)
    run_deterministic(c)
    return c


def test_memory_persists_dips_and_campaigns(tmp_path):
    path = tmp_path / "memory.json"
    c = campaign((2.0, 1.0, 0.5), 1, LabMemory(path))
    again = LabMemory(path)
    assert len(again.dips) == len(c.shots)
    assert again.campaigns[-1]["status"] == c.status


def test_a_remembered_lot_starts_closer(tmp_path):
    memory = LabMemory(tmp_path / "memory.json")
    for i, r in enumerate([(3.0, 0.5, 0.5), (0.4, 4.0, 1.0), (0.2, 0.3, 5.0), (2.0, 2.0, 2.0), (1.5, 0.8, 3.0)]):
        campaign(r, 10 + i, memory)
    warm = campaign((2.2, 1.1, 0.6), 30, memory)
    cold = campaign((2.2, 1.1, 0.6), 30, None)
    assert warm.status == "matched"
    assert warm.shots[0].delta_e < cold.shots[0].delta_e  # the first dip lands closer
    assert len(warm.shots) <= len(cold.shots)
    assert any(n["who"] == "lab" for n in warm.notebook)


def test_nearest_remembered_dip(tmp_path):
    m = LabMemory(tmp_path / "m.json")
    m.record_dip([1, 0, 0], [60, -20, -10], "c1")
    m.record_dip([0, 1, 0], [55, 40, 0], "c1")
    conc, lab, de = m.nearest(np.array([56, 38, 1]))
    assert list(conc) == [0, 1, 0] and de < 3
