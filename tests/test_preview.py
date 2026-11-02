"""The bulk preview is corrected on the garment, not on the bedsheet under it."""

import numpy as np

from rangrez import preview
from rangrez.colour import delta_e_2000
from rangrez.photoreal import Scene, SceneSegmenter, render
from rangrez.vision import CallLog


def test_preview_is_measured_and_corrected_on_the_garment(monkeypatch):
    scene = Scene.load("kurta_target")  # a kurta on paper on a bedsheet SAM also calls "fabric"
    dip = np.array([58.0, -22.0, -14.0])
    customer, masks = render(scene, target_lab=[60.0, -5.0, -15.0], card=False, seed=1)
    # An image model's recolour that came out noticeably off the requested shade.
    off, _ = render(scene, target_lab=[52.0, -30.0, -6.0], card=False, seed=1)
    monkeypatch.setattr(preview, "_edit", lambda photo, concept, hex_colour, log: off)
    out = preview.bulk_preview(customer, "kurta", dip, CallLog(), segmenter=SceneSegmenter(masks))
    assert out["raw_de00"] > 5
    assert out["de00"] < 1.0
    assert float(delta_e_2000(np.array([52.0, -30.0, -6.0]), dip)) > 5
