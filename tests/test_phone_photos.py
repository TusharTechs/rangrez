"""What phones and printers actually hand the lab: rotated 12 MP JPEGs, PNG screenshots,
iPhone HEIC, and a colour card printed on a home inkjet."""

import io

import numpy as np
import pytest
from PIL import Image
from pillow_heif import from_pillow

from rangrez.agent import run_deterministic
from rangrez.assisted import normalise_photo
from rangrez.campaign import Campaign
from rangrez.colour import delta_e_2000
from rangrez.rig import Recipe, execute
from rangrez.sim import VirtualRig
from rangrez.vision import measure


@pytest.fixture
def dyed():
    rig = VirtualRig(seed=4, scene="daylight_window")
    return rig, execute(rig, Recipe((0.4, 4.0, 1.0))).image_jpeg


def de(a, b):
    return float(delta_e_2000(a, b))


def test_sideways_12mp_phone_jpeg_is_turned_upright_and_shrunk(dyed):
    rig, jpeg = dyed
    big = Image.open(io.BytesIO(jpeg)).resize((4032, 2688))
    # Phones store the sensor's pixels sideways and an EXIF flag saying "rotate 90 degrees".
    sideways = big.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    sideways.save(buf, format="JPEG", quality=92, exif=exif.tobytes())
    upright = normalise_photo(buf.getvalue())
    img = Image.open(io.BytesIO(upright))
    assert img.size == (1920, 1280)
    m = measure(upright, rig.segmenter())
    assert de(m.lab, rig.last_swatch_lab) < 2.0


def test_png_screenshot_with_alpha(dyed):
    rig, jpeg = dyed
    buf = io.BytesIO()
    Image.open(io.BytesIO(jpeg)).convert("RGBA").save(buf, format="PNG")
    m = measure(normalise_photo(buf.getvalue()), rig.segmenter())
    assert de(m.lab, rig.last_swatch_lab) < 2.0


def test_iphone_heic(dyed):
    rig, jpeg = dyed
    buf = io.BytesIO()
    from_pillow(Image.open(io.BytesIO(jpeg))).save(buf, quality=90)
    m = measure(normalise_photo(buf.getvalue()), rig.segmenter())
    assert de(m.lab, rig.last_swatch_lab) < 2.0


def test_not_a_photo_is_refused_politely():
    with pytest.raises(ValueError, match="JPEG, PNG or HEIC"):
        normalise_photo(b"%PDF-1.4 not a photo")


def test_home_printed_card_shifts_colour_but_matching_stays_fair():
    """An inkjet misprints the card by a few dE00. The target and every dip are shot with
    the same misprinted card, so the comparison that decides a match stays fair."""
    for seed, (strip, target) in enumerate([((2.0, 1.0, 0.5), (2.5, 0.8, 0.6)), ((0.5, 3.0, 1.0), (0.5, 3.0, 2.0))]):
        rig = VirtualRig(seed=seed, scene="with_target", printed_card=True)
        rig.target_lab = rig.true_swatch_lab(Recipe(target))
        m = measure(execute(rig, Recipe(strip)).image_jpeg, rig.segmenter(), with_target=True)
        assert de(m.lab, rig.last_swatch_lab) < 4.5
        assert abs(de(m.lab, m.target_lab) - de(rig.last_swatch_lab, rig.target_lab)) < 2.0


def test_campaign_with_a_home_printed_card_truly_matches():
    rig = VirtualRig(seed=8, scene="with_target", printed_card=True)
    rig.target_lab = rig.true_swatch_lab(Recipe((2.2, 1.4, 0.6)))
    # The customer's cloth is photographed beside an undyed strip, with the same card.
    target = measure(rig.render(), rig.segmenter(), with_target=True).target_lab
    c = Campaign(target, rig, rig.segmenter(), tolerance=1.5, max_shots=8)
    run_deterministic(c)
    assert c.status == "matched", c.state()
    assert de(rig.last_swatch_lab, rig.target_lab) < 2.5
    assert np.isfinite(c.shots[-1].delta_e)
